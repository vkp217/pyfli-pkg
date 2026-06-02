import warnings
import numpy as np
import torch
from scipy.optimize import curve_fit, minimize, OptimizeWarning
from scipy.signal import fftconvolve
from joblib import Parallel, delayed

# ── Suppress known non-fatal numerical warnings ───────────────────────────────
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in exp",            category=RuntimeWarning)
warnings.filterwarnings("ignore", message="overflow encountered in multiply",       category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in multiply",  category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide",    category=RuntimeWarning)
warnings.filterwarnings("ignore", message="divide by zero encountered",             category=RuntimeWarning)
# ─────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
#  p0 FORMAT  (the ONLY accepted format)
#  ─────────────────────────────────────
#  p0 is a list of (lb, ub) tuples, one per model parameter, in order.
#
#  Mono  [A, tau, B]:
#      p0 = [(0, np.inf), (0.6, 1.5), (0, np.inf)]
#
#  Bi    [A1, tau1, A2, tau2, B]:
#      p0 = [(0, np.inf), (0.1, 0.6), (0, np.inf), (0.6, 1.5), (0, np.inf)]
#
#  Rules
#  ─────
#  lb < ub  → free; start = geometric mean (or midpoint / numeric fallback)
#  lb == ub → CONSTANT; optimizer never moves it; chi² uses remaining dof
#  p0=None  → all free; numeric starts; physical defaults for bounds
#
#  weighted argument
#  ─────────────────
#  weighted=False (default):
#      FLIFitter           : minimise sum((y - yfit)²)         [unweighted LSQ]
#      PoissonLikelihood   : minimise Poisson NLL = sum(yfit - y*log(yfit))
#  weighted=True:
#      FLIFitter           : minimise sum(((y - yfit)/σᵢ)²)    [chi² weighted]
#                            σᵢ = sqrt(max(yᵢ, 1))  (Poisson counting noise)
#                            CPU: curve_fit(sigma=σ, absolute_sigma=True)
#                            GPU: W = diag(1/σᵢ) applied to residuals & Jacobian
#      PoissonLikelihood   : same Poisson NLL (naturally weighted; σ absorbed
#                            into the likelihood structure)
#
#  GPU equivalence guarantee
#  ─────────────────────────
#  Both GPU methods solve the SAME objective as their CPU counterparts:
#
#  FLIFitter GPU (weighted=False) : LSQ via batched LM
#    • W = identity  →  residual r = (d - pred),  J = d(pred)/d(θ)
#  FLIFitter GPU (weighted=True)  : χ² via batched LM
#    • W = diag(1/σᵢ) →  r = (d - pred)/σ,  J = d(pred)/d(θ)/σ
#
#  PoissonLikelihoodFitter GPU    : Poisson NLL via batched LM (no softplus)
#    • Score residual  r = (d - pred)/pred
#    • Score Jacobian  J_k = d(pred)/d(θk) / pred
#    • This is the exact gradient of Poisson NLL w.r.t. each parameter
#
#  All GPU paths share:
#    • Parameters in physical space (NO softplus reparameterisation)
#    • True hard bounds: clamp(p_new, lb, ub) after each accepted step
#    • Per-pixel adaptive damping with gain-ratio (ρ) trust-region
#    • τ₁ < τ₂ component-swap after every accepted step (bi-exponential only)
#    • Constant parameters: Jacobian column zeroed + value re-pinned each step
# ──────────────────────────────────────────────────────────────────────────────

class FLIFitter:
    """
    Non-Linear Least-Squares Fitter for Fluorescence Lifetime Imaging (FLIM).

    Parameters
    ----------
    decay         : ndarray (X, Y, T)
    irf           : ndarray (X, Y, T)
    fitting_model : 'mono-exponential' | 'bi-exponential'
    frequency     : float   laser repetition rate in MHz (default 80)
    solver        : 'lm' | 'trf'   scipy solver for CPU path (default 'lm')
    min_photons   : int     minimum photon count to attempt a fit (default 50)
    weighted      : bool
        False (default) — unweighted least-squares: min sum((y - ŷ)²)
        True            — chi²-weighted least-squares: min sum(((y - ŷ)/σ)²)
                          where σᵢ = sqrt(max(yᵢ, 1))
    """

    def __init__(self, decay, irf, fitting_model="mono-exponential",
                 frequency=80, solver="lm", min_photons=50, weighted=False):
        if decay.ndim != 3 or irf.ndim != 3:
            raise ValueError("decay and irf must be 3D arrays (X,Y,T)")
        self.decay       = decay.astype(np.float64)
        self.irf         = irf.astype(np.float64)
        self.model       = fitting_model.lower()
        self.solver      = solver.lower()
        self.min_photons = min_photons
        self.weighted    = bool(weighted)
        self.X, self.Y, self.T = self.decay.shape
        self.period      = 1e3 / frequency
        self.t           = np.linspace(0, self.period, self.T)

        # Normalize IRF pixel-wise
        self.irf /= (np.sum(self.irf, axis=2, keepdims=True) + 1e-12)

    # ──────────────────────────────────────────────────────────────────────────
    # SANITY CHECK
    # ──────────────────────────────────────────────────────────────────────────
    def _sanity_check_pixel(self, decay_pixel, irf_pixel, threshold=0.0):
        """Returns True (failed) when all decay or all IRF values ≤ threshold."""
        if np.all(decay_pixel <= threshold):
            return True
        if np.all(irf_pixel <= threshold):
            return True
        return False

    def _nan_pixel_result(self, pix_x, pix_y):
        n_params   = 3 if self.model == "mono-exponential" else 5
        nan_params = np.full(n_params, np.nan)
        nan_fitted = np.full(self.T, np.nan)
        return pix_x, pix_y, nan_params, nan_fitted, np.nan, np.nan, np.nan

    # ──────────────────────────────────────────────────────────────────────────
    # CONVOLVED MODEL
    # ──────────────────────────────────────────────────────────────────────────
    def _convolved_model(self, params, irf_pixel):
        if self.model == "mono-exponential":
            A, tau, B = params
            decay_model = A * np.exp(-self.t / (tau + 1e-12))
        else:
            A1, tau1, A2, tau2, B = params
            decay_model = (A1 * np.exp(-self.t / (tau1 + 1e-12)) +
                           A2 * np.exp(-self.t / (tau2 + 1e-12)))
        conv = fftconvolve(irf_pixel, decay_model, mode="full")[: self.T]
        return conv + params[-1]

    # ──────────────────────────────────────────────────────────────────────────
    # NUMERIC INITIAL GUESS
    # ──────────────────────────────────────────────────────────────────────────
    def _estimate_p0_numeric(self, decay_pixel):
        """
        Mathematically grounded per-pixel starting value estimator.

        Mono  → [A, tau, B]
          B   : 10th-percentile of decay (robust baseline)
          A   : peak of background-subtracted signal
          tau : intensity-weighted mean arrival time  (MLE for single-exp)

        Bi    → [A1, tau1, A2, tau2, B]
          tau1: weighted log-linear slope in the early window (5–35 % of T)
          tau2: weighted log-linear slope in the late  window (55–95 % of T)
          A1,A2: energy-fraction split of the peak amplitude
        """
        eps    = 1e-12
        d      = decay_pixel.copy()
        t      = self.t
        B_est  = max(float(np.percentile(d, 10)), 0.0)
        d_corr = np.clip(d - B_est, 0, None)
        total  = np.sum(d_corr) + eps

        if self.model == "mono-exponential":
            A_est   = max(float(np.max(d_corr)), eps)
            tau_est = float(np.sum(t * d_corr) / total)
            tau_est = float(np.clip(tau_est, 0.05 * self.period, 0.9 * self.period))
            return [A_est, tau_est, B_est]

        T    = len(t)
        e_lo = max(int(0.05 * T), 1);   e_hi = max(int(0.35 * T), e_lo + 2)
        l_lo = min(int(0.55 * T), T-3); l_hi = min(int(0.95 * T), T-1)

        def _log_slope_tau(lo, hi):
            seg   = np.clip(d_corr[lo:hi], eps, None)
            t_seg = t[lo:hi]
            w     = seg / (seg.sum() + eps)
            t_mu  = np.sum(w * t_seg)
            l_mu  = np.sum(w * np.log(seg))
            cov   = np.sum(w * (t_seg - t_mu) * (np.log(seg) - l_mu))
            var_t = np.sum(w * (t_seg - t_mu) ** 2) + eps
            slope = cov / var_t
            return float(np.clip(-1.0 / (slope - eps),
                                 0.05 * self.period, 0.9 * self.period))

        tau1_est = _log_slope_tau(e_lo, e_hi)
        tau2_est = _log_slope_tau(l_lo, l_hi)
        if tau1_est >= tau2_est:
            tau1_est = tau2_est * 0.4
        tau1_est = float(np.clip(tau1_est, 0.05 * self.period, tau2_est * 0.9))

        E_early  = float(np.sum(d_corr[e_lo:e_hi])) + eps
        E_late   = float(np.sum(d_corr[l_lo:l_hi]))  + eps
        A_peak   = max(float(np.max(d_corr)), eps)
        A1_est   = max((E_early / (E_early + E_late)) * A_peak, eps)
        A2_est   = max((E_late  / (E_early + E_late)) * A_peak, eps)

        return [A1_est, tau1_est, A2_est, tau2_est, B_est]

    # ──────────────────────────────────────────────────────────────────────────
    # P0 RANGE PARSING
    # ──────────────────────────────────────────────────────────────────────────
    def _parse_p0_ranges(self, p0_ranges, decay_pixel):
        """
        Parse list-of-(lb,ub) tuples.
        Returns (start, lb_vec, ub_vec, const_mask).

        Starting-value strategy per parameter
        ──────────────────────────────────────
        lb == ub              → constant;  start = lb
        lb < 0,  ub finite   → arithmetic midpoint  (lb+ub)/2
        lb < 0,  ub = inf    → 0.0
        lb = 0,  ub finite   → ub/2
        lb = 0,  ub = inf    → _estimate_p0_numeric fallback
        lb > 0,  ub finite   → geometric mean sqrt(lb*ub)
        lb > 0,  ub = inf    → lb * 2
        """
        n_params = 3 if self.model == "mono-exponential" else 5

        if p0_ranges is None:
            numeric = self._estimate_p0_numeric(decay_pixel)
            if self.model == "mono-exponential":
                lb_vec = [0.0,  0.01, 0.0]
                ub_vec = [np.inf, self.period, np.inf]
            else:
                lb_vec = [0.0, 0.01, 0.0, 0.01, 0.0]
                ub_vec = [np.inf, self.period, np.inf, self.period, np.inf]
            return numeric, lb_vec, ub_vec, [False] * n_params

        if len(p0_ranges) != n_params:
            raise ValueError(
                f"p0 must have {n_params} (lb, ub) tuples for {self.model}; "
                f"got {len(p0_ranges)}.")

        numeric    = self._estimate_p0_numeric(decay_pixel)
        start      = []
        lb_vec     = []
        ub_vec     = []
        const_mask = []

        for idx, (lb, ub) in enumerate(p0_ranges):
            lb = float(lb) if lb is not None else 0.0
            ub = float(ub) if ub is not None else np.inf

            if lb > ub:
                raise ValueError(f"p0[{idx}]: lower bound {lb} > upper bound {ub}.")

            lb_vec.append(lb)
            ub_vec.append(ub)

            if lb == ub:
                start.append(lb)
                const_mask.append(True)
                continue

            const_mask.append(False)
            ub_inf = np.isinf(ub)

            if lb < 0:
                s = 0.0 if ub_inf else (lb + ub) / 2.0
            elif lb == 0.0:
                s = float(numeric[idx]) if ub_inf else ub / 2.0
            else:
                s = lb * 2.0 if ub_inf else float(np.sqrt(lb * ub))

            eps_clip = 1e-9
            lo_clip  = lb + eps_clip if lb == 0.0 else lb
            hi_clip  = ub - eps_clip if not ub_inf else ub
            s = float(np.clip(s, lo_clip, hi_clip))
            start.append(s)

        return start, lb_vec, ub_vec, const_mask

    # ──────────────────────────────────────────────────────────────────────────
    # PHYSICAL DEFAULT BOUNDS
    # ──────────────────────────────────────────────────────────────────────────
    def _default_p0_ranges(self):
        """Physical-default p0_ranges (all free) for the current model."""
        if self.model == "mono-exponential":
            return [(0, np.inf), (0.01, self.period), (0, np.inf)]
        return [(0, np.inf), (0.01, self.period),
                (0, np.inf), (0.01, self.period),
                (0, np.inf)]

    # ──────────────────────────────────────────────────────────────────────────
    # STATS
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_stats(self, y, y_fit, n_free, sigma=None):
        """
        Compute residuals, R², RMSE, and reduced chi².

        Parameters
        ----------
        y      : 1-D array  observed data
        y_fit  : 1-D array  model prediction
        n_free : int         number of free parameters
        sigma  : 1-D array or None
            Per-bin uncertainty.  Only used when self.weighted=True.
            If None and weighted=True, defaults to sqrt(max(y, 1)).
        """
        residual = y - y_fit
        ss_res   = np.sum(residual ** 2)
        ss_tot   = np.sum((y - np.mean(y)) ** 2)
        r2       = 1 - ss_res / (ss_tot + 1e-12)
        rmse     = np.sqrt(ss_res / len(y))

        if self.weighted:
            _s   = sigma if sigma is not None else np.sqrt(np.maximum(y, 1.0))
            _s   = np.where(_s < 1e-12, 1e-12, _s)
            chi2 = np.sum((residual / _s) ** 2) / max(len(y) - n_free, 1)
        else:
            chi2 = ss_res / max(len(y) - n_free, 1)

        return residual, r2, rmse, chi2

    def _populate_param_maps(self, param_cube):
        params = {}
        if self.model == "mono-exponential":
            params["A_map"]      = param_cube[..., 0]
            params["tau_map"]    = param_cube[..., 1]
            params["Offset_map"] = param_cube[..., 2]
        else:
            A1   = param_cube[..., 0];  tau1 = param_cube[..., 1]
            A2   = param_cube[..., 2];  tau2 = param_cube[..., 3]
            params["A1_map"]       = A1
            params["tau1_map"]     = tau1
            params["A2_map"]       = A2
            params["tau2_map"]     = tau2
            params["Offset_map"]   = param_cube[..., 4]
            params["tau_mean_map"] = (A1 * tau1 + A2 * tau2) / (A1 + A2 + 1e-12)
        return params

    # ──────────────────────────────────────────────────────────────────────────
    # SINGLE PIXEL  (CPU / scipy)
    # ──────────────────────────────────────────────────────────────────────────
    def _single_pixel(self, pix_x, pix_y, p0=None, maxfev=2000):
        """
        Fit one pixel with NLSF (scipy curve_fit).

        weighted=False : standard unweighted least-squares
        weighted=True  : chi²-weighted via curve_fit(sigma=σ, absolute_sigma=True)
                         where σᵢ = sqrt(max(dᵢ, 1))
        """
        d     = self.decay[pix_x, pix_y]
        irf_p = self.irf[pix_x, pix_y]

        if self._sanity_check_pixel(d, irf_p):
            return self._nan_pixel_result(pix_x, pix_y)
        if np.sum(d) < self.min_photons:
            return None

        start, lb_vec, ub_vec, const_mask = self._parse_p0_ranges(p0, d)
        n_params = len(start)
        n_const  = sum(const_mask)
        n_free   = n_params - n_const

        free_idx  = [i for i, c in enumerate(const_mask) if not c]
        const_idx = [i for i, c in enumerate(const_mask) if c]  # noqa: F841

        def _expand(free_params):
            full = list(start)
            for k, fi in enumerate(free_idx):
                full[fi] = free_params[k]
            return full

        start_free = [start[i]  for i in free_idx]
        lb_free    = [lb_vec[i] for i in free_idx]
        ub_free    = [ub_vec[i] for i in free_idx]

        # Sigma vector for weighted fitting
        sigma = np.sqrt(np.maximum(d, 1.0)) if self.weighted else None

        # Build model function (operates in reduced free-parameter space)
        if self.model == "mono-exponential":
            if n_const == 0:
                func = lambda t, A, tau, B: \
                    self._convolved_model([A, tau, B], irf_p)
            elif n_free == 2:
                def func(t, *fp):
                    return self._convolved_model(_expand(list(fp)), irf_p)
            elif n_free == 1:
                def func(t, fp0):
                    return self._convolved_model(_expand([fp0]), irf_p)
            else:  # all constant
                fitted = self._convolved_model(start, irf_p)
                _, r2, rmse, chi2 = self._compute_stats(d, fitted, 0, sigma)
                return pix_x, pix_y, np.array(start), fitted, r2, rmse, chi2
        else:
            if n_const == 0:
                func = lambda t, A1, tau1, A2, tau2, B: \
                    self._convolved_model([A1, tau1, A2, tau2, B], irf_p)
            else:
                def func(t, *fp):
                    return self._convolved_model(_expand(list(fp)), irf_p)

        try:
            common_kw = dict(p0=start_free, maxfev=maxfev)
            if self.weighted:
                common_kw.update(sigma=sigma, absolute_sigma=True)

            if self.solver == "lm":
                popt_free, _ = curve_fit(func, self.t, d, **common_kw)
            else:
                popt_free, _ = curve_fit(func, self.t, d,
                                         bounds=(lb_free, ub_free),
                                         method="trf",
                                         **common_kw)

            popt   = np.array(_expand(list(popt_free)))
            fitted = self._convolved_model(popt, irf_p)
            _, r2, rmse, chi2 = self._compute_stats(d, fitted, n_free, sigma)
            return pix_x, pix_y, popt, fitted, r2, rmse, chi2

        except Exception:
            return None

    def fit_single_pixel(self, pix_x, pix_y, p0=None, maxfev=2000):
        """
        Fit one pixel and return parameter maps + diagnostics.

        Parameters
        ----------
        pix_x, pix_y : int
        p0    : list of (lb, ub) tuples | None
        maxfev: int  (default 2000)
        """
        out = self._single_pixel(pix_x, pix_y, p0=p0, maxfev=maxfev)
        if out:
            px, py, popt, fitted, R2, RMSE, chi2 = out
            return (self._populate_param_maps(popt),
                    fitted,
                    self.decay[px, py] - fitted,
                    {"R2_map": R2, "RMSE_map": RMSE, "chi2_map": chi2})
        return None

    def fit_entire_image_cpu(self, n_jobs=-1, p0=None, maxfev=2000,
                              progress_callback=None):
        """
        Fit the entire image on CPU (parallel row-by-row).

        Parameters
        ----------
        n_jobs : int   (-1 = all CPUs)
        p0     : list of (lb, ub) tuples | None
        maxfev : int   (default 2000)
        progress_callback : callable(pct, msg) | None
        """
        n_params    = 3 if self.model == "mono-exponential" else 5
        param_cube  = np.full((self.X, self.Y, n_params), np.nan)
        fitted_cube = np.full_like(self.decay, np.nan)
        R2   = np.full((self.X, self.Y), np.nan)
        RMSE = np.full((self.X, self.Y), np.nan)
        chi2 = np.full((self.X, self.Y), np.nan)

        def process(i, j):
            return self._single_pixel(i, j, p0=p0, maxfev=maxfev)

        for i in range(self.X):
            row_res = Parallel(n_jobs=n_jobs)(
                delayed(process)(i, j) for j in range(self.Y)
            )
            for r in row_res:
                if r:
                    idx_i, idx_j, popt, f, r2, rmse, c2 = r
                    param_cube[idx_i, idx_j]  = popt
                    fitted_cube[idx_i, idx_j] = f
                    R2[idx_i, idx_j]   = r2
                    RMSE[idx_i, idx_j] = rmse
                    chi2[idx_i, idx_j] = c2

            if progress_callback:
                progress_callback(int(((i + 1) / self.X) * 100),
                                  f"CPU Fitting Row {i+1}/{self.X}")

        return (self._populate_param_maps(param_cube),
                fitted_cube,
                self.decay - fitted_cube,
                {"R2_map": R2, "RMSE_map": RMSE, "chi2_map": chi2})

    # ──────────────────────────────────────────────────────────────────────────
    # GPU WHOLE-IMAGE — batched LM, solves the same LSQ as CPU
    # ──────────────────────────────────────────────────────────────────────────
    def fit_entire_image_gpu(
        self,
        n_iter=400,
        lambda_init=1e-2,
        tau_min=0.05,
        tau_max=10.0,
        tau_prior_mu=1.5,
        tau_prior_sigma=1.0,
        reg_weight=0,
        p0=None,
        progress_callback=None,
    ):
        """
        Fit the entire image on GPU using batched Levenberg-Marquardt.

        Solves the SAME objective as fit_entire_image_cpu:
          weighted=False → min sum((d - pred)²)                 [unweighted LSQ]
          weighted=True  → min sum(((d - pred)/σ)²)             [chi²-weighted]
                           where σᵢ = sqrt(max(dᵢ, 1))

        Implementation guarantees vs previous version
        ─────────────────────────────────────────────
        • NO softplus: parameters live in physical space throughout.
        • True hard bounds: clamp(p_new, lb, ub) after every accepted step.
        • Per-pixel adaptive damping with gain-ratio (ρ) trust-region.
        • τ₁ < τ₂ swap enforced after every accepted step (bi only).
        • Constant parameters: Jacobian column zeroed + value re-pinned.
        • λ clamped to [1e-10, 1e8] to prevent numerical blow-up.

        Parameters
        ----------
        n_iter          : int    LM iterations (default 400)
        lambda_init     : float  Initial damping factor (default 1e-2)
        tau_min,tau_max : float  Hard lifetime bounds (ns) when p0=None
        tau_prior_mu, tau_prior_sigma : float  Gaussian prior on lifetimes
        reg_weight      : float  Weight of the lifetime prior term
        p0              : list of (lb, ub) tuples | None
        progress_callback : callable(pct, msg) | None
        """
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_params = 3 if self.model == "mono-exponential" else 5
        eps      = 1e-8

        # ── Bounds & const_mask ───────────────────────────────────────────────
        _dummy = np.ones(self.T, dtype=np.float64)
        _, lb_vec, ub_vec, const_mask = self._parse_p0_ranges(p0, _dummy)

        if p0 is None:
            if self.model == "mono-exponential":
                lb_vec[1] = tau_min;  ub_vec[1] = tau_max
            else:
                lb_vec[1] = tau_min;  ub_vec[1] = tau_max
                lb_vec[3] = tau_min;  ub_vec[3] = tau_max

        const_vals_np = np.array([lb_vec[i] if const_mask[i] else 0.0
                                   for i in range(n_params)], dtype=np.float32)
        free_mask_np  = np.array([not c for c in const_mask], dtype=bool)

        # ── Data ─────────────────────────────────────────────────────────────
        decay_t  = torch.tensor(self.decay, dtype=torch.float32, device=device)
        irf_t    = torch.tensor(self.irf,   dtype=torch.float32, device=device)
        X, Y, T  = decay_t.shape
        Npix     = X * Y

        decay_t = decay_t.reshape(Npix, T)
        irf_t   = irf_t.reshape(Npix, T)

        t_gpu   = torch.tensor(self.t, dtype=torch.float32, device=device).unsqueeze(0)
        n_fft   = 2 * T
        IRF_fft = torch.fft.rfft(irf_t, n=n_fft)

        # ── Sanity mask ───────────────────────────────────────────────────────
        invalid_mask = (decay_t.sum(dim=1) <= 0) | (irf_t.sum(dim=1) <= 0)

        # ── Per-pixel weight vector W (Npix, T) ──────────────────────────────
        # weighted=True  : Wᵢ = 1 / σᵢ = 1 / sqrt(max(dᵢ, 1))
        # weighted=False : Wᵢ = 1
        if self.weighted:
            W_pix = 1.0 / torch.sqrt(torch.clamp(decay_t, min=1.0))  # (Npix, T)
        else:
            W_pix = torch.ones_like(decay_t)                          # (Npix, T)

        # ── Per-pixel initial guess ───────────────────────────────────────────
        decay_np = self.decay.astype(np.float32).reshape(Npix, T)
        init_np  = np.zeros((Npix, n_params), dtype=np.float32)

        for k in range(Npix):
            num = self._estimate_p0_numeric(decay_np[k])
            for i in range(n_params):
                if const_mask[i]:
                    init_np[k, i] = float(lb_vec[i])
                else:
                    lo = lb_vec[i];  hi = ub_vec[i]
                    init_np[k, i] = float(
                        np.clip(num[i],
                                lo + 1e-9 if lo == 0 else lo,
                                hi - 1e-9 if not np.isinf(hi) else hi))

        p   = torch.tensor(init_np, dtype=torch.float32, device=device)
        lam = torch.full((Npix, 1, 1), lambda_init, device=device)
        I   = torch.eye(n_params, device=device).unsqueeze(0)

        # ── Bounds tensors ────────────────────────────────────────────────────
        lb_t = torch.tensor([float(lb_vec[i]) for i in range(n_params)],
                              dtype=torch.float32, device=device
                              ).unsqueeze(0).expand(Npix, -1)
        # Replace +inf with large finite value for torch.min
        ub_np_clamped = [float(ub_vec[i]) if not np.isinf(ub_vec[i]) else 1e9
                         for i in range(n_params)]
        ub_t = torch.tensor(ub_np_clamped, dtype=torch.float32, device=device
                             ).unsqueeze(0).expand(Npix, -1)

        const_col   = torch.tensor(const_mask, dtype=torch.bool, device=device)
        const_val_t = torch.tensor(const_vals_np, dtype=torch.float32, device=device)

        # ── LM ITERATIONS ─────────────────────────────────────────────────────
        for it in range(n_iter):

            # Forward model + analytic Jacobian in physical space
            if self.model == "mono-exponential":
                A        = p[:, 0:1];  tau = p[:, 1:2]
                exp_term = torch.exp(-t_gpu / (tau + eps))
                m        = A * exp_term
                dA       = exp_term
                dtau     = A * (t_gpu / (tau ** 2 + eps)) * exp_term
                db       = torch.ones(Npix, T, device=device)
                deriv_list = [dA, dtau, db]
            else:
                A1   = p[:, 0:1];  tau1 = p[:, 1:2]
                A2   = p[:, 2:3];  tau2 = p[:, 3:4]
                exp1 = torch.exp(-t_gpu / (tau1 + eps))
                exp2 = torch.exp(-t_gpu / (tau2 + eps))
                m    = A1 * exp1 + A2 * exp2
                dA1   = exp1
                dtau1 = A1 * (t_gpu / (tau1 ** 2 + eps)) * exp1
                dA2   = exp2
                dtau2 = A2 * (t_gpu / (tau2 ** 2 + eps)) * exp2
                db    = torch.ones(Npix, T, device=device)
                deriv_list = [dA1, dtau1, dA2, dtau2, db]

            M_fft = torch.fft.rfft(m, n=n_fft)
            pred  = torch.fft.irfft(IRF_fft * M_fft, n=n_fft)[:, :T] + p[:, -1:]
            pred  = torch.clamp(pred, min=eps)

            # Weighted residuals:  r = W * (d - pred)
            r = (decay_t - pred) * W_pix                         # (Npix, T)

            # Weighted Jacobian:   J[:,t,k] = W_t * d(pred)/d(θ_k)
            J_list = []
            for dtheta in deriv_list:
                d_fft  = torch.fft.rfft(dtheta, n=n_fft)
                d_conv = torch.fft.irfft(IRF_fft * d_fft, n=n_fft)[:, :T]
                J_list.append(d_conv * W_pix)
            J = torch.stack(J_list, dim=2)                       # (Npix, T, n_params)

            # Zero Jacobian columns for constant parameters
            if const_col.any():
                J[:, :, const_col] = 0.0

            JT = J.transpose(1, 2)                               # (Npix, n_params, T)
            H  = JT @ J                                          # (Npix, n_params, n_params)
            g  = JT @ r.unsqueeze(-1)                            # (Npix, n_params, 1)

            # Bayesian lifetime prior (regularisation, optional)
            if self.model == "mono-exponential":
                if not const_mask[1]:
                    tp         = (p[:, 1] - tau_prior_mu) / (tau_prior_sigma ** 2)
                    H[:, 1, 1] += reg_weight / (tau_prior_sigma ** 2)
                    g[:, 1, 0] += tp
            else:
                if not const_mask[1]:
                    tp1        = (p[:, 1] - tau_prior_mu) / (tau_prior_sigma ** 2)
                    H[:, 1, 1] += reg_weight / (tau_prior_sigma ** 2)
                    g[:, 1, 0] += tp1
                if not const_mask[3]:
                    tp2        = (p[:, 3] - tau_prior_mu) / (tau_prior_sigma ** 2)
                    H[:, 3, 3] += reg_weight / (tau_prior_sigma ** 2)
                    g[:, 3, 0] += tp2

            # LM damping + linear solve
            H_lm  = H + lam * I
            # delta = torch.linalg.solve(H_lm, g).squeeze(-1)     # (Npix, n_params)
            try:
                delta = torch.linalg.solve(H_lm, g).squeeze(-1)
            except RuntimeError:
                # Fallback to least-squares solve for singular batches
                delta = torch.linalg.lstsq(H_lm, g).solution.squeeze(-1)
            p_new = p + delta

            # True hard bounds clamp
            p_new = torch.max(p_new, lb_t)
            p_new = torch.min(p_new, ub_t)

            # Re-pin constant parameters
            if const_col.any():
                p_new[:, const_col] = const_val_t[const_col].unsqueeze(0)

            # τ₁ < τ₂ ordering swap (bi-exponential only)
            if self.model != "mono-exponential":
                swap = p_new[:, 1] > p_new[:, 3]
                if swap.any():
                    A1c = p_new[:, 0].clone(); A2c = p_new[:, 2].clone()
                    p_new[swap, 0] = A2c[swap]; p_new[swap, 2] = A1c[swap]
                    t1c = p_new[:, 1].clone(); t2c = p_new[:, 3].clone()
                    p_new[swap, 1] = t2c[swap]; p_new[swap, 3] = t1c[swap]

            # ── Gain-ratio trust-region evaluation (per-pixel) ────────────────
            if self.model == "mono-exponential":
                m_new = p_new[:, 0:1] * torch.exp(-t_gpu / (p_new[:, 1:2] + eps))
            else:
                m_new = (p_new[:, 0:1] * torch.exp(-t_gpu / (p_new[:, 1:2] + eps)) +
                         p_new[:, 2:3] * torch.exp(-t_gpu / (p_new[:, 3:4] + eps)))

            M2_fft = torch.fft.rfft(m_new, n=n_fft)
            pred2  = torch.fft.irfft(IRF_fft * M2_fft, n=n_fft)[:, :T] + p_new[:, -1:]
            pred2  = torch.clamp(pred2, min=eps)
            r2_    = (decay_t - pred2) * W_pix

            loss      = torch.sum(r   ** 2, dim=1, keepdim=True)   # current loss
            loss2     = torch.sum(r2_ ** 2, dim=1, keepdim=True)   # proposed loss
            pred_gain = torch.sum(delta * g.squeeze(-1), dim=1, keepdim=True)
            rho       = (loss - loss2) / (pred_gain + eps)

            good       = (rho > 0.25).squeeze(-1)
            lam[good]  = torch.clamp(lam[good]  * 0.5, min=1e-10)
            lam[~good] = torch.clamp(lam[~good] * 2.0, max=1e8)
            p = torch.where(good.unsqueeze(-1), p_new, p)

            if progress_callback:
                progress_callback(int((it + 1) / n_iter * 100),
                                  f"GPU LM {it+1}/{n_iter}")

        # ── Final prediction ──────────────────────────────────────────────────
        with torch.no_grad():
            if self.model == "mono-exponential":
                m_fin = p[:, 0:1] * torch.exp(-t_gpu / (p[:, 1:2] + eps))
            else:
                m_fin = (p[:, 0:1] * torch.exp(-t_gpu / (p[:, 1:2] + eps)) +
                         p[:, 2:3] * torch.exp(-t_gpu / (p[:, 3:4] + eps)))
            M_fin  = torch.fft.rfft(m_fin, n=n_fft)
            pred_f = torch.fft.irfft(IRF_fft * M_fin, n=n_fft)[:, :T] + p[:, -1:]
            pred_f = torch.clamp(pred_f, min=eps)

        pred_final = pred_f.detach().cpu().numpy().reshape(X, Y, T)
        p_final    = p.detach().cpu().numpy().reshape(X, Y, n_params)

        invalid_np             = invalid_mask.cpu().numpy().reshape(X, Y)
        pred_final[invalid_np] = np.nan
        p_final[invalid_np]    = np.nan

        residual   = self.decay - pred_final
        n_free_gpu = int(np.sum(free_mask_np))

        ss_res = np.sum(residual ** 2, axis=2)
        ss_tot = np.sum((self.decay - np.mean(self.decay, axis=2, keepdims=True)) ** 2,
                        axis=2)
        R2   = 1 - ss_res / (ss_tot + 1e-12)
        RMSE = np.sqrt(ss_res / T)

        if self.weighted:
            sigma_img = np.sqrt(np.maximum(self.decay, 1.0))
            chi2 = np.sum((residual / sigma_img) ** 2, axis=2) / max(T - n_free_gpu, 1)
        else:
            chi2 = ss_res / max(T - n_free_gpu, 1)

        R2[invalid_np]   = np.nan
        RMSE[invalid_np] = np.nan
        chi2[invalid_np] = np.nan

        return (self._populate_param_maps(p_final), pred_final, residual,
                {"R2_map": R2, "RMSE_map": RMSE, "chi2_map": chi2})


# ══════════════════════════════════════════════════════════════════════════════
class PoissonLikelihoodFitter(FLIFitter):
    """
    Poisson Maximum-Likelihood Fitter for FLIM.

    CPU: minimises Poisson NLL = sum(ŷᵢ - yᵢ·log(ŷᵢ)) via L-BFGS-B.
         weighted=True additionally scales by 1/σᵢ² for API symmetry.

    GPU: minimises the same Poisson NLL via batched Levenberg-Marquardt,
         using the exact Poisson score residual and score Jacobian —
         mathematically equivalent to the CPU L-BFGS-B path.

    GPU vs old (softplus + L-BFGS) fixes
    ─────────────────────────────────────
    1. No softplus reparameterisation: parameters live in physical space.
       Eliminates curvature distortion, biased lifetimes, and slow Hessian.
    2. True hard bounds via clamp after each accepted LM step.
       Optimizer is always bound-aware (no post-projection).
    3. Per-pixel adaptive damping with gain-ratio trust-region (ρ).
    4. τ₁ < τ₂ ordering swap after every accepted step (bi only).
    5. Poisson score Jacobian:  J_ik = d(ŷᵢ)/d(θk) / ŷᵢ
       This is the exact partial derivative of Poisson NLL.
    """

    # ── CPU single pixel ──────────────────────────────────────────────────────
    def _single_pixel(self, pix_x, pix_y, p0=None, maxfev=2000):
        """
        Fit one pixel with Poisson MLE (scipy L-BFGS-B).

        weighted=False : Poisson NLL = sum(ŷ - y·log(ŷ))
        weighted=True  : weighted Poisson NLL = sum((ŷ - y·log(ŷ)) / σ²)
                         where σᵢ = sqrt(max(dᵢ, 1))
        """
        d     = self.decay[pix_x, pix_y]
        irf_p = self.irf[pix_x, pix_y]

        if self._sanity_check_pixel(d, irf_p):
            return self._nan_pixel_result(pix_x, pix_y)
        if np.sum(d) < self.min_photons:
            return None

        start, lb_vec, ub_vec, const_mask = self._parse_p0_ranges(p0, d)
        n_params = len(start)
        n_const  = sum(const_mask)
        n_free   = n_params - n_const

        free_idx  = [i for i, c in enumerate(const_mask) if not c]
        const_idx = [i for i, c in enumerate(const_mask) if c]  # noqa: F841

        def _expand(free_params):
            full = list(start)
            for k, fi in enumerate(free_idx):
                full[fi] = free_params[k]
            return full

        start_free  = [start[i]  for i in free_idx]
        lb_free     = [lb_vec[i] for i in free_idx]
        ub_free     = [ub_vec[i] for i in free_idx]
        lbfgsb_bnds = list(zip(lb_free,
                               [u if not np.isinf(u) else None for u in ub_free]))

        sigma = np.sqrt(np.maximum(d, 1.0)) if self.weighted else None

        if n_free == 0:  # all constant — just evaluate
            fitted = self._convolved_model(start, irf_p)
            _, r2, rmse, chi2 = self._compute_stats(d, fitted, 0, sigma)
            return pix_x, pix_y, np.array(start), fitted, r2, rmse, chi2

        def objective(fp):
            full = _expand(list(fp))
            m    = np.clip(self._convolved_model(full, irf_p), 1e-12, None)
            nll  = np.sum(m - d * np.log(m))
            if self.weighted:
                nll = nll / (sigma ** 2).sum()   # normalised weighted NLL
            return nll

        res = minimize(objective, start_free,
                       method="L-BFGS-B",
                       bounds=lbfgsb_bnds,
                       options={"maxfun": maxfev})

        if not res.success:
            return None

        popt   = np.array(_expand(list(res.x)))
        fitted = self._convolved_model(popt, irf_p)
        _, r2, rmse, chi2 = self._compute_stats(d, fitted, n_free, sigma)
        return pix_x, pix_y, popt, fitted, r2, rmse, chi2

    # ── CPU fit_single_pixel ──────────────────────────────────────────────────
    def fit_single_pixel(self, pix_x, pix_y, p0=None, maxfev=2000):
        """
        Fit one pixel with Poisson MLE and return maps + diagnostics.

        Parameters
        ----------
        pix_x, pix_y : int
        p0    : list of (lb, ub) tuples | None
        maxfev: int  (default 2000)
        """
        out = self._single_pixel(pix_x, pix_y, p0=p0, maxfev=maxfev)
        if out:
            px, py, popt, fitted, R2, RMSE, chi2 = out
            return (self._populate_param_maps(popt),
                    fitted,
                    self.decay[px, py] - fitted,
                    {"R2_map": R2, "RMSE_map": RMSE, "chi2_map": chi2})
        return None

    # ── CPU whole-image ───────────────────────────────────────────────────────
    def fit_entire_image_cpu(self, n_jobs=-1, p0=None, maxfev=2000,
                              progress_callback=None):
        """
        Fit the entire image on CPU with Poisson MLE.

        Parameters
        ----------
        n_jobs : int
        p0     : list of (lb, ub) tuples | None
        maxfev : int  (default 2000)
        progress_callback : callable(pct, msg) | None
        """
        n_params    = 3 if self.model == "mono-exponential" else 5
        param_cube  = np.full((self.X, self.Y, n_params), np.nan)
        fitted_cube = np.full_like(self.decay, np.nan)
        chi2_map    = np.full((self.X, self.Y), np.nan)
        R2_map      = np.full((self.X, self.Y), np.nan)
        RMSE_map    = np.full((self.X, self.Y), np.nan)

        def process_mle(i, j):
            return self._single_pixel(i, j, p0=p0, maxfev=maxfev)

        for i in range(self.X):
            row_res = Parallel(n_jobs=n_jobs)(
                delayed(process_mle)(i, j) for j in range(self.Y)
            )
            for r in row_res:
                if r:
                    idx_i, idx_j, popt, f, r2, rmse, c2 = r
                    param_cube[idx_i, idx_j]  = popt
                    fitted_cube[idx_i, idx_j] = f
                    R2_map[idx_i, idx_j]   = r2
                    RMSE_map[idx_i, idx_j] = rmse
                    chi2_map[idx_i, idx_j] = c2

            if progress_callback:
                progress_callback(int(((i + 1) / self.X) * 100),
                                  f"MLE CPU Row {i+1}/{self.X}")

        return (self._populate_param_maps(param_cube),
                fitted_cube,
                self.decay - fitted_cube,
                {"R2_map": R2_map, "RMSE_map": RMSE_map, "chi2_map": chi2_map})

    # ── GPU whole-image — Poisson LM, equivalent to CPU L-BFGS-B ─────────────
    def fit_entire_image_gpu(self, max_outer_iter=400, p0=None,
                              lambda_init=1e-2, progress_callback=None):
        """
        Fit the entire image on GPU using batched Levenberg-Marquardt
        minimising the Poisson NLL — the SAME objective as CPU L-BFGS-B.

        Poisson LM derivation
        ─────────────────────
        Poisson NLL: L = sum_i (ŷᵢ - yᵢ·log(ŷᵢ))
        Gradient:    ∂L/∂θk = sum_i (1 - yᵢ/ŷᵢ) · d(ŷᵢ)/d(θk)
                             = sum_i [(ŷᵢ - yᵢ)/ŷᵢ] · d(ŷᵢ)/d(θk)

        Define score residual:    rᵢ = (yᵢ - ŷᵢ) / ŷᵢ     (negative gradient direction)
        Define score Jacobian:    Jᵢk = d(ŷᵢ)/d(θk) / ŷᵢ
        Then:  JᵀJ·δ = Jᵀr  is the Gauss-Newton system for Poisson NLL.

        This is identical in form to weighted LSQ with Wᵢ = 1/ŷᵢ, which is
        the Fisher information weighting — the correct weighting for Poisson data.

        weighted=True additionally scales by 1/σᵢ² (σᵢ = sqrt(max(dᵢ,1))),
        equivalent to reweighting the Poisson score by observation noise.

        Parameters
        ----------
        max_outer_iter : int    LM iterations (default 400)
        p0             : list of (lb, ub) tuples | None
        lambda_init    : float  Initial damping (default 1e-2)
        progress_callback : callable(pct, msg) | None
        """
        device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_params = 3 if self.model == "mono-exponential" else 5
        eps      = 1e-8

        # ── Bounds & const_mask ───────────────────────────────────────────────
        _dummy = np.ones(self.T, dtype=np.float64)
        _, lb_vec, ub_vec, const_mask = self._parse_p0_ranges(p0, _dummy)

        const_vals_np = np.array([lb_vec[i] if const_mask[i] else 0.0
                                   for i in range(n_params)], dtype=np.float32)
        free_mask_np  = np.array([not c for c in const_mask], dtype=bool)

        # ── Flatten + valid-pixel mask ────────────────────────────────────────
        decay_np = self.decay.astype(np.float32)
        irf_np   = self.irf.astype(np.float32)

        decay_t  = torch.tensor(decay_np, device=device).reshape(-1, self.T)
        irf_t    = torch.tensor(irf_np,   device=device).reshape(-1, self.T)

        photon_sum     = decay_t.sum(dim=1)
        sanity_invalid = (decay_t.sum(dim=1) <= 0) | (irf_t.sum(dim=1) <= 0)
        mask           = (photon_sum >= self.min_photons) & (~sanity_invalid)
        valid_idx      = torch.where(mask)[0]

        if len(valid_idx) == 0:
            raise ValueError("No pixels above photon threshold.")

        decay_v  = decay_t[mask]                  # (Nv, T)
        irf_v    = irf_t[mask]
        n_pixels = decay_v.shape[0]

        # ── Per-pixel initial guess ───────────────────────────────────────────
        valid_decay_np = decay_np.reshape(-1, self.T)[valid_idx.cpu().numpy()]
        init_np = np.zeros((n_pixels, n_params), dtype=np.float32)

        for k in range(n_pixels):
            num = self._estimate_p0_numeric(valid_decay_np[k])
            for i in range(n_params):
                if const_mask[i]:
                    init_np[k, i] = float(lb_vec[i])
                else:
                    lo = lb_vec[i];  hi = ub_vec[i]
                    init_np[k, i] = float(
                        np.clip(num[i],
                                lo + 1e-9 if lo == 0 else lo,
                                hi - 1e-9 if not np.isinf(hi) else hi))

        p   = torch.tensor(init_np, dtype=torch.float32, device=device)
        lam = torch.full((n_pixels, 1, 1), lambda_init, device=device)
        I   = torch.eye(n_params, device=device).unsqueeze(0)

        # ── Bounds tensors ────────────────────────────────────────────────────
        lb_t = torch.tensor([float(lb_vec[i]) for i in range(n_params)],
                              dtype=torch.float32, device=device
                              ).unsqueeze(0).expand(n_pixels, -1)
        ub_np_clamped = [float(ub_vec[i]) if not np.isinf(ub_vec[i]) else 1e9
                         for i in range(n_params)]
        ub_t = torch.tensor(ub_np_clamped, dtype=torch.float32, device=device
                             ).unsqueeze(0).expand(n_pixels, -1)

        const_col   = torch.tensor(const_mask, dtype=torch.bool, device=device)
        const_val_t = torch.tensor(const_vals_np, dtype=torch.float32, device=device)

        # ── FFT setup ─────────────────────────────────────────────────────────
        t_gpu   = torch.tensor(self.t.astype(np.float32), device=device).unsqueeze(0)
        n_fft   = 2 * self.T
        IRF_fft = torch.fft.rfft(irf_v, n=n_fft)

        # Extra weighting for weighted=True: scale by 1/σᵢ² on top of Poisson
        if self.weighted:
            W_extra = 1.0 / torch.clamp(decay_v, min=1.0)  # (Nv, T)
        else:
            W_extra = torch.ones_like(decay_v)              # (Nv, T)

        # ── LM ITERATIONS ─────────────────────────────────────────────────────
        for it in range(max_outer_iter):

            # Forward model + derivatives in physical parameter space
            if self.model == "mono-exponential":
                A        = p[:, 0:1];  tau = p[:, 1:2]
                exp_term = torch.exp(-t_gpu / (tau + eps))
                m        = A * exp_term
                dA       = exp_term
                dtau     = A * (t_gpu / (tau ** 2 + eps)) * exp_term
                db       = torch.ones(n_pixels, self.T, device=device)
                deriv_list = [dA, dtau, db]
            else:
                A1   = p[:, 0:1];  tau1 = p[:, 1:2]
                A2   = p[:, 2:3];  tau2 = p[:, 3:4]
                exp1 = torch.exp(-t_gpu / (tau1 + eps))
                exp2 = torch.exp(-t_gpu / (tau2 + eps))
                m    = A1 * exp1 + A2 * exp2
                dA1   = exp1
                dtau1 = A1 * (t_gpu / (tau1 ** 2 + eps)) * exp1
                dA2   = exp2
                dtau2 = A2 * (t_gpu / (tau2 ** 2 + eps)) * exp2
                db    = torch.ones(n_pixels, self.T, device=device)
                deriv_list = [dA1, dtau1, dA2, dtau2, db]

            M_fft = torch.fft.rfft(m, n=n_fft)
            pred  = torch.fft.irfft(IRF_fft * M_fft, n=n_fft)[:, :self.T] + p[:, -1:]
            pred  = torch.clamp(pred, min=eps)

            # Poisson score residual:  rᵢ = (yᵢ - ŷᵢ) / ŷᵢ  × W_extra
            inv_pred = 1.0 / pred
            r        = (decay_v - pred) * inv_pred * W_extra        # (Nv, T)

            # Poisson score Jacobian:  Jᵢk = d(ŷᵢ)/d(θk) / ŷᵢ × W_extra
            J_list = []
            for dtheta in deriv_list:
                d_fft  = torch.fft.rfft(dtheta, n=n_fft)
                d_conv = torch.fft.irfft(IRF_fft * d_fft, n=n_fft)[:, :self.T]
                J_list.append(d_conv * inv_pred * W_extra)
            J = torch.stack(J_list, dim=2)                           # (Nv, T, n_params)

            if const_col.any():
                J[:, :, const_col] = 0.0

            JT = J.transpose(1, 2)
            H  = JT @ J
            g  = JT @ r.unsqueeze(-1)

            # LM damping + solve
            H_lm  = H + lam * I
            delta = torch.linalg.solve(H_lm, g).squeeze(-1)
            p_new = p + delta

            # True hard bounds clamp
            p_new = torch.max(p_new, lb_t)
            p_new = torch.min(p_new, ub_t)

            # Re-pin constants
            if const_col.any():
                p_new[:, const_col] = const_val_t[const_col].unsqueeze(0)

            # τ₁ < τ₂ ordering swap (bi-exponential only)
            if self.model != "mono-exponential":
                swap = p_new[:, 1] > p_new[:, 3]
                if swap.any():
                    A1c = p_new[:, 0].clone(); A2c = p_new[:, 2].clone()
                    p_new[swap, 0] = A2c[swap]; p_new[swap, 2] = A1c[swap]
                    t1c = p_new[:, 1].clone(); t2c = p_new[:, 3].clone()
                    p_new[swap, 1] = t2c[swap]; p_new[swap, 3] = t1c[swap]

            # ── Gain-ratio trust-region evaluation ───────────────────────────
            if self.model == "mono-exponential":
                m_new = p_new[:, 0:1] * torch.exp(-t_gpu / (p_new[:, 1:2] + eps))
            else:
                m_new = (p_new[:, 0:1] * torch.exp(-t_gpu / (p_new[:, 1:2] + eps)) +
                         p_new[:, 2:3] * torch.exp(-t_gpu / (p_new[:, 3:4] + eps)))

            M2_fft = torch.fft.rfft(m_new, n=n_fft)
            pred2  = torch.fft.irfft(IRF_fft * M2_fft, n=n_fft)[:, :self.T] + p_new[:, -1:]
            pred2  = torch.clamp(pred2, min=eps)

            inv_p2 = 1.0 / pred2
            r2_    = (decay_v - pred2) * inv_p2 * W_extra

            loss      = torch.sum(r   ** 2, dim=1, keepdim=True)
            loss2     = torch.sum(r2_ ** 2, dim=1, keepdim=True)
            pred_gain = torch.sum(delta * g.squeeze(-1), dim=1, keepdim=True)
            rho       = (loss - loss2) / (pred_gain + eps)

            good       = (rho > 0.25).squeeze(-1)
            lam[good]  = torch.clamp(lam[good]  * 0.5, min=1e-10)
            lam[~good] = torch.clamp(lam[~good] * 2.0, max=1e8)
            p = torch.where(good.unsqueeze(-1), p_new, p)

            if progress_callback:
                progress_callback(int((it + 1) / max_outer_iter * 100),
                                  f"GPU Poisson LM {it+1}/{max_outer_iter}")

        # ── Final prediction ──────────────────────────────────────────────────
        with torch.no_grad():
            if self.model == "mono-exponential":
                m_fin = p[:, 0:1] * torch.exp(-t_gpu / (p[:, 1:2] + eps))
            else:
                m_fin = (p[:, 0:1] * torch.exp(-t_gpu / (p[:, 1:2] + eps)) +
                         p[:, 2:3] * torch.exp(-t_gpu / (p[:, 3:4] + eps)))
            M_fin  = torch.fft.rfft(m_fin, n=n_fft)
            pred_f = torch.fft.irfft(IRF_fft * M_fin, n=n_fft)[:, :self.T] + p[:, -1:]
            pred_f = torch.clamp(pred_f, min=eps)

        # ── Scatter back to full image ────────────────────────────────────────
        p_full      = torch.full((self.X * self.Y, n_params), float("nan"), device=device)
        fitted_full = torch.full((self.X * self.Y, self.T),   float("nan"), device=device)
        p_full[valid_idx]      = p
        fitted_full[valid_idx] = pred_f

        p_np      = p_full.reshape(self.X, self.Y, n_params).detach().cpu().numpy()
        fitted_np = fitted_full.reshape(self.X, self.Y, self.T).detach().cpu().numpy()

        residual   = self.decay - fitted_np
        n_free_gpu = int(np.sum(free_mask_np))

        ss_res = np.sum(residual ** 2, axis=2)
        ss_tot = np.sum((self.decay - np.mean(self.decay, axis=2, keepdims=True)) ** 2,
                        axis=2)
        R2_map   = 1 - ss_res / (ss_tot + 1e-12)
        RMSE_map = np.sqrt(ss_res / self.T)

        if self.weighted:
            sigma_img = np.sqrt(np.maximum(self.decay, 1.0))
            chi2_map  = (np.sum((residual / sigma_img) ** 2, axis=2)
                         / max(self.T - n_free_gpu, 1))
        else:
            # Poisson chi²: sum((d - ŷ)² / ŷ)
            chi2_map = (np.sum((self.decay - fitted_np) ** 2 / (fitted_np + 1e-12), axis=2)
                        / max(self.T - n_free_gpu, 1))

        sanity_np = sanity_invalid.cpu().numpy().reshape(self.X, self.Y)
        R2_map[sanity_np]   = np.nan
        RMSE_map[sanity_np] = np.nan
        chi2_map[sanity_np] = np.nan

        return (self._populate_param_maps(p_np), fitted_np, residual,
                {"R2_map": R2_map, "RMSE_map": RMSE_map, "chi2_map": chi2_map})

    # ── Poisson deviance map ──────────────────────────────────────────────────
    def _poisson_deviance_map(self, decay, fitted, n_params):
        """
        Compute reduced Poisson deviance per pixel.
        D = 2 · sum( d·log(d/ŷ) − (d − ŷ) ),  reduced by dof = T − n_params.
        """
        eps  = 1e-12
        d    = decay;  m = fitted + eps
        term = np.zeros_like(d)
        nz   = d > 0
        term[nz] = d[nz] * np.log(d[nz] / m[nz])
        return 2 * np.sum(term - (d - m), axis=2) / max(self.T - n_params, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  FLIAnalysisSuite  — high-level convenience wrapper
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

class FLIAnalysisSuite:
    """
    High-level wrapper orchestrating FLIFitter and PoissonLikelihoodFitter.

    Parameters
    ----------
    decay, irf    : ndarray (X, Y, T)
    frequency     : float  laser repetition rate MHz (default 80)
    min_photons   : int    minimum photon count per pixel (default 50)
    weighted      : bool   passed to both fitter classes (default False)
                    False → unweighted LSQ / standard Poisson NLL
                    True  → chi²-weighted LSQ / weighted Poisson NLL
    """

    def __init__(self, decay, irf, frequency=80, min_photons=50, weighted=False):
        self.decay    = decay
        self.irf      = irf
        self.freq     = frequency
        self.min_p    = min_photons
        self.weighted = bool(weighted)
        self.last_results = None

        self.mp   = ["A_map", "tau_map", "Offset_map"]
        self.bp   = ["A1_map", "tau1_map", "A2_map", "tau2_map",
                     "tau_mean_map", "Offset_map"]
        self.st   = ["R2_map", "RMSE_map", "chi2_map"]
        self.keys = {"mono": self.mp, "bi": self.bp, "st": self.st}

    def run_analysis(self, model_type="mono-exponential", fitting_method="both",
                     device="cpu", analysis="single pixel", px=(0, 0), p0=None,
                     maxfev=2000):
        """
        Parameters
        ----------
        model_type     : 'mono-exponential' | 'bi-exponential' | 'both'
        fitting_method : 'nlsf' | 'mle' | 'both'
        device         : 'cpu' | 'gpu' | 'both'  ('both' only for whole-image)
        analysis       : 'single pixel' | 'whole'
        px             : (row, col) for single-pixel mode
        p0             : list of (lb, ub) tuples | None
        maxfev         : int  Max function evaluations (CPU, default 2000)
        """
        models  = (["mono-exponential", "bi-exponential"]
                   if model_type == "both" else [model_type])
        methods = (["nlsf", "mle"]
                   if fitting_method.lower() == "both"
                   else [fitting_method.lower()])
        devices = (["cpu", "gpu"]
                   if (device == "both" and analysis == "whole")
                   else [device])

        results = {}
        for m in models:
            results[m] = {}
            f_inst = {
                "nlsf": FLIFitter(self.decay, self.irf, m,
                                   self.freq, "lm", self.min_p, self.weighted),
                "mle":  PoissonLikelihoodFitter(self.decay, self.irf, m,
                                                self.freq, "lm", self.min_p,
                                                self.weighted),
            }
            for dev in devices:
                results[m][dev] = {}
                for meth in methods:
                    fitter = f_inst[meth]
                    if analysis == "single pixel":
                        res = fitter.fit_single_pixel(px[0], px[1],
                                                       p0=p0, maxfev=maxfev)
                        if res is None:
                            total_photons = int(np.sum(self.decay[px[0], px[1]]))
                            print(f"  [{meth.upper()} {m}] Pixel {px} skipped — "
                                  f"only {total_photons} photons "
                                  f"(min_photons={self.min_p}). "
                                  f"Lower min_photons or choose a brighter pixel.")
                            T_    = self.decay.shape[2]
                            nan1d = np.full(T_, np.nan)
                            if "mono" in m:
                                nan_params = {"A_map": np.nan, "tau_map": np.nan,
                                              "Offset_map": np.nan}
                            else:
                                nan_params = {"A1_map": np.nan, "tau1_map": np.nan,
                                              "A2_map": np.nan, "tau2_map": np.nan,
                                              "tau_mean_map": np.nan, "Offset_map": np.nan}
                            nan_stats = {"R2_map": np.nan, "RMSE_map": np.nan,
                                         "chi2_map": np.nan}
                            res = (nan_params, nan1d, nan1d, nan_stats)
                    elif dev == "cpu":
                        res = fitter.fit_entire_image_cpu(p0=p0, maxfev=maxfev)
                    else:
                        res = fitter.fit_entire_image_gpu(p0=p0)

                    results[m][dev][meth] = {
                        "params":    res[0], "fitted":    res[1],
                        "residuals": res[2], "stats":     res[3],
                    }

        self.last_results = results
        self._display(results, analysis, px, fitting_method)
        return results

    def _display(self, results, analysis, px, fitting_method):
        for m_type, dev_dict in results.items():
            header = ("MONO-EXPONENTIAL" if "mono" in m_type else "BI-EXPONENTIAL")
            print(f"\n--- {header} RESULTS ---")

            for dev, fit_dict in dev_dict.items():
                for meth in fit_dict:
                    res  = fit_dict[meth]
                    p, s = res["params"], res["stats"]
                    idx  = (px[0], px[1]) if analysis == "whole" else ...

                    def get_val(data, key):
                        return data[key][idx] if analysis == "whole" else data[key]

                    def fmt(v):
                        try:    return f"{float(v):.3f}"
                        except: return "NaN"

                    if "mono" in m_type:
                        print(f"  [{dev.upper()}] {meth.upper()} Mono: "
                              f"A={fmt(get_val(p, self.mp[0]))}  "
                              f"tau={fmt(get_val(p, self.mp[1]))}  "
                              f"offset={fmt(get_val(p, self.mp[2]))}  "
                              f"R2={fmt(get_val(s, self.st[0]))}  "
                              f"chi2={fmt(get_val(s, self.st[2]))}")
                    else:
                        print(f"  [{dev.upper()}] {meth.upper()} Bi: "
                              f"A1={fmt(get_val(p, self.bp[0]))}  "
                              f"tau1={fmt(get_val(p, self.bp[1]))}  "
                              f"A2={fmt(get_val(p, self.bp[2]))}  "
                              f"tau2={fmt(get_val(p, self.bp[3]))}  "
                              f"tau_mean={fmt(get_val(p, self.bp[4]))}  "
                              f"R2={fmt(get_val(s, self.st[0]))}  "
                              f"chi2={fmt(get_val(s, self.st[2]))}")

            # Plot: one block per model, outside the dev loop
            if analysis == "single pixel":
                first_fit_dict = next(iter(dev_dict.values()))
                pixel_data = {
                    meth: {**fit_result,
                           "raw_decay": self.decay[px[0], px[1], :],
                           "params":    fit_result["params"],
                           "stats":     fit_result["stats"]}
                    for meth, fit_result in first_fit_dict.items()
                }
                self._plot_pixel_logic(pixel_data, m_type, fitting_method, px)
            else:
                for dev, fit_dict in dev_dict.items():
                    self._plot_whole(fit_dict, m_type, dev)

    def _draw_decay_fit(self, ax, data, title, m_type):
        raw = data["raw_decay"]
        fit = data["fitted"]
        res = data["residuals"]

        ax.plot(raw, "k.", alpha=0.3, label="Decay")
        plot_irf = self.irf[0, 0, :] if self.irf.ndim == 3 else self.irf
        ax.plot(plot_irf, "g--", alpha=0.5, label="IRF")

        if not np.all(np.isnan(fit)):
            ax.plot(fit, "r-", label="Fit",       linewidth=1.5)
            ax.plot(res, "b-", alpha=0.6, label="Residuals")
        else:
            ax.text(0.5, 0.5,
                    "Fit failed\n(insufficient photons or convergence error)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10, color="red",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

        param_key = "mono" if "mono" in m_type else "bi"
        text_lines = []
        for k in self.keys[param_key]:
            v = data["params"].get(k, np.nan)
            try:    text_lines.append(f"{k.split('_')[0]}: {float(v):.3f}")
            except: text_lines.append(f"{k.split('_')[0]}: NaN")
        for k in self.st:
            v = data["stats"].get(k, np.nan)
            try:    text_lines.append(f"{k.split('_')[0]}: {float(v):.4f}")
            except: text_lines.append(f"{k.split('_')[0]}: NaN")

        ax.text(1.05, 0.95, "\n".join(text_lines),
                transform=ax.transAxes, verticalalignment="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize="x-small")

    def _plot_pixel_logic(self, fit_dict, m_type, method_choice, px):
        if method_choice.lower() == "both":
            fig, axes = plt.subplots(1, 3, figsize=(20, 5))
            self._draw_decay_fit(axes[0], fit_dict["nlsf"], f"NLSF: {m_type}", m_type)
            self._draw_decay_fit(axes[1], fit_dict["mle"],  f"MLE: {m_type}",  m_type)
            nlsf_fit = fit_dict["nlsf"]["fitted"]
            mle_fit  = fit_dict["mle"]["fitted"]
            if not np.all(np.isnan(nlsf_fit)):
                axes[2].plot(nlsf_fit, label="NLSF Fit")
            if not np.all(np.isnan(mle_fit)):
                axes[2].plot(mle_fit, "--", label="MLE Fit")
            axes[2].set_title(f"Comparison at {px}")
            axes[2].legend()
        else:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            meth = method_choice.lower()
            self._draw_decay_fit(ax, fit_dict[meth],
                                 f"{meth.upper()}: {m_type}", m_type)
        fig.tight_layout()
        plt.show()
        return fig

    def check_random_pixel(self, px=(0, 0)):
        """
        Re-plot a pixel from the most recent run_analysis call.
        Works for both 'single pixel' and 'whole' analysis modes.
        """
        if self.last_results is None:
            print("Error: Run analysis first (single pixel or whole image).")
            return

        print(f"\n--- PIXEL CHECK AT {px} ---")
        for m_type, dev_dict in self.last_results.items():
            for dev, fit_dict in dev_dict.items():
                methods_available = list(fit_dict.keys())
                method_choice     = ("both" if len(methods_available) > 1
                                     else methods_available[0])

                sample_fitted = fit_dict[methods_available[0]]["fitted"]
                is_whole = (sample_fitted.ndim == 3)

                pixel_data = {}
                for meth in methods_available:
                    r = fit_dict[meth]
                    if is_whole:
                        pixel_data[meth] = {
                            "raw_decay": self.decay[px[0], px[1], :],
                            "fitted":    r["fitted"][px[0], px[1], :],
                            "residuals": r["residuals"][px[0], px[1], :],
                            "params":    {k: v[px[0], px[1]]
                                          for k, v in r["params"].items()},
                            "stats":     {k: v[px[0], px[1]]
                                          for k, v in r["stats"].items()},
                        }
                    else:
                        pixel_data[meth] = {
                            "raw_decay": self.decay[px[0], px[1], :],
                            "fitted":    r["fitted"],
                            "residuals": r["residuals"],
                            "params":    r["params"],
                            "stats":     r["stats"],
                        }

                self._plot_pixel_logic(pixel_data, m_type, method_choice, px)

    def _plot_whole(self, fit_dict, m_type, dev):
        m_key    = "mono" if "mono" in m_type else "bi"
        all_keys = self.keys[m_key] + self.st
        methods  = list(fit_dict.keys())
        n_rows   = len(methods)
        n_cols   = len(all_keys)

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(3.2 * n_cols, 3.5 * n_rows))
        axes = np.array(axes)
        if axes.ndim == 1:
            axes = axes.reshape(1, -1) if n_rows == 1 else axes.reshape(-1, 1)
        if axes.ndim == 0:
            axes = axes.reshape(1, 1)

        header = "MONO" if "mono" in m_type else "BI"
        w_tag  = " | weighted=True" if self.weighted else " | weighted=False"
        fig.suptitle(f"{header}-EXPONENTIAL  |  device: {dev.upper()}{w_tag}",
                     fontsize=13, fontweight="bold")

        for i, meth in enumerate(methods):
            combined = {**fit_dict[meth]["params"], **fit_dict[meth]["stats"]}
            for j, k in enumerate(all_keys):
                ax  = axes[i, j]
                img = combined.get(k, None)

                is_valid = (img is not None
                            and isinstance(img, np.ndarray)
                            and img.ndim == 2
                            and np.any(np.isfinite(img)))

                if not is_valid:
                    ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                            ha="center", va="center", fontsize=10, color="grey")
                    ax.set_facecolor("#eeeeee")
                else:
                    finite_vals = img[np.isfinite(img)]
                    vmin = float(np.percentile(finite_vals, 2))
                    vmax = float(np.percentile(finite_vals, 98))
                    if vmin == vmax:
                        vmin -= 0.01; vmax += 0.01
                    im = ax.imshow(img, cmap="jet", vmin=vmin, vmax=vmax)
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                if i == 0:
                    ax.set_title(k.replace("_map", ""), fontsize=9)
                if j == 0:
                    ax.set_ylabel(meth.upper(), fontsize=11, fontweight="bold")
                ax.axis("off")

        fig.tight_layout()
        plt.show()
        return fig

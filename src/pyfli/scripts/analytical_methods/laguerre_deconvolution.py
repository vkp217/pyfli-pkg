from __future__ import annotations
from typing import Optional
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, nnls
from scipy.signal import lfilter, fftconvolve


class LaguerreFLI:
    """
    Laguerre-expansion FLIM deconvolution.

    Pipeline:
      1. Expand the IRF-free fluorescence decay h(n) on an orthonormal discrete
         Laguerre basis {b_j}. The measured decay is h convolved with the IRF, so
         the design matrix is V = IRF * B and we solve  V c = y  (linear).
      2. Reconstruct h(n) = sum_j c_j b_j(n).
      3. Fit N exponentials to the reconstructed (deconvolved) decay to read off
         lifetimes and amplitudes.

    Notes on the math (vs. the previous version):
      * Laguerre coefficients c_j are NOT sign-constrained. The basis functions
        themselves oscillate, and the expansion of a positive decay generically
        has alternating-sign coefficients. They are solved by ordinary least
        squares. (Forcing c_j >= 0 is incorrect and destroys the fit.)
      * Non-negativity is enforced only where it is physically meaningful: the
        exponential AMPLITUDES in step 3 (NNLS), which must be >= 0.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_laguerre: Optional[int] = None,
        alpha: float = 0.85,
        dt: float = 1.0,
        auto_alpha: bool = False,
        taus_init: Optional[np.ndarray] = None,
        laser_period_ns: Optional[float] = None,
        reg_strength: float = 0.0,
        reg_power: float = 2.0,
    ):
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly in (0, 1).")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if laser_period_ns is not None and laser_period_ns <= 0:
            raise ValueError("laser_period_ns must be positive.")

        self.n_components = int(n_components)
        self.n_laguerre = int(n_laguerre) if n_laguerre is not None else max(4, 2 * n_components)
        if self.n_laguerre < self.n_components:
            raise ValueError("n_laguerre must be >= n_components.")

        self.alpha           = float(alpha)
        self.dt              = float(dt)
        self.auto_alpha      = bool(auto_alpha)
        self.laser_period_ns = float(laser_period_ns) if laser_period_ns is not None else None
        self.taus_init       = np.asarray(taus_init, float) if taus_init is not None else None
        self.reg_strength = float(reg_strength)   # 0 => pure OLS (mathematically exact)
        self.reg_power    = float(reg_power)       # >0 => penalise high Laguerre orders more

        self.basis_:          Optional[np.ndarray] = None
        self.V_:              Optional[np.ndarray] = None
        self.coeffs_:         Optional[np.ndarray] = None
        self.taus_:           Optional[np.ndarray] = None
        self.n_unique_irf_:   Optional[int]         = None
        self.amplitudes_:     Optional[np.ndarray] = None
        self.fractions_:      Optional[np.ndarray] = None
        self.tau_mean_:       Optional[np.ndarray] = None
        self.reconstructed_:  Optional[np.ndarray] = None
        self.residuals_:      Optional[np.ndarray] = None
        self.fit_curve_:      Optional[np.ndarray] = None
        self.residual_curve_: Optional[np.ndarray] = None
        self.decay_:          Optional[np.ndarray] = None

    # ---- basis -------------------------------------------------------------
    @staticmethod
    def _discrete_laguerre_basis(T: int, alpha: float, L: int) -> np.ndarray:
        """
        Orthonormal discrete Laguerre functions, (L, T).
        Recurrence (pole = sqrt(alpha)):
            b_0(n) = sqrt(1-a) a^{n/2}
            b_j(n) = sqrt(a) b_j(n-1) + sqrt(a) b_{j-1}(n) - b_{j-1}(n-1)
        Inner time recurrence vectorised with an IIR filter (lfilter).
        """
        b = np.zeros((L, T), dtype=np.float64)
        n = np.arange(T)
        b[0] = np.sqrt(1.0 - alpha) * alpha ** (n / 2.0)
        sa = np.sqrt(alpha)
        a_coef = [1.0, -sa]
        for j in range(1, L):
            prev = b[j - 1]
            shifted = np.empty_like(prev)      # b_{j-1}(n-1)
            shifted[0] = 0.0
            shifted[1:] = prev[:-1]
            u = sa * prev - shifted            # sqrt(a) b_{j-1}(n) - b_{j-1}(n-1)
            b[j] = lfilter([1.0], a_coef, u)   # apply 1/(1 - sqrt(a) z^-1)
        return b

    @staticmethod
    def _convolve_with_irf(basis: np.ndarray, irf: np.ndarray) -> np.ndarray:
        """(T, L) IRF-convolved basis, causally truncated. Vectorised over order."""
        L, T = basis.shape
        irf = np.asarray(irf, float).ravel()
        s = irf.sum()
        if s > 0:
            irf = irf / s
        full = fftconvolve(basis, irf[None, :], mode="full", axes=1)
        return full[:, :T].T

    @staticmethod
    def _unique_irf_groups(irf_2d: np.ndarray, decimals: int = 6):
        """
        Group pixels by their IRF *shape* so a design matrix can be built once
        per distinct IRF and reused across the group.

        IRFs are compared after sum-normalisation (the convolution normalises
        anyway, so two IRFs that differ only by a scale factor are equivalent),
        and quantised to `decimals` places to absorb floating-point noise.
        Empty/all-zero IRF pixels (e.g. masked) are pooled into one group.

        Returns
        -------
        labels   : (P,) int array, group id per pixel.
        rep_idx  : list of pixel indices, one representative per group.
        """
        P, T = irf_2d.shape
        s = irf_2d.sum(axis=1, keepdims=True)
        norm = np.divide(irf_2d, s, out=np.zeros_like(irf_2d), where=s > 0)
        keys = np.round(norm, decimals)
        _, first_idx, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        inverse = inverse.ravel()
        rep_idx = [int(np.flatnonzero(inverse == g)[0]) for g in range(len(first_idx))]
        return inverse, rep_idx

    # ---- coefficient solve (ordinary least squares; sign-unconstrained) ----
    def _penalty(self, L: int) -> np.ndarray:
        """Diagonal Tikhonov weights, graded by Laguerre order (0,1,..,L-1).
        Higher orders oscillate fastest and absorb tail noise, so they are
        penalised more strongly."""
        return (np.arange(L, dtype=float) + 1.0) ** self.reg_power

    def _solve_coefficients(self, V: np.ndarray, Y2d: np.ndarray) -> np.ndarray:
        """Solve V C = Y2d for all pixels at once (sign-unconstrained). (L, P)."""
        if self.reg_strength > 0.0:
            L = V.shape[1]
            lam = self.reg_strength * float(np.mean(np.diag(V.T @ V)))
            VtV = V.T @ V + lam * np.diag(self._penalty(L))
            return np.linalg.solve(VtV, V.T @ Y2d)
        C, *_ = np.linalg.lstsq(V, Y2d, rcond=None)
        return C

    def _optimize_alpha(self, avg_decay: np.ndarray, avg_irf: np.ndarray, T: int) -> float:
        """
        Choose the Laguerre pole alpha on the spatially averaged decay by
        minimising the reconstruction residual of the IRF-convolved model.
        """
        def obj(a):
            if not (1e-3 < a < 0.999):
                return 1e30
            B = self._discrete_laguerre_basis(T, float(a), self.n_laguerre)
            V = self._convolve_with_irf(B, avg_irf)
            c, *_ = np.linalg.lstsq(V, avg_decay, rcond=None)
            return float(((V @ c - avg_decay) ** 2).sum())

        res = minimize_scalar(obj, bounds=(0.05, 0.98), method="bounded",
                              options={"xatol": 1e-3})
        return float(res.x)

    # ---- exponential stage (amplitudes ARE non-negative -> NNLS) -----------
    @staticmethod
    def _nnls_safe(E: np.ndarray, h: np.ndarray, maxiter: int) -> np.ndarray:
        try:
            a, _ = nnls(E, h, maxiter=maxiter)
            return a
        except RuntimeError:
            a, *_ = np.linalg.lstsq(E, h, rcond=None)
            return np.clip(a, 0.0, None)

    def _tau_bounds(self, T: int):
        tau_lo = self.dt
        tau_hi = self.laser_period_ns if self.laser_period_ns is not None else T * self.dt
        tau_hi = max(tau_hi, tau_lo + 1e-6)
        return tau_lo, tau_hi

    def _safe_tau0(self, tau0: np.ndarray, tau_lo: float, tau_hi: float) -> np.ndarray:
        return np.clip(tau0, tau_lo + 1e-7, tau_hi - 1e-7)

    def _estimate_global_taus(self, h_avg: np.ndarray) -> np.ndarray:
        T = h_avg.shape[0]
        n = np.arange(T)
        N = self.n_components
        tau_lo, tau_hi = self._tau_bounds(T)

        if self.taus_init is not None and self.taus_init.size == N:
            tau0 = self._safe_tau0(self.taus_init.astype(float), tau_lo, tau_hi)
        else:
            span = T * self.dt
            tau0 = self._safe_tau0(
                np.geomspace(max(0.05 * span, tau_lo), min(0.5 * span, tau_hi * 0.9), N),
                tau_lo, tau_hi,
            )

        def residual(params):
            taus = np.clip(params, tau_lo, tau_hi)
            E = np.exp(-n[:, None] * self.dt / taus[None, :])
            a = self._nnls_safe(E, h_avg, 200 * N)
            return E @ a - h_avg

        res = least_squares(residual, tau0, method="trf",
                            bounds=([tau_lo] * N, [tau_hi] * N), max_nfev=2000)
        return np.sort(np.clip(np.abs(res.x), tau_lo, tau_hi))

    def _fit_pixel_exponentials(
        self, h_stack: np.ndarray, tau_init: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X, Y, T = h_stack.shape
        N = self.n_components
        n = np.arange(T)
        tau_lo, tau_hi = self._tau_bounds(T)

        tau_init_safe = self._safe_tau0(tau_init, tau_lo, tau_hi)
        bounds = ([tau_lo] * N, [tau_hi] * N)

        taus_map = np.zeros((X, Y, N), dtype=np.float64)
        amps_map = np.zeros((X, Y, N), dtype=np.float64)

        for x in range(X):
            for y in range(Y):
                h = h_stack[x, y, :]
                if h.sum() < 1e-10:
                    continue

                def residual(params, h=h):
                    taus = np.clip(params, tau_lo, tau_hi)
                    E = np.exp(-n[:, None] * self.dt / taus[None, :])
                    a = self._nnls_safe(E, h, 200 * N)
                    return E @ a - h

                try:
                    res = least_squares(residual, tau_init_safe.copy(), method="trf",
                                        bounds=bounds, max_nfev=1000)
                    taus_px = np.sort(np.clip(np.abs(res.x), tau_lo, tau_hi))
                except Exception:
                    taus_px = tau_init_safe.copy()

                E_px = np.exp(-n[:, None] * self.dt / taus_px[None, :])
                taus_map[x, y, :] = taus_px
                amps_map[x, y, :] = self._nnls_safe(E_px, h, 200 * N)

        return taus_map, amps_map

    # ---- driver ------------------------------------------------------------
    def fit(self, decay: np.ndarray, irf: np.ndarray) -> "LaguerreFLI":
        decay = np.asarray(decay, dtype=np.float64)
        irf   = np.asarray(irf,   dtype=np.float64)

        if decay.ndim == 1:
            decay = decay[None, None, :]
        if decay.ndim != 3:
            raise ValueError("decay must have shape (X, Y, T) or (T,).")
        X, Y, T = decay.shape
        self.decay_ = decay.astype(np.float32)

        ppirf = (irf.ndim == 3)
        if ppirf and irf.shape != decay.shape:
            raise ValueError("per-pixel IRF must match decay shape.")
        if not ppirf and not (irf.ndim == 1 and irf.shape[0] == T):
            raise ValueError("irf must be (T,) or (X, Y, T).")

        avg_decay = decay.reshape(-1, T).mean(0)

        # Determine distinct IRF count up-front so the spatially invariant case
        # (1 distinct IRF) is handled identically to a 1-D IRF.
        if ppirf:
            irf_2d = irf.reshape(-1, T)
            labels, rep_idx = self._unique_irf_groups(irf_2d)
            self.n_unique_irf_ = len(rep_idx)
        else:
            self.n_unique_irf_ = 1

        single_irf = (not ppirf) or self.n_unique_irf_ == 1
        if not ppirf:
            alpha_irf = irf
        elif self.n_unique_irf_ == 1:
            alpha_irf = irf_2d[rep_idx[0]]
        else:
            alpha_irf = irf_2d.mean(0)

        if self.auto_alpha:
            self.alpha = self._optimize_alpha(avg_decay, alpha_irf, T)

        self.basis_ = self._discrete_laguerre_basis(T, self.alpha, self.n_laguerre)
        Y2d = decay.reshape(-1, T).T   # (T, P)

        if single_irf:
            self.V_ = self._convolve_with_irf(self.basis_, alpha_irf)
            C = self._solve_coefficients(self.V_, Y2d)
            model_y = (self.V_ @ C).T.reshape(X, Y, T)
        else:
            # Per-pixel IRF: build one design matrix per unique IRF group.
            P = Y2d.shape[1]
            self.V_ = None
            C = np.zeros((self.n_laguerre, P), dtype=np.float64)
            fit_2d = np.zeros((P, T), dtype=np.float64)
            for g, rep in enumerate(rep_idx):
                cols = np.flatnonzero(labels == g)
                Vg = self._convolve_with_irf(self.basis_, irf_2d[rep])
                Cg = self._solve_coefficients(Vg, Y2d[:, cols])
                C[:, cols] = Cg
                fit_2d[cols] = (Vg @ Cg).T
            model_y = fit_2d.reshape(X, Y, T)

        self.coeffs_         = C.T.reshape(X, Y, self.n_laguerre)
        self.fit_curve_      = model_y
        self.residual_curve_ = decay - model_y
        self.residuals_      = (self.residual_curve_ ** 2).sum(-1)

        h_stack = (self.basis_.T @ C).T.reshape(X, Y, T)   # IRF-free decay
        self.reconstructed_ = h_stack

        h_avg           = h_stack.reshape(-1, T).mean(0)
        taus_init       = self._estimate_global_taus(h_avg)
        self.taus_, A   = self._fit_pixel_exponentials(h_stack, taus_init)
        self.amplitudes_ = A
        total = A.sum(axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.fractions_ = np.where(total > 0, A / total, 0.0)

        n_idx = np.arange(T)
        num   = (h_stack * (n_idx * self.dt)).sum(axis=-1)
        den   = h_stack.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.tau_mean_ = np.where(den > 0, num / den, 0.0)

        return self

    # ---- outputs -----------------------------------------------------------
    def get_parameters(self, data_name: str = "LaguerreFLI_Dataset") -> dict:
        if self.coeffs_ is None:
            raise RuntimeError("Call .fit(decay, irf) first.")

        N    = self.n_components
        X, Y = self.tau_mean_.shape
        T    = self.reconstructed_.shape[-1]
        eps  = 1e-8

        fit_map = (self.fit_curve_ if self.fit_curve_ is not None
                   else self.reconstructed_).astype(np.float32)
        res_map = (self.residual_curve_ if self.residual_curve_ is not None
                   else np.zeros_like(fit_map)).astype(np.float32)
        sdf_map = self.reconstructed_.astype(np.float32)

        if self.decay_ is not None:
            photon_count = self.decay_.sum(axis=-1).astype(np.float32)
        else:
            photon_count = self.amplitudes_.sum(axis=-1).astype(np.float32)

        scaled_fit  = fit_map.astype(np.float64)
        decay_d     = self.decay_.astype(np.float64) if self.decay_ is not None else scaled_fit
        variance    = scaled_fit.copy()
        variance[variance <= 0] = 1.0
        # DOF uses n_laguerre free parameters (the Laguerre model, not the exponential fit)
        dof         = max(T - self.n_laguerre, 1)
        residuals_d = decay_d - scaled_fit
        chi_sq_raw     = np.sum((residuals_d ** 2) / variance, axis=-1).astype(np.float32)
        chi_sq_reduced = (chi_sq_raw / dof).astype(np.float32)

        ss_res = np.sum(residuals_d ** 2, axis=-1)
        ss_tot = np.sum((decay_d - decay_d.mean(axis=-1, keepdims=True)) ** 2, axis=-1)
        r2_map = (1.0 - np.where(ss_tot > eps, ss_res / ss_tot, 1.0)).astype(np.float32)

        pixel_health = (photon_count > 0).astype(np.float32)

        tau_maps   = {f'tau{i+1}_map':   self.taus_[..., i].astype(np.float32)      for i in range(N)}
        alpha_maps = {f'alpha{i+1}_map': self.fractions_[..., i].astype(np.float32) for i in range(N)}

        maps = {
            **tau_maps,
            **alpha_maps,
            'Area_map':             photon_count,
            'tau_mean_map':         self.tau_mean_.astype(np.float32),
            'offset_map':           np.zeros((X, Y), dtype=np.float32),
            'R2_map':               r2_map,
            'chi2_map': chi_sq_raw,
            'reduced_stat_map':     chi_sq_reduced,
            'convergence_map':      pixel_health.copy(),
            'pixel_health_map':     pixel_health,
            'photon_count_map':     photon_count,
        }

        internal_popt_len = 2 * N + 1
        error_maps = np.zeros((X, Y, internal_popt_len), dtype=np.float32)

        tr_maps = {
            'fit_map':      fit_map,
            'residual_map': res_map,
            'sdf_map':      sdf_map,
        }

        mask = photon_count > 0
        mean_chi_sq = float(chi_sq_reduced[mask].mean()) if mask.any() else float('nan')
        print(f"Mean Reduced Chi-Squared (Active Pixels): {mean_chi_sq:.4f}")

        return {
            'name':   data_name,
            'method': f'LaguerreFLI_{N}exp',
            'results': {
                'maps':       maps,
                'error_maps': error_maps,
                'TR_maps':    tr_maps,
            },
        }

    def save_results(self, dataset: dict, folder: str = "results") -> None:
        """Save structured dataset to HDF5 with compression."""
        import h5py, os
        if dataset is None:
            return
        os.makedirs(folder, exist_ok=True)
        h5_path = os.path.join(folder, f"{dataset['name']}_results.h5")
        with h5py.File(h5_path, "w") as f:
            f.attrs['method'] = dataset['method']
            res_grp  = f.create_group("results")
            maps_grp = res_grp.create_group("maps")
            for k, v in dataset['results']['maps'].items():
                maps_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            res_grp.create_group("error_maps").create_dataset(
                "errors", data=dataset['results']['error_maps'],
                compression="gzip", compression_opts=4)
            tr_grp = res_grp.create_group("TR_maps")
            for k, v in dataset['results']['TR_maps'].items():
                tr_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)
        print(f"Analysis complete. Results saved to: {h5_path}")

    def load_map(self, h5_path: str, map_name: str = "tau1_map") -> Optional[np.ndarray]:
        import h5py
        with h5py.File(h5_path, "r") as f:
            key = f"results/maps/{map_name}"
            if key in f:
                return f[key][()]
            print(f"Map '{map_name}' not found in {h5_path}")
            return None

    def predict(self) -> np.ndarray:
        if self.reconstructed_ is None:
            raise RuntimeError("Call .fit(decay, irf) first.")
        return self.reconstructed_

    def __repr__(self) -> str:
        period = f"{self.laser_period_ns} ns" if self.laser_period_ns is not None else "not set"
        return (
            f"LaguerreFLI(n_components={self.n_components}, "
            f"n_laguerre={self.n_laguerre}, alpha={self.alpha:.3f}, "
            f"dt={self.dt} ns, laser_period={period}, "
            f"reg_strength={self.reg_strength})"
        )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X, Y, T = 16, 16, 256
    dt = 0.05
    tau_true = np.array([0.5, 2.5])
    a1 = rng.uniform(0.2, 0.8, size=(X, Y))
    a2 = 1.0 - a1
    n  = np.arange(T)
    h_true = (a1[..., None] * np.exp(-n * dt / tau_true[0])
              + a2[..., None] * np.exp(-n * dt / tau_true[1]))
    t   = n * dt
    irf = np.exp(-0.5 * ((t - 1.0) / 0.08) ** 2)
    irf /= irf.sum()
    y_clean = np.zeros_like(h_true)
    for i in range(X):
        for j in range(Y):
            y_clean[i, j] = np.convolve(irf, h_true[i, j], mode="full")[:T]
    y_meas = rng.poisson(y_clean * 5000).astype(float) / 5000
    model  = LaguerreFLI(n_components=2, n_laguerre=8, alpha=0.85, dt=dt,
                         auto_alpha=True, laser_period_ns=12.5)
    model.fit(y_meas, irf)
    maps = model.get_parameters("SyntheticFLI")['results']['maps']
    print(model)
    print(f"  mean recovered taus (ns) = {model.taus_.mean(axis=(0, 1))}")
    print(f"  true taus (ns)           = {tau_true}")
    print(f"  mean tau_mean            = {maps['tau_mean_map'].mean():.3f} ns")

from __future__ import annotations
from typing import Optional, Union
import numpy as np
from scipy.optimize import least_squares, minimize_scalar, nnls


class LaguerreFLI:
    def __init__(
        self,
        n_components: int = 2,
        n_laguerre: Optional[int] = None,
        alpha: float = 0.85,
        dt: float = 1.0,
        auto_alpha: bool = False,
        nonneg: bool = True,
        taus_init: Optional[np.ndarray] = None,):

        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must lie strictly in (0, 1).")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        self.n_components = int(n_components)
        self.n_laguerre = (
            int(n_laguerre) if n_laguerre is not None else max(4, 2 * n_components)
        )
        if self.n_laguerre < self.n_components:
            raise ValueError("n_laguerre must be >= n_components.")

        self.alpha = float(alpha)
        self.dt = float(dt)
        self.auto_alpha = bool(auto_alpha)
        self.nonneg = bool(nonneg)
        self.taus_init = (
            np.asarray(taus_init, dtype=float) if taus_init is not None else None
        )

        # Results
        self.basis_: Optional[np.ndarray] = None
        self.V_: Optional[np.ndarray] = None
        self.coeffs_: Optional[np.ndarray] = None
        self.taus_: Optional[np.ndarray] = None
        self.amplitudes_: Optional[np.ndarray] = None
        self.fractions_: Optional[np.ndarray] = None
        self.tau_mean_: Optional[np.ndarray] = None
        self.reconstructed_: Optional[np.ndarray] = None
        self.residuals_: Optional[np.ndarray] = None
        self.fit_curve_: Optional[np.ndarray] = None       # (X, Y, T) measurement-space fit
        self.residual_curve_: Optional[np.ndarray] = None  # (X, Y, T) measurement-space residuals


    # Building the discrete Laguerre basis  
    @staticmethod
    def _discrete_laguerre_basis(T: int, alpha: float, L: int) -> np.ndarray:
        """
        Generate the (L, T) discrete Laguerre function matrix via the
        recursive definition from the LET literature.
        """
        b = np.zeros((L, T), dtype=np.float64)
        n = np.arange(T)
        b[0] = np.sqrt(1.0 - alpha) * alpha ** (n / 2.0)

        sa = np.sqrt(alpha)
        s1ma = np.sqrt(1.0 - alpha)
        for j in range(1, L):
            for k in range(T):
                bj_km1   = b[j,     k - 1] if k > 0 else 0.0
                bjm1_k   = b[j - 1, k]
                bjm1_km1 = b[j - 1, k - 1] if k > 0 else 0.0
                b[j, k] = sa * bj_km1 + s1ma * bjm1_k - sa * bjm1_km1
        return b

    # Pre-computing V[:, j] = IRF(n) * b_j(n)

    @staticmethod
    def _convolve_with_irf(basis: np.ndarray, irf: np.ndarray) -> np.ndarray:
        """Compute the (T, L) IRF-convolved-basis matrix, truncated causally."""
        L, T = basis.shape
        irf = np.asarray(irf, dtype=np.float64).ravel()
        s = irf.sum()
        if s > 0:
            irf = irf / s                                 # area-normalize
        V = np.empty((T, L), dtype=np.float64)
        for j in range(L):
            V[:, j] = np.convolve(irf, basis[j], mode="full")[:T]
        return V

    # Solving V c = y for every pixel
    @staticmethod
    def _nnls_safe(V: np.ndarray, y: np.ndarray, maxiter: int) -> np.ndarray:
        """NNLS with a graceful OLS fallback on non-convergence."""
        try:
            c, _ = nnls(V, y, maxiter=maxiter)
            return c
        except RuntimeError:
            # Fallback: clip OLS solution to non-negative orthant.
            c, *_ = np.linalg.lstsq(V, y, rcond=None)
            return np.clip(c, 0.0, None)

    def _solve_coefficients(self, V: np.ndarray, Y2d: np.ndarray) -> np.ndarray:
        """
        Solve for Laguerre coefficients c for a stack of pixel decays.

        Parameters
        ----------
        V   : (T, L) IRF-convolved basis matrix.
        Y2d : (T, P) decay traces for P = X*Y pixels.

        Returns
        -------
        C   : (L, P) Laguerre coefficients.
        """
        if self.nonneg:
            L = V.shape[1]
            P = Y2d.shape[1]
            maxiter = 50 * L                # generous cap for small L
            C = np.zeros((L, P), dtype=np.float64)
            for p in range(P):
                C[:, p] = self._nnls_safe(V, Y2d[:, p], maxiter)
            return C
        # Plain OLS, fully vectorized across pixels.
        VtV = V.T @ V
        VtY = V.T @ Y2d
        return np.linalg.solve(VtV, VtY)

    # minimizing residual for optimal alpha on the average decay
    def _optimize_alpha(
        self, avg_decay: np.ndarray, avg_irf: np.ndarray, T: int
    ) -> float:
        def obj(a):
            if not (1e-3 < a < 0.999):
                return 1e30
            B = self._discrete_laguerre_basis(T, float(a), self.n_laguerre)
            V = self._convolve_with_irf(B, avg_irf)
            if self.nonneg:
                c = self._nnls_safe(V, avg_decay, 50 * self.n_laguerre)
            else:
                c, *_ = np.linalg.lstsq(V, avg_decay, rcond=None)
            return float(((V @ c - avg_decay) ** 2).sum())

        res = minimize_scalar(
            obj, bounds=(0.05, 0.98), method="bounded", options={"xatol": 1e-3}
        )
        return float(res.x)

    # Global N-exponential fit on averaged deconvolved decay

    def _estimate_global_taus(self, h_avg: np.ndarray) -> np.ndarray:
        """
        Non-linear least squares fit of N exponentials to the spatially
        averaged, IRF-free decay. Returns N lifetimes (ascending).
        """
        T = h_avg.shape[0]
        n = np.arange(T)
        N = self.n_components

        if self.taus_init is not None and self.taus_init.size == N:
            tau0 = self.taus_init.astype(float).copy()
        else:
            span = T * self.dt
            # Spread initial guesses geometrically across plausible range.
            tau0 = np.geomspace(max(0.05 * span, self.dt),
                                max(0.5 * span, 2 * self.dt), N)

        def residual(params):
            taus = np.maximum(np.abs(params), 1e-6)
            E = np.exp(-n[:, None] * self.dt / taus[None, :])
            a = self._nnls_safe(E, h_avg, 200 * N)        # non-negative amplitudes
            return E @ a - h_avg

        res = least_squares(residual, tau0, method="lm", max_nfev=2000)
        taus = np.sort(np.abs(res.x))            # ascending: fast -> slow
        return taus

    # Per-pixel exponential fit on the IRF-free decay stack

    def _fit_pixel_exponentials(
        self, h_stack: np.ndarray, tau_init: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit N exponentials to every pixel's IRF-free decay independently.

        Uses global tau_init as a warm start so per-pixel fits converge
        quickly without starting from scratch.

        Returns
        -------
        taus_map : (X, Y, N)  per-pixel lifetimes, ascending within each pixel
        amps_map : (X, Y, N)  per-pixel amplitudes
        """
        X, Y, T = h_stack.shape
        N = self.n_components
        n = np.arange(T)

        taus_map = np.zeros((X, Y, N), dtype=np.float64)
        amps_map = np.zeros((X, Y, N), dtype=np.float64)

        for x in range(X):
            for y in range(Y):
                h = h_stack[x, y, :]
                if h.sum() < 1e-10:
                    continue

                def residual(params, h=h):
                    taus = np.maximum(np.abs(params), 1e-6)
                    E = np.exp(-n[:, None] * self.dt / taus[None, :])
                    a = self._nnls_safe(E, h, 200 * N)
                    return E @ a - h

                try:
                    res = least_squares(
                        residual, tau_init.copy(), method="lm", max_nfev=500
                    )
                    taus_px = np.sort(np.abs(res.x))
                except Exception:
                    taus_px = np.sort(np.abs(tau_init))

                E_px = np.exp(-n[:, None] * self.dt / taus_px[None, :])
                a_px = self._nnls_safe(E_px, h, 200 * N)

                taus_map[x, y, :] = taus_px
                amps_map[x, y, :] = a_px

        return taus_map, amps_map

    # Public API

    def fit(self,
        decay: np.ndarray,
        irf: np.ndarray,) -> "LaguerreFLI":

        """
        Fit the LET model to a FLI image cube.

        Parameters
        ----------
        decay : ndarray, shape (X, Y, T) or (T,)
            Measured fluorescence decay per pixel. A 1-D trace is treated
            as a single pixel.
        irf : ndarray, shape (T,) or (X, Y, T)
            Instrument response function. Pass a 1-D vector to use one
            global IRF for the whole image (much faster). Pass a 3-D array
            to use per-pixel IRFs (e.g. for spatially varying systems).

        Returns
        -------
        self
        """
        decay = np.asarray(decay, dtype=np.float64)
        irf = np.asarray(irf, dtype=np.float64)

        # ---- shape bookkeeping --------------------------------------- #
        if decay.ndim == 1:
            decay = decay[None, None, :]
        if decay.ndim != 3:
            raise ValueError("decay must have shape (X, Y, T) or (T,).")
        X, Y, T = decay.shape

        per_pixel_irf = (irf.ndim == 3)
        if per_pixel_irf:
            if irf.shape != decay.shape:
                raise ValueError("per-pixel IRF must match decay shape (X, Y, T).")
        elif irf.ndim == 1:
            if irf.shape[0] != T:
                raise ValueError("IRF length must match decay's time axis.")
        else:
            raise ValueError("irf must have shape (T,) or (X, Y, T).")

        avg_decay = decay.reshape(-1, T).mean(axis=0)
        avg_irf = irf.reshape(-1, T).mean(axis=0) if per_pixel_irf else irf

        # ---- (optional) optimize alpha ------------------------------- #
        if self.auto_alpha:
            self.alpha = self._optimize_alpha(avg_decay, avg_irf, T)

        # ---- Step 2: build basis ------------------------------------- #
        self.basis_ = self._discrete_laguerre_basis(T, self.alpha, self.n_laguerre)

        # ---- Step 3: solve for coefficients per pixel ---------------- #
        Y2d = decay.reshape(-1, T).T                        # (T, P)
        if not per_pixel_irf:
            self.V_ = self._convolve_with_irf(self.basis_, avg_irf)
            C = self._solve_coefficients(self.V_, Y2d)      # (L, P)
        else:
            P = Y2d.shape[1]
            C = np.zeros((self.n_laguerre, P), dtype=np.float64)
            irf_2d = irf.reshape(-1, T)
            maxiter = 50 * self.n_laguerre
            for p in range(P):
                Vp = self._convolve_with_irf(self.basis_, irf_2d[p])
                if self.nonneg:
                    C[:, p] = self._nnls_safe(Vp, Y2d[:, p], maxiter)
                else:
                    C[:, p], *_ = np.linalg.lstsq(Vp, Y2d[:, p], rcond=None)

        self.coeffs_ = C.T.reshape(X, Y, self.n_laguerre)

        # ---- Reconstruct IRF-free decay h(n) = B c ------------------- #
        h_stack = (self.basis_.T @ C).T.reshape(X, Y, T)
        self.reconstructed_ = h_stack

        # Residuals in measurement space (only when V_ is a single matrix)
        if self.V_ is not None:
            model_y = (self.V_ @ C).T.reshape(X, Y, T)
            self.fit_curve_ = model_y
            self.residual_curve_ = decay - model_y
            self.residuals_ = (self.residual_curve_ ** 2).sum(axis=-1)

        # ---- Step 4: per-pixel exponential fit on IRF-free decays ------- #
        h_avg = h_stack.reshape(-1, T).mean(axis=0)
        taus_init = self._estimate_global_taus(h_avg)        # warm-start

        # taus_ and amplitudes_ are now both pixel-wise maps
        self.taus_, A = self._fit_pixel_exponentials(h_stack, taus_init)
        self.amplitudes_ = A                                 # (X, Y, N)
        total = A.sum(axis=-1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.fractions_ = np.where(total > 0, A / total, 0.0)

        # ---- Method B: intensity-weighted mean lifetime -------------- #
        n_idx = np.arange(T)
        num = (h_stack * (n_idx * self.dt)).sum(axis=-1)
        den = h_stack.sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.tau_mean_ = np.where(den > 0, num / den, 0.0)

        return self

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #
    def get_parameters(self, data_name: str = "LaguerreFLI_Dataset") -> dict:
        """
        Return all fitted quantities in the standardized package dictionary format,
        compatible with Fli_CPUProcessor and Fli_GPUProcessor outputs.

        Structure
        ---------
        {
          'name': str,
          'method': str,
          'results': {
            'maps': {
              'tau1_map' .. 'tauN_map'   : (X, Y) per-pixel lifetimes,
              'alpha1_map' .. 'alphaN_map': (X, Y) per-pixel fractional intensities,
              'Area_map'                 : (X, Y) total signal amplitude,
              'tau_mean_map'             : (X, Y) intensity-weighted mean lifetime,
              'chi2_or_deviance_map'     : (X, Y) sum-of-squared residuals,
              'pixel_health_map'         : (X, Y) 1 = valid fit, 0 = empty/failed,
            },
            'error_maps': (X, Y, 2*N) zeros (Laguerre gives no formal uncertainties),
            'TR_maps': {
              'fit_map'     : (X, Y, T) measurement-space fitted curve,
              'residual_map': (X, Y, T) measurement-space residuals,
            }
          }
        }
        """
        if self.coeffs_ is None:
            raise RuntimeError("Call .fit(decay, irf) first.")

        N = self.n_components
        X, Y = self.tau_mean_.shape

        tau_maps   = {f'tau{i+1}_map':   self.taus_[..., i].astype(np.float32)     for i in range(N)}
        alpha_maps = {f'alpha{i+1}_map': self.fractions_[..., i].astype(np.float32) for i in range(N)}

        pixel_health = (self.amplitudes_.sum(axis=-1) > 0).astype(np.float32)
        ssr = (self.residuals_ if self.residuals_ is not None
               else np.zeros((X, Y), dtype=np.float32))

        maps = {
            **tau_maps,
            **alpha_maps,
            'Area_map':             self.amplitudes_.sum(axis=-1).astype(np.float32),
            'tau_mean_map':         self.tau_mean_.astype(np.float32),
            'chi2_or_deviance_map': ssr.astype(np.float32),
            'pixel_health_map':     pixel_health,
        }

        error_maps = np.zeros((X, Y, 2 * N), dtype=np.float32)

        fit_map = (self.fit_curve_ if self.fit_curve_ is not None
                   else self.reconstructed_).astype(np.float32)
        res_map = (self.residual_curve_ if self.residual_curve_ is not None
                   else np.zeros_like(fit_map)).astype(np.float32)

        return {
            'name':   data_name,
            'method': f'LaguerreFLI_{N}exp',
            'results': {
                'maps':       maps,
                'error_maps': error_maps,
                'TR_maps': {
                    'fit_map':      fit_map,
                    'residual_map': res_map,
                },
            },
        }

    def save_results(self, dataset: dict, folder: str = "results") -> None:
        """Save the structured dataset to HDF5 with compression (mirrors solver interface)."""
        import h5py, os
        if dataset is None:
            return
        if not os.path.exists(folder):
            os.makedirs(folder)
        h5_path = os.path.join(folder, f"{dataset['name']}_results.h5")
        with h5py.File(h5_path, "w") as f:
            f.attrs['method'] = dataset['method']
            res_grp = f.create_group("results")

            maps_grp = res_grp.create_group("maps")
            for k, v in dataset['results']['maps'].items():
                maps_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)

            err_grp = res_grp.create_group("error_maps")
            err_grp.create_dataset("errors", data=dataset['results']['error_maps'],
                                   compression="gzip", compression_opts=4)

            tr_grp = res_grp.create_group("TR_maps")
            for k, v in dataset['results']['TR_maps'].items():
                tr_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)

        print(f"Analysis complete. Results saved to: {h5_path}")

    def load_map(self, h5_path: str, map_name: str = "tau1_map") -> Optional[np.ndarray]:
        """Reload a specific parameter map from a saved HDF5 file."""
        import h5py
        with h5py.File(h5_path, "r") as f:
            key = f"results/maps/{map_name}"
            if key in f:
                return f[key][()]
            print(f"Map '{map_name}' not found in {h5_path}")
            return None

    def predict(self) -> np.ndarray:
        """Return the IRF-free reconstructed decay h(n) per pixel."""
        if self.reconstructed_ is None:
            raise RuntimeError("Call .fit(decay, irf) first.")
        return self.reconstructed_

    def __repr__(self) -> str:
        return (
            f"LaguerreFLI(n_components={self.n_components}, "
            f"n_laguerre={self.n_laguerre}, alpha={self.alpha:.3f}, "
            f"dt={self.dt}, nonneg={self.nonneg})"
        )


# ---------------------------------------------------------------------- #
# Minimal self-test with synthetic data
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Synthetic FLI cube: 16 x 16 pixels, 256 time bins, dt = 0.05 ns
    X, Y, T = 16, 16, 256
    dt = 0.05

    # Ground-truth bi-exponential lifetimes (ns)
    tau_true = np.array([0.5, 2.5])

    # Random per-pixel amplitudes
    a1 = rng.uniform(0.2, 0.8, size=(X, Y))
    a2 = 1.0 - a1
    n = np.arange(T)
    h_true = (
        a1[..., None] * np.exp(-n * dt / tau_true[0])
        + a2[..., None] * np.exp(-n * dt / tau_true[1])
    )

    # Gaussian IRF, FWHM ~ 0.2 ns, centered at bin 20
    t = np.arange(T) * dt
    irf = np.exp(-0.5 * ((t - 1.0) / 0.08) ** 2)
    irf /= irf.sum()

    # Convolve and add Poisson-like noise
    y_clean = np.zeros_like(h_true)
    for i in range(X):
        for j in range(Y):
            y_clean[i, j] = np.convolve(irf, h_true[i, j], mode="full")[:T]
    photons = 5000
    y_meas = rng.poisson(y_clean * photons).astype(float) / photons

    # Fit a bi-exponential model
    model = LaguerreFLI(
        n_components=2, n_laguerre=5, alpha=0.85, dt=dt,
        auto_alpha=True, nonneg=True,
    )
    model.fit(y_meas, irf)

    params = model.get_parameters(data_name="SyntheticFLI")
    maps = params['results']['maps']

    print(model)
    print(f"  method                  = {params['method']}")
    print(f"  optimal alpha           = {model.alpha:.3f}")
    print(f"  mean recovered taus (ns)= {model.taus_.mean(axis=(0, 1))}")
    print(f"  true taus (ns)          = {tau_true}")
    print(f"  mean recovered a1       = {maps['alpha1_map'].mean():.3f}")
    print(f"  mean true a1            = {a1.mean():.3f}")
    print(f"  mean tau_mean (Method B) = {maps['tau_mean_map'].mean():.3f} ns")
    print(f"  output maps             : {list(maps.keys())}")
    print(f"  TR_maps shapes          : fit={params['results']['TR_maps']['fit_map'].shape}, "
          f"res={params['results']['TR_maps']['residual_map'].shape}")
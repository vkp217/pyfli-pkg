from __future__ import annotations
from typing import Optional
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
        taus_init: Optional[np.ndarray] = None,
        laser_period_ns: Optional[float] = None,
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
        self.n_laguerre = (
            int(n_laguerre) if n_laguerre is not None else max(4, 2 * n_components)
        )
        if self.n_laguerre < self.n_components:
            raise ValueError("n_laguerre must be >= n_components.")

        self.alpha            = float(alpha)
        self.dt               = float(dt)
        self.auto_alpha       = bool(auto_alpha)
        self.nonneg           = bool(nonneg)
        self.laser_period_ns  = float(laser_period_ns) if laser_period_ns is not None else None
        self.taus_init        = (
            np.asarray(taus_init, dtype=float) if taus_init is not None else None
        )

        self.basis_:          Optional[np.ndarray] = None
        self.V_:              Optional[np.ndarray] = None
        self.coeffs_:         Optional[np.ndarray] = None
        self.taus_:           Optional[np.ndarray] = None
        self.amplitudes_:     Optional[np.ndarray] = None
        self.fractions_:      Optional[np.ndarray] = None
        self.tau_mean_:       Optional[np.ndarray] = None
        self.reconstructed_:  Optional[np.ndarray] = None
        self.residuals_:      Optional[np.ndarray] = None
        self.fit_curve_:      Optional[np.ndarray] = None
        self.residual_curve_: Optional[np.ndarray] = None
        self.decay_:          Optional[np.ndarray] = None

    @staticmethod
    def _discrete_laguerre_basis(T: int, alpha: float, L: int) -> np.ndarray:
        """Return (L, T) discrete Laguerre basis matrix."""
        b = np.zeros((L, T), dtype=np.float64)
        n = np.arange(T)
        b[0] = np.sqrt(1.0 - alpha) * alpha ** (n / 2.0)
        sa, s1ma = np.sqrt(alpha), np.sqrt(1.0 - alpha)
        for j in range(1, L):
            for k in range(T):
                bj_km1   = b[j,     k - 1] if k > 0 else 0.0
                bjm1_k   = b[j - 1, k]
                bjm1_km1 = b[j - 1, k - 1] if k > 0 else 0.0
                b[j, k]  = sa * bj_km1 + s1ma * bjm1_k - sa * bjm1_km1
        return b

    @staticmethod
    def _convolve_with_irf(basis: np.ndarray, irf: np.ndarray) -> np.ndarray:
        """Return (T, L) IRF-convolved basis matrix, causally truncated."""
        L, T = basis.shape
        irf = np.asarray(irf, dtype=np.float64).ravel()
        s = irf.sum()
        if s > 0:
            irf = irf / s
        V = np.empty((T, L), dtype=np.float64)
        for j in range(L):
            V[:, j] = np.convolve(irf, basis[j], mode="full")[:T]
        return V

    @staticmethod
    def _nnls_safe(V: np.ndarray, y: np.ndarray, maxiter: int) -> np.ndarray:
        try:
            c, _ = nnls(V, y, maxiter=maxiter)
            return c
        except RuntimeError:
            c, *_ = np.linalg.lstsq(V, y, rcond=None)
            return np.clip(c, 0.0, None)

    def _solve_coefficients(self, V: np.ndarray, Y2d: np.ndarray) -> np.ndarray:
        """Solve V c = y for all pixels; returns (L, P) coefficient matrix."""
        if self.nonneg:
            L, P = V.shape[1], Y2d.shape[1]
            C = np.zeros((L, P), dtype=np.float64)
            for p in range(P):
                C[:, p] = self._nnls_safe(V, Y2d[:, p], 50 * L)
            return C
        VtV = V.T @ V
        VtY = V.T @ Y2d
        return np.linalg.solve(VtV, VtY)

    def _optimize_alpha(self, avg_decay: np.ndarray, avg_irf: np.ndarray, T: int) -> float:
        def obj(a):
            if not (1e-3 < a < 0.999):
                return 1e30
            B = self._discrete_laguerre_basis(T, float(a), self.n_laguerre)
            V = self._convolve_with_irf(B, avg_irf)
            c = (self._nnls_safe(V, avg_decay, 50 * self.n_laguerre)
                 if self.nonneg else
                 np.linalg.lstsq(V, avg_decay, rcond=None)[0])
            return float(((V @ c - avg_decay) ** 2).sum())
        res = minimize_scalar(obj, bounds=(0.05, 0.98), method="bounded",
                              options={"xatol": 1e-3})
        return float(res.x)

    def _tau_bounds(self, T: int):
        """
        tau_lo = dt (time resolution), tau_hi = laser_period_ns or T*dt.
        Failsafe mirrors resolve_params_and_bounds in base_static.py:
          high = max(high, low + 1e-6), p0 clipped to [low+1e-7, high-1e-7].
        """
        tau_lo = self.dt
        tau_hi = self.laser_period_ns if self.laser_period_ns is not None else T * self.dt
        tau_hi = max(tau_hi, tau_lo + 1e-6)           # base_static failsafe
        return tau_lo, tau_hi

    def _safe_tau0(self, tau0: np.ndarray, tau_lo: float, tau_hi: float) -> np.ndarray:
        """Clip initial guess inside bounds with 1e-7 margin (base_static pattern)."""
        return np.clip(tau0, tau_lo + 1e-7, tau_hi - 1e-7)

    def _estimate_global_taus(self, h_avg: np.ndarray) -> np.ndarray:
        """Fit N exponentials to the spatially averaged IRF-free decay (warm-start)."""
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
                tau_lo, tau_hi
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
        """Per-pixel TRF fit of N exponentials on the IRF-free decay stack."""
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

                E_px  = np.exp(-n[:, None] * self.dt / taus_px[None, :])
                a_px  = self._nnls_safe(E_px, h, 200 * N)
                taus_map[x, y, :] = taus_px
                amps_map[x, y, :] = a_px

        return taus_map, amps_map

    def fit(self, decay: np.ndarray, irf: np.ndarray) -> "LaguerreFLI":
        decay = np.asarray(decay, dtype=np.float64)
        irf   = np.asarray(irf,   dtype=np.float64)

        if decay.ndim == 1:
            decay = decay[None, None, :]
        if decay.ndim != 3:
            raise ValueError("decay must have shape (X, Y, T) or (T,).")
        X, Y, T = decay.shape
        self.decay_ = decay.astype(np.float32)

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
        avg_irf   = irf.reshape(-1, T).mean(axis=0) if per_pixel_irf else irf

        if self.auto_alpha:
            self.alpha = self._optimize_alpha(avg_decay, avg_irf, T)

        self.basis_ = self._discrete_laguerre_basis(T, self.alpha, self.n_laguerre)

        Y2d = decay.reshape(-1, T).T                    # (T, P)
        if not per_pixel_irf:
            self.V_ = self._convolve_with_irf(self.basis_, avg_irf)
            C = self._solve_coefficients(self.V_, Y2d)  # (L, P)
        else:
            P      = Y2d.shape[1]
            C      = np.zeros((self.n_laguerre, P), dtype=np.float64)
            fit_2d = np.zeros((P, T), dtype=np.float64)
            irf_2d = irf.reshape(-1, T)
            for p in range(P):
                Vp = self._convolve_with_irf(self.basis_, irf_2d[p])
                if self.nonneg:
                    C[:, p] = self._nnls_safe(Vp, Y2d[:, p], 50 * self.n_laguerre)
                else:
                    C[:, p], *_ = np.linalg.lstsq(Vp, Y2d[:, p], rcond=None)
                fit_2d[p] = Vp @ C[:, p]
            model_y              = fit_2d.reshape(X, Y, T)
            self.fit_curve_      = model_y
            self.residual_curve_ = decay - model_y
            self.residuals_      = (self.residual_curve_ ** 2).sum(axis=-1)

        self.coeffs_ = C.T.reshape(X, Y, self.n_laguerre)

        h_stack = (self.basis_.T @ C).T.reshape(X, Y, T)
        self.reconstructed_ = h_stack

        if self.V_ is not None:
            model_y = (self.V_ @ C).T.reshape(X, Y, T)
            self.fit_curve_      = model_y
            self.residual_curve_ = decay - model_y
            self.residuals_      = (self.residual_curve_ ** 2).sum(axis=-1)

        h_avg       = h_stack.reshape(-1, T).mean(axis=0)
        taus_init   = self._estimate_global_taus(h_avg)
        self.taus_, A = self._fit_pixel_exponentials(h_stack, taus_init)
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

    def get_parameters(self, data_name: str = "LaguerreFLI_Dataset") -> dict:
        if self.coeffs_ is None:
            raise RuntimeError("Call .fit(decay, irf) first.")

        N   = self.n_components
        X, Y = self.tau_mean_.shape
        T   = self.reconstructed_.shape[-1]
        eps = 1e-8

        # fit_curve_ is always IRF-convolved (fixed for per-pixel IRF case too);
        # reconstructed_ is the IRF-free Laguerre signal -> sdf_map
        fit_map = (self.fit_curve_ if self.fit_curve_ is not None
                   else self.reconstructed_).astype(np.float32)
        res_map = (self.residual_curve_ if self.residual_curve_ is not None
                   else np.zeros_like(fit_map)).astype(np.float32)
        sdf_map = self.reconstructed_.astype(np.float32)

        # Photon counts from stored measured decay
        if self.decay_ is not None:
            photon_count = self.decay_.sum(axis=-1).astype(np.float32)
        else:
            photon_count = self.amplitudes_.sum(axis=-1).astype(np.float32)

        # Poisson chi-square against the IRF-convolved fit
        scaled_fit = fit_map.astype(np.float64)
        decay_d    = self.decay_.astype(np.float64) if self.decay_ is not None else scaled_fit
        variance   = scaled_fit.copy()
        variance[variance <= 0] = 1.0
        dof        = max(T - (2 * N + 1) - 1, 1)   # T bins – free params – 1
        residuals_d = decay_d - scaled_fit
        chi_sq_raw     = np.sum((residuals_d ** 2) / variance, axis=-1).astype(np.float32)
        chi_sq_reduced = (chi_sq_raw / dof).astype(np.float32)

        # R² per pixel
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
            'chi2_or_deviance_map': chi_sq_raw,
            'reduced_stat_map':     chi_sq_reduced,
            'convergence_map':      pixel_health.copy(),
            'pixel_health_map':     pixel_health,
            'photon_count_map':     photon_count,
        }

        # popt length mirrors base_static convention: [A, tau1..tauN, f1..f(N-1), offset]
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
        if not os.path.exists(folder):
            os.makedirs(folder)
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
            f"dt={self.dt} ns, laser_period={period}, nonneg={self.nonneg})"
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
    t   = np.arange(T) * dt
    irf = np.exp(-0.5 * ((t - 1.0) / 0.08) ** 2)
    irf /= irf.sum()
    y_clean = np.zeros_like(h_true)
    for i in range(X):
        for j in range(Y):
            y_clean[i, j] = np.convolve(irf, h_true[i, j], mode="full")[:T]
    y_meas = rng.poisson(y_clean * 5000).astype(float) / 5000
    model  = LaguerreFLI(n_components=2, n_laguerre=5, alpha=0.85, dt=dt,
                         auto_alpha=True, nonneg=True, laser_period_ns=12.5)
    model.fit(y_meas, irf)
    params = model.get_parameters(data_name="SyntheticFLI")
    maps   = params['results']['maps']
    print(model)
    print(f"  mean recovered taus (ns) = {model.taus_.mean(axis=(0, 1))}")
    print(f"  true taus (ns)           = {tau_true}")
    print(f"  mean tau_mean            = {maps['tau_mean_map'].mean():.3f} ns")

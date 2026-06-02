from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares

try:
    from .detector_weights import make_observation
except ImportError:
    from detector_weights import make_observation

EPS = 1e-9


def cyclic_conv(h, f):
    N = h.shape[-1]
    return np.fft.irfft(np.fft.rfft(h, axis=-1) * np.fft.rfft(f, axis=-1), n=N, axis=-1)


def cyclic_corr(u, f):
    N = u.shape[-1]
    return np.fft.irfft(np.conj(np.fft.rfft(f, axis=-1)) * np.fft.rfft(u, axis=-1),
                        n=N, axis=-1)

def decay_basis(taus, t, T):
    taus = np.atleast_1d(np.asarray(taus, float))
    return np.stack([np.exp(-t / tau) / (1.0 - np.exp(-T / tau)) for tau in taus], 0)


def build_gate_matrix(t, T, n_gates, width, edge=0.0, eta=None):
    N = t.size
    dt = T / N
    centers = np.linspace(0, T, n_gates, endpoint=False)
    G = np.zeros((n_gates, N))
    for g, t0 in enumerate(centers):
        if edge <= 0:
            rel = (t - t0) % T
            G[g] = ((rel >= 0) & (rel < width)).astype(float)
        else:
            from scipy.special import erf
            rel = (t - t0)
            prof = 0.5 * (erf(rel / (np.sqrt(2) * edge))
                          - erf((rel - width) / (np.sqrt(2) * edge)))
            G[g] = np.clip(prof, 0, 1)
    G *= dt
    if eta is not None:
        G *= np.asarray(eta)[:, None]
    return G


def project_simplex(V):
    V = np.atleast_2d(V)
    n = V.shape[1]
    U = np.sort(V, axis=1)[:, ::-1]
    cssv = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, n + 1)
    cond = U - cssv / ind > 0
    rho = cond.sum(axis=1)
    theta = cssv[np.arange(V.shape[0]), rho - 1] / rho
    return np.maximum(V - theta[:, None], 0.0)


def huber_tv_grad(h, delta):
    d = h - np.roll(h, 1, axis=-1)
    psi = np.where(np.abs(d) <= delta, d / delta, np.sign(d))
    return psi - np.roll(psi, -1, axis=-1)


def spatial_laplacian(H, ny, nx):
    N = H.shape[-1]
    Himg = H.reshape(ny, nx, N)
    lap = 4 * Himg \
        - np.roll(Himg, 1, 0) - np.roll(Himg, -1, 0) \
        - np.roll(Himg, 1, 1) - np.roll(Himg, -1, 1)
    return lap.reshape(ny * nx, N)


def fourier_shift(H, s):
    N = H.shape[-1]
    k = np.fft.rfftfreq(N, d=1.0 / N)
    phase = np.exp(-2j * np.pi * k * s / N)
    return np.fft.irfft(np.fft.rfft(H, axis=-1) * phase, n=N, axis=-1)


def pin_barycenter(H, c_target):
    idx = np.arange(H.shape[-1])
    cbar = float(np.mean((H * idx).sum(-1) / np.maximum(H.sum(-1), EPS)))
    return np.maximum(fourier_shift(H, c_target - cbar), 0.0)


@dataclass
class SolverConfig:
    T: float = 12.5
    n_models: int = 2
    tau_init: tuple = (0.5, 2.0)
    tau_bounds: tuple = (0.05, 8.0)
    tau_sep: float = 1.4
    rho1: float = 0.02
    rho2: float = 0.10
    outer_iters: int = 8
    irf_inner_iters: int = 250
    irf_step: float = 0.5
    estimate_irf: bool = True
    pin_global_shift: bool = False
    verbose: bool = True


def _phi(taus, h, t, T, G):
    B = decay_basis(taus, t, T)
    M = cyclic_conv(h[None, :], B)
    return (G @ M.T)


def fit_decay_pixel(lam_obs, w, h, t, T, G, cfg):
    sw = np.sqrt(w)
    lo, hi = cfg.tau_bounds

    def resid(log_taus):
        taus = np.exp(log_taus)
        Phi = _phi(taus, h, t, T, G)
        A, *_ = np.linalg.lstsq(sw[:, None] * Phi, sw * lam_obs, rcond=None)
        A = np.maximum(A, 0.0)
        return sw * (Phi @ A - lam_obs)

    x0 = np.log(np.clip(cfg.tau_init[:cfg.n_models], lo, hi))
    sol = least_squares(resid, x0, method="trf",
                        bounds=(np.log(lo), np.log(hi)), max_nfev=200)
    taus = np.sort(np.exp(sol.x))
    Phi = _phi(taus, h, t, T, G)
    A, *_ = np.linalg.lstsq(sw[:, None] * Phi, sw * lam_obs, rcond=None)
    A = np.maximum(A, 0.0)
    return taus, A, Phi @ A


def update_irf(H, F, lam_obs, W, G, mu1, mu2, ny, nx, cfg):
    P, N = H.shape

    def data_grad(Hx):
        lam = cyclic_conv(Hx, F) @ G.T
        resid = W * (lam - lam_obs)
        back = resid @ G
        return 2.0 * cyclic_corr(back, F)

    def data_hess(V):
        lam = cyclic_conv(V, F) @ G.T
        return 2.0 * cyclic_corr((W * lam) @ G, F)

    v = np.random.default_rng(0).standard_normal((P, N))
    v /= np.linalg.norm(v) + EPS
    L_data = 1.0
    for _ in range(12):
        Av = data_hess(v)
        L_data = np.linalg.norm(Av)
        v = Av / (L_data + EPS)

    delta = 0.05 * float(H.max())
    L = L_data + mu1 / max(delta, EPS) + 8.0 * mu2 + EPS
    step = 1.0 / L

    for _ in range(cfg.irf_inner_iters):
        g = data_grad(H)
        g += mu1 * huber_tv_grad(H, delta)
        g += mu2 * 2.0 * spatial_laplacian(H, ny, nx)
        H = project_simplex(H - step * g)
    return H


def solve_flim(y, detector, det_params, ny, nx, gate_spec, cfg: SolverConfig,
               h_init=None):
    P = y.shape[0]
    N = gate_spec.get("N", 256)
    t = np.linspace(0, cfg.T, N, endpoint=False)
    G = build_gate_matrix(t, cfg.T, gate_spec["n_gates"], gate_spec["width"],
                          gate_spec.get("edge", 0.0), gate_spec.get("eta", None))

    lam_obs, W = make_observation(y, detector, det_params)

    if h_init is None:
        sig = max(2.0, N / 64)
        c = N // 8
        h0 = np.exp(-0.5 * ((np.arange(N) - c) / sig) ** 2)
        H = np.tile(h0 / h0.sum(), (P, 1))
    else:
        H = h_init.copy()

    taus = np.tile(cfg.tau_init[:cfg.n_models], (P, 1)).astype(float)
    amps = np.zeros((P, cfg.n_models))
    F = np.zeros((P, N))

    def assemble_F(taus, amps):
        out = np.zeros((P, N))
        for k in range(P):
            Bk = decay_basis(taus[k], t, cfg.T)
            out[k] = amps[k] @ Bk
        return out

    for k in range(P):
        tk, ak, _ = fit_decay_pixel(lam_obs[k], W[k], H[k], t, cfg.T, G, cfg)
        taus[k], amps[k] = tk, ak
    F = assemble_F(taus, amps)
    lam_model = cyclic_conv(H, F) @ G.T
    D0 = float(np.sum(W * (lam_model - lam_obs) ** 2))
    TV0 = float(np.sum(np.abs(H - np.roll(H, 1, -1))))
    E_h = float(np.sum(H ** 2))
    mu1 = cfg.rho1 * D0 / (TV0 + EPS)
    mu2 = cfg.rho2 * D0 / (E_h + EPS)
    if cfg.verbose and cfg.estimate_irf:
        print(f"[init] data={D0:.3g}  TV0={TV0:.3g}  E_h={E_h:.3g}"
              f"  ->  mu1={mu1:.4g}  mu2={mu2:.4g}")

    idx = np.arange(N)
    c_target = float(np.mean((H * idx).sum(-1) / np.maximum(H.sum(-1), EPS)))
    for it in range(cfg.outer_iters):
        for k in range(P):
            taus[k], amps[k], _ = fit_decay_pixel(lam_obs[k], W[k], H[k],
                                                  t, cfg.T, G, cfg)
        F = assemble_F(taus, amps)
        lam_model = cyclic_conv(H, F) @ G.T
        _, W = make_observation(y, detector, det_params)
        if cfg.estimate_irf:
            H = update_irf(H, F, lam_obs, W, G, mu1, mu2, ny, nx, cfg)
            if cfg.pin_global_shift:
                H = pin_barycenter(H, c_target)
        if cfg.verbose:
            misfit = float(np.sum(W * (cyclic_conv(H, F) @ G.T - lam_obs) ** 2))
            print(f"[iter {it+1:2d}] weighted misfit = {misfit:.5g}"
                  f"   mean tau = {taus.mean(0)}")
        if not cfg.estimate_irf:
            break

    return dict(taus=taus, amps=amps, irf=H, mu1=mu1, mu2=mu2, gate=G, t=t)

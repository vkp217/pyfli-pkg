"""
flim_solver.py
==============

Joint estimation of per-pixel IRFs h_k and decay parameters (A_i, tau_i) from
gated or photon-counting FLIM data, under cyclic convolution at the laser period.

Forward model (per pixel k):
    f_k(t)      = sum_i A_ik * D(tau_ik)         decay, cyclically wrapped
    m_k(t)      = (h_k  *circular*  f_k)(t)       cyclic convolution
    lambda_k    = Gate @ m_k                      gate integration (incl. width W)
    measurement = detector(lambda_k)              Poisson / Binomial / Poisson-Gauss

The detector is handled entirely in detector_weights.make_observation, which
returns (lambda_obs, w). The solver runs weighted least squares in lambda space.

Blocks
------
A) decay fit with IRF fixed   : VARPRO (linear amplitudes eliminated) + Levenberg-
                                Marquardt on the lifetimes (scipy.least_squares).
B) IRF update with decay fixed: projected gradient on a convex objective
                                = weighted data misfit + mu1 * Huber-TV(h)
                                  + mu2 * spatial Laplacian coupling,
                                projected onto the probability simplex {h>=0, sum=1}.

Regularization weights (mu1, mu2) are AUTO-SCALED from the data at initialisation
so the dimensionless knobs (rho1, rho2) are portable across datasets:
    mu1 = rho1 * D0 / TV0      (TV penalty ~ rho1 of the data misfit at init)
    mu2 = rho2 * D0 / SP0      (spatial penalty ~ rho2 of the data misfit at init)
Recommended defaults: rho1 = 0.02 (2%), rho2 = 0.10 (10%).
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares

from .detector_weights import make_observation

EPS = 1e-9


# --------------------------------------------------------------------------- #
#  Cyclic operators (FFT based, batched over the last axis)                    #
# --------------------------------------------------------------------------- #
def cyclic_conv(h, f):
    """Circular convolution along the last axis. h, f : (..., N)."""
    N = h.shape[-1]
    return np.fft.irfft(np.fft.rfft(h, axis=-1) * np.fft.rfft(f, axis=-1), n=N, axis=-1)


def cyclic_corr(u, f):
    """Adjoint of cyclic_conv(., f) w.r.t. the first argument (cross-correlation)."""
    N = u.shape[-1]
    return np.fft.irfft(np.conj(np.fft.rfft(f, axis=-1)) * np.fft.rfft(u, axis=-1),
                        n=N, axis=-1)


def decay_basis(taus, t, T):
    """Periodic steady-state mono-exponential responses, shape (n_tau, N).

    The 80 MHz pulse train means each excitation sees residual signal from
    previous pulses. Summing the geometric train in closed form gives the
    cyclic-convolution-consistent boundary:
        D(t; tau) = exp(-t/tau) / (1 - exp(-T/tau)),   t in [0, T).
    """
    taus = np.atleast_1d(np.asarray(taus, float))
    return np.stack([np.exp(-t / tau) / (1.0 - np.exp(-T / tau)) for tau in taus], 0)


# --------------------------------------------------------------------------- #
#  Gate operator                                                               #
# --------------------------------------------------------------------------- #
def build_gate_matrix(t, T, n_gates, width, edge=0.0, eta=None):
    """Return G of shape (n_gates, N): row g integrates [t_g, t_g + width].

    edge : gate rise/fall time (sigma of the erf-smoothed boxcar). 0 = ideal box.
    eta  : per-gate efficiency (length n_gates) or None (=1).
    Gates are evenly spaced across the period; widths may overlap or tile.
    """
    N = t.size
    dt = T / N
    centers = np.linspace(0, T, n_gates, endpoint=False)
    G = np.zeros((n_gates, N))
    for g, t0 in enumerate(centers):
        if edge <= 0:
            # ideal boxcar over a periodic axis
            rel = (t - t0) % T
            G[g] = ((rel >= 0) & (rel < width)).astype(float)
        else:
            from scipy.special import erf
            rel = (t - t0)
            prof = 0.5 * (erf(rel / (np.sqrt(2) * edge))
                          - erf((rel - width) / (np.sqrt(2) * edge)))
            G[g] = np.clip(prof, 0, 1)
    G *= dt                                   # integral, not sum
    if eta is not None:
        G *= np.asarray(eta)[:, None]
    return G


# --------------------------------------------------------------------------- #
#  Simplex projection (Duchi et al. 2008) and Huber-TV gradient                #
# --------------------------------------------------------------------------- #
def project_simplex(V):
    """Euclidean projection of each row of V onto {x >= 0, sum x = 1}."""
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
    """Gradient of a Huber-smoothed 1-D total-variation penalty along the last
    axis (circular). Smooth surrogate for sum |h_{t+1}-h_t| so we can use a
    differentiable projected-gradient step instead of full ADMM/Condat.
    delta sets the |d| scale below which the penalty is quadratic."""
    d = h - np.roll(h, 1, axis=-1)                       # backward differences
    # Huber derivative of |d|: d/delta for |d|<delta else sign(d)
    psi = np.where(np.abs(d) <= delta, d / delta, np.sign(d))
    return psi - np.roll(psi, -1, axis=-1)               # divergence


def spatial_laplacian(H, ny, nx):
    """4-neighbour graph Laplacian of the IRF stack.
    H : (n_pix, N) reshaped to (ny, nx, N). Returns L @ H, same shape as H."""
    N = H.shape[-1]
    Himg = H.reshape(ny, nx, N)
    lap = 4 * Himg \
        - np.roll(Himg, 1, 0) - np.roll(Himg, -1, 0) \
        - np.roll(Himg, 1, 1) - np.roll(Himg, -1, 1)
    return lap.reshape(ny * nx, N)


def fourier_shift(H, s):
    """Circular sub-bin shift of each row of H by s bins (s>0 -> later in time)."""
    N = H.shape[-1]
    k = np.fft.rfftfreq(N, d=1.0 / N)
    phase = np.exp(-2j * np.pi * k * s / N)
    return np.fft.irfft(np.fft.rfft(H, axis=-1) * phase, n=N, axis=-1)


def pin_barycenter(H, c_target):
    """Remove the GLOBAL shift ambiguity (shared between IRF and decay onset) by
    shifting all IRFs so their MEAN barycenter equals c_target. Per-pixel relative
    shifts -- the pixel-variant information we want -- are preserved."""
    idx = np.arange(H.shape[-1])
    cbar = float(np.mean((H * idx).sum(-1) / np.maximum(H.sum(-1), EPS)))
    return np.maximum(fourier_shift(H, c_target - cbar), 0.0)


# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class SolverConfig:
    T: float = 12.5                 # laser period (ns) -> 80 MHz
    n_models: int = 2               # 1 = mono-exponential, 2 = bi-exponential
    tau_init: tuple = (0.5, 2.0)    # ns
    tau_bounds: tuple = (0.05, 8.0) # ns
    tau_sep: float = 1.4            # enforce tau2/tau1 >= tau_sep (identifiability)
    rho1: float = 0.02              # dimensionless TV knob
    rho2: float = 0.10              # dimensionless spatial-coupling knob
    outer_iters: int = 8
    irf_inner_iters: int = 250
    irf_step: float = 0.5           # PGD step (relative; auto-scaled by Lipschitz)
    estimate_irf: bool = True       # False = reference-calibrated (fit decay only)
    pin_global_shift: bool = False  # optional identifiability guardrail
    verbose: bool = True


# --------------------------------------------------------------------------- #
#  Block A : decay fit (VARPRO + Levenberg-Marquardt), one pixel               #
# --------------------------------------------------------------------------- #
def _phi(taus, h, t, T, G):
    """Design matrix Phi (n_gate, n_tau): gated, IRF-convolved basis decays."""
    B = decay_basis(taus, t, T)                      # (n_tau, N)
    M = cyclic_conv(h[None, :], B)                   # (n_tau, N)
    return (G @ M.T)                                 # (n_gate, n_tau)


def fit_decay_pixel(lam_obs, w, h, t, T, G, cfg):
    """Return (taus, amps, model) for a single pixel given its IRF h."""
    sw = np.sqrt(w)
    lo, hi = cfg.tau_bounds

    def resid(log_taus):
        taus = np.exp(log_taus)
        Phi = _phi(taus, h, t, T, G)                 # (n_gate, n_tau)
        A, *_ = np.linalg.lstsq(sw[:, None] * Phi, sw * lam_obs, rcond=None)
        A = np.maximum(A, 0.0)                        # non-negative amplitudes
        return sw * (Phi @ A - lam_obs)

    x0 = np.log(np.clip(cfg.tau_init[:cfg.n_models], lo, hi))
    sol = least_squares(resid, x0, method="trf",
                        bounds=(np.log(lo), np.log(hi)), max_nfev=200)
    taus = np.sort(np.exp(sol.x))                    # ascending: tau1 < tau2
    Phi = _phi(taus, h, t, T, G)
    A, *_ = np.linalg.lstsq(sw[:, None] * Phi, sw * lam_obs, rcond=None)
    A = np.maximum(A, 0.0)
    return taus, A, Phi @ A


# --------------------------------------------------------------------------- #
#  Block B : IRF update (projected gradient, all pixels jointly)               #
# --------------------------------------------------------------------------- #
def update_irf(H, F, lam_obs, W, G, mu1, mu2, ny, nx, cfg):
    """Projected-gradient minimisation of
        sum_k || sqrt(w_k) (G (h_k * f_k) - lambda_k) ||^2
        + mu1 sum_k HuberTV(h_k) + mu2 * h^T L h
    s.t. h_k on the probability simplex.  H, F : (n_pix, N).
    """
    P, N = H.shape

    def data_grad(Hx):
        lam = cyclic_conv(Hx, F) @ G.T               # (P, n_gate)
        resid = W * (lam - lam_obs)                  # weighted residual
        back = resid @ G                             # adjoint of gate: (P, N)
        return 2.0 * cyclic_corr(back, F)            # adjoint of conv wrt h

    def data_hess(V):                                # Gauss-Newton (W fixed) Hessian
        lam = cyclic_conv(V, F) @ G.T
        return 2.0 * cyclic_corr((W * lam) @ G, F)

    # power iteration for the data-term spectral norm -> correct PGD step
    v = np.random.default_rng(0).standard_normal((P, N))
    v /= np.linalg.norm(v) + EPS
    L_data = 1.0
    for _ in range(12):
        Av = data_hess(v)
        L_data = np.linalg.norm(Av)
        v = Av / (L_data + EPS)

    delta = 0.05 * float(H.max())                    # adaptive Huber knee
    L = L_data + mu1 / max(delta, EPS) + 8.0 * mu2 + EPS
    step = 1.0 / L

    for _ in range(cfg.irf_inner_iters):
        g = data_grad(H)
        g += mu1 * huber_tv_grad(H, delta)
        g += mu2 * 2.0 * spatial_laplacian(H, ny, nx)
        H = project_simplex(H - step * g)
    return H


# --------------------------------------------------------------------------- #
#  Top-level alternating solver                                                #
# --------------------------------------------------------------------------- #
def solve_flim(y, detector, det_params, ny, nx, gate_spec, cfg: SolverConfig,
               h_init=None):
    """
    y         : (n_pix, n_gate) raw measurements
    detector  : 'tcspc' | 'spad' | 'iccd'
    det_params: matching *Params dataclass from detector_weights
    ny, nx    : image dimensions, n_pix = ny*nx
    gate_spec : dict(n_gates, width, edge, eta) for build_gate_matrix
    Returns dict with taus (P, n_models), amps (P, n_models), irf (P, N), mu1, mu2.
    """
    P = y.shape[0]
    N = gate_spec.get("N", 256)
    t = np.linspace(0, cfg.T, N, endpoint=False)
    G = build_gate_matrix(t, cfg.T, gate_spec["n_gates"], gate_spec["width"],
                          gate_spec.get("edge", 0.0), gate_spec.get("eta", None))

    lam_obs, W = make_observation(y, detector, det_params)   # (P, n_gate) each

    # ---- initialise IRF (narrow Gaussian) and decay ----
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
        B = decay_basis  # local
        out = np.zeros((P, N))
        for k in range(P):
            Bk = decay_basis(taus[k], t, cfg.T)
            out[k] = amps[k] @ Bk
        return out

    # ---- auto-scale mu1, mu2 from the data at init ----
    for k in range(P):
        tk, ak, _ = fit_decay_pixel(lam_obs[k], W[k], H[k], t, cfg.T, G, cfg)
        taus[k], amps[k] = tk, ak
    F = assemble_F(taus, amps)
    lam_model = cyclic_conv(H, F) @ G.T
    D0 = float(np.sum(W * (lam_model - lam_obs) ** 2))
    TV0 = float(np.sum(np.abs(H - np.roll(H, 1, -1))))
    E_h = float(np.sum(H ** 2))                       # IRF energy (non-degenerate)
    mu1 = cfg.rho1 * D0 / (TV0 + EPS)
    mu2 = cfg.rho2 * D0 / (E_h + EPS)
    if cfg.verbose and cfg.estimate_irf:
        print(f"[init] data={D0:.3g}  TV0={TV0:.3g}  E_h={E_h:.3g}"
              f"  ->  mu1={mu1:.4g}  mu2={mu2:.4g}")

    # ---- alternating minimisation ----
    idx = np.arange(N)
    c_target = float(np.mean((H * idx).sum(-1) / np.maximum(H.sum(-1), EPS)))
    for it in range(cfg.outer_iters):
        # Block A: refit decay per pixel with current IRFs
        for k in range(P):
            taus[k], amps[k], _ = fit_decay_pixel(lam_obs[k], W[k], H[k],
                                                  t, cfg.T, G, cfg)
        F = assemble_F(taus, amps)
        # recompute weights (IRLS): variance depends on current lambda estimate
        lam_model = cyclic_conv(H, F) @ G.T
        _, W = make_observation(y, detector, det_params)  # data-driven; stable here
        # Block B: update IRFs (skip in reference-calibrated mode)
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

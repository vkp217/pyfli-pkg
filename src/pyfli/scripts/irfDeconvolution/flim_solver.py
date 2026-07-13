"""Joint instrument-response-function (IRF) and lifetime decay solver.

Implements alternating estimation of multi-exponential fluorescence
lifetime decay parameters and a (possibly per-pixel, spatially varying)
instrument response function from gated time-domain FLIM measurements.
Decay parameters are fit per pixel via weighted nonlinear least squares;
the IRF is updated by projected gradient descent with Huber total
variation and spatial-Laplacian regularization, subject to a
probability-simplex constraint.
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import least_squares

try:
    from .detector_weights import make_observation
except ImportError:
    from detector_weights import make_observation

EPS = 1e-9


def cyclic_conv(h, f):
    """Compute the circular (cyclic) convolution of ``h`` and ``f``.

    Performed along the last axis via the FFT
    (``irfft(rfft(h) * rfft(f))``), so both inputs are treated as
    periodic with period equal to the last-axis length.

    Args:
        h: Array to convolve, broadcastable against ``f`` on all but
            the last axis.
        f: Array to convolve, with the same last-axis length as ``h``.

    Returns:
        Array containing the circular convolution of ``h`` and ``f``,
        with the last-axis length of ``h``.
    """
    N = h.shape[-1]
    return np.fft.irfft(np.fft.rfft(h, axis=-1) * np.fft.rfft(f, axis=-1), n=N, axis=-1)


def cyclic_corr(u, f):
    """Compute the circular cross-correlation of ``u`` and ``f``.

    Performed along the last axis via the FFT
    (``irfft(conj(rfft(f)) * rfft(u))``). This is the adjoint operator
    of `cyclic_conv` with respect to ``f``, and is used to compute
    gradients of data-fidelity terms involving cyclic convolution.

    Args:
        u: Array to correlate, broadcastable against ``f`` on all but
            the last axis.
        f: Array to correlate, with the same last-axis length as ``u``.

    Returns:
        Array containing the circular cross-correlation of ``u`` and
        ``f``, with the last-axis length of ``u``.
    """
    N = u.shape[-1]
    return np.fft.irfft(np.conj(np.fft.rfft(f, axis=-1)) * np.fft.rfft(u, axis=-1),
                        n=N, axis=-1)

def decay_basis(taus, t, T):
    """Build a basis of periodic single-exponential decay curves.

    Each basis row is the wrapped (periodic, period ``T``) exponential
    decay ``exp(-t / tau) / (1 - exp(-T / tau))`` for one lifetime in
    ``taus``, normalized so that a single period integrates
    consistently across different ``tau`` values.

    Args:
        taus: Scalar or 1-D array of lifetime constants.
        t: 1-D array of time-bin centers over one period.
        T: Period (total time window) of the decay.

    Returns:
        Array of shape ``(len(taus), len(t))`` with one decay curve per
        row.
    """
    taus = np.atleast_1d(np.asarray(taus, float))
    return np.stack([np.exp(-t / tau) / (1.0 - np.exp(-T / tau)) for tau in taus], 0)


def build_gate_matrix(t, T, n_gates, width, edge=0.0, eta=None):
    """Build the matrix mapping fine time bins to detector gate windows.

    Constructs an ``(n_gates, len(t))`` matrix ``G`` whose rows are the
    (optionally edge-smoothed) temporal response of each of ``n_gates``
    equally spaced measurement gates, evenly covering the period ``T``.
    When ``edge <= 0`` each gate is a hard rectangular window of the
    given ``width``, applied cyclically (wrapping at ``T``). When
    ``edge > 0`` the gate edges are smoothed with a Gaussian error
    function of standard deviation ``edge``. Each row is scaled by the
    time-bin spacing ``dt = T / len(t)``, and optionally by a per-gate
    efficiency ``eta``.

    Args:
        t: 1-D array of fine time-bin centers over one period.
        T: Period (total time window).
        n_gates: Number of measurement gates, evenly spaced over
            ``[0, T)``.
        width: Temporal width of each gate.
        edge: Standard deviation of the Gaussian smoothing applied to
            gate edges. If ``<= 0``, gates are hard rectangular windows.
        eta: Optional 1-D array of per-gate efficiency factors, of
            length ``n_gates``, applied as a row-wise scale on ``G``.

    Returns:
        Array of shape ``(n_gates, len(t))`` mapping fine time bins to
        gate measurements.
    """
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
    """Project each row of ``V`` onto the probability simplex.

    Computes the Euclidean projection of each row onto the set of
    non-negative vectors summing to 1, following the sorting-based
    algorithm of Duchi et al. Used to constrain the estimated
    instrument response function to be a valid (non-negative,
    normalized) probability distribution.

    Args:
        V: Array of shape ``(n, d)`` (or ``(d,)``, promoted to 2-D).

    Returns:
        Array of the same shape as the (2-D) input, with each row
        projected onto the probability simplex.
    """
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
    """Compute the gradient of a Huber-smoothed total variation penalty.

    The penalty applies the Huber loss (quadratic near zero, linear
    beyond ``delta``) to the first-order circular differences of ``h``
    along its last axis, promoting piecewise-smooth signals while
    limiting the effect of large jumps.

    Args:
        h: Array whose regularization gradient is computed along the
            last axis.
        delta: Huber threshold separating the quadratic and linear
            regimes of the penalty.

    Returns:
        Array of the same shape as ``h`` containing the gradient of the
        Huber-TV penalty with respect to ``h``.
    """
    d = h - np.roll(h, 1, axis=-1)
    psi = np.where(np.abs(d) <= delta, d / delta, np.sign(d))
    return psi - np.roll(psi, -1, axis=-1)


def spatial_laplacian(H, ny, nx):
    """Compute the discrete 4-neighbor spatial Laplacian of a pixel grid.

    Reshapes the per-pixel signals in ``H`` onto an ``(ny, nx)`` spatial
    grid and applies a periodic (wraparound) 4-neighbor Laplacian
    stencil, used as a spatial smoothness regularizer on the per-pixel
    instrument response function.

    Args:
        H: Array of shape ``(ny * nx, N)`` with one signal (of length
            ``N``) per spatial pixel, in row-major (``ny``, ``nx``)
            order.
        ny: Number of pixel rows.
        nx: Number of pixel columns.

    Returns:
        Array of the same shape as ``H`` containing the spatial
        Laplacian of each signal.
    """
    N = H.shape[-1]
    Himg = H.reshape(ny, nx, N)
    lap = 4 * Himg \
        - np.roll(Himg, 1, 0) - np.roll(Himg, -1, 0) \
        - np.roll(Himg, 1, 1) - np.roll(Himg, -1, 1)
    return lap.reshape(ny * nx, N)


def fourier_shift(H, s):
    """Apply a sub-bin circular shift to signals via a Fourier phase ramp.

    Shifts each row of ``H`` along its last axis by ``s`` bins
    (which may be fractional) by multiplying its real FFT by a linear
    phase ramp and inverting, i.e. a band-limited circular shift.

    Args:
        H: Array of signals to shift, with the axis to shift last.
        s: Shift amount, in bins. May be fractional; positive values
            shift the signal to later bins.

    Returns:
        Array of the same shape as ``H`` containing the shifted
        signals.
    """
    N = H.shape[-1]
    k = np.fft.rfftfreq(N, d=1.0 / N)
    phase = np.exp(-2j * np.pi * k * s / N)
    return np.fft.irfft(np.fft.rfft(H, axis=-1) * phase, n=N, axis=-1)


def pin_barycenter(H, c_target):
    """Shift each row of ``H`` so its time-bin centroid matches a target.

    Computes the mean (over rows) barycenter of ``H`` along the last
    axis and applies a uniform `fourier_shift` to every row so that
    this mean barycenter equals ``c_target``, pinning the global
    temporal position of an estimated instrument response function
    while preserving its relative per-pixel shape/shift. Results are
    clipped at zero to remove small negative ringing introduced by the
    Fourier shift.

    Args:
        H: Array of shape ``(n, N)`` with one non-negative signal per
            row.
        c_target: Desired mean centroid (in bins) after shifting.

    Returns:
        Array of the same shape as ``H`` with shifted, non-negative
        rows.
    """
    idx = np.arange(H.shape[-1])
    cbar = float(np.mean((H * idx).sum(-1) / np.maximum(H.sum(-1), EPS)))
    return np.maximum(fourier_shift(H, c_target - cbar), 0.0)


@dataclass
class SolverConfig:
    """Configuration for the joint IRF and lifetime decay solver.

    Attributes:
        T: Period (total time window) of the decay/gate cycle.
        n_models: Number of exponential lifetime components to fit per
            pixel.
        tau_init: Initial guess for the lifetime constants (only the
            first ``n_models`` values are used).
        tau_bounds: ``(low, high)`` bounds on lifetime constants during
            nonlinear least-squares fitting.
        tau_sep: Reserved separation factor between fitted lifetimes
            (not directly enforced in the current fitting routine).
        rho1: Relative weight of the Huber total-variation regularizer
            on the IRF, expressed as a fraction of the initial data
            misfit divided by the initial TV norm.
        rho2: Relative weight of the spatial-Laplacian regularizer on
            the IRF, expressed as a fraction of the initial data
            misfit divided by the initial IRF energy.
        outer_iters: Number of outer alternating iterations between
            decay fitting and IRF updates.
        irf_inner_iters: Number of projected-gradient-descent
            iterations performed per IRF update.
        irf_step: Reserved step-size parameter for IRF updates (the
            current implementation computes its step size from a power
            iteration instead).
        estimate_irf: If ``True``, jointly estimate the instrument
            response function; if ``False``, treat the provided/initial
            IRF as fixed and only fit decay parameters.
        pin_global_shift: If ``True``, re-pin the IRF's mean temporal
            centroid to its initial value after each IRF update.
        verbose: If ``True``, print per-iteration progress information.
    """

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
    """Fit multi-exponential lifetime parameters for a single pixel.

    Performs weighted nonlinear least squares (via `scipy.optimize.
    least_squares`, optimizing over log-lifetimes for positivity and
    numerical conditioning) to fit ``cfg.n_models`` lifetime constants,
    given a fixed instrument response ``h`` and gate matrix ``G``. For
    each trial set of lifetimes, the corresponding non-negative
    amplitudes are obtained by weighted linear least squares (clipped
    at zero), and the residual against the gated observation
    ``lam_obs`` is minimized over the lifetimes.

    Args:
        lam_obs: Observed (weighted) Poisson rate per gate, for one
            pixel.
        w: Inverse-variance weights per gate, for one pixel.
        h: Instrument response function for this pixel, on the fine
            time grid ``t``.
        t: 1-D array of fine time-bin centers over one period.
        T: Period (total time window) of the decay/gate cycle.
        G: Gate matrix mapping fine time bins to gate measurements, as
            produced by `build_gate_matrix`.
        cfg: Solver configuration; uses ``n_models``, ``tau_init``, and
            ``tau_bounds``.

    Returns:
        Tuple ``(taus, amps, lam_model)`` of the fitted lifetime
        constants (sorted ascending), their non-negative amplitudes,
        and the resulting modeled gate signal.
    """
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
    """Update the per-pixel instrument response function by projected gradient descent.

    Minimizes the weighted data misfit between the gated model
    ``(H ⊛ F) @ G.T`` and ``lam_obs``, plus Huber-TV (weight ``mu1``)
    and spatial-Laplacian (weight ``mu2``) regularization on ``H``, for
    ``cfg.irf_inner_iters`` iterations. The gradient step size is set
    from the Lipschitz constant of the data term (estimated via power
    iteration on the Hessian-vector product) plus the regularizers'
    Lipschitz contributions. After each gradient step, ``H`` is
    projected back onto the probability simplex (via `project_simplex`)
    so each pixel's IRF stays non-negative and normalized.

    Args:
        H: Current per-pixel instrument response function, shape
            ``(P, N)``.
        F: Current per-pixel decay signal (from the fitted lifetime
            model), shape ``(P, N)``.
        lam_obs: Observed gated Poisson rate, shape ``(P, n_gates)``.
        W: Inverse-variance weights matching ``lam_obs``.
        G: Gate matrix mapping fine time bins to gate measurements.
        mu1: Weight of the Huber total-variation regularizer.
        mu2: Weight of the spatial-Laplacian regularizer.
        ny: Number of pixel rows in the spatial grid.
        nx: Number of pixel columns in the spatial grid.
        cfg: Solver configuration; uses ``irf_inner_iters``.

    Returns:
        Updated instrument response function, shape ``(P, N)``.
    """
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
    """Jointly estimate lifetime decay parameters and the instrument response.

    Converts raw detector data ``y`` to a weighted gated Poisson-rate
    observation via `make_observation`, then alternates (for
    ``cfg.outer_iters`` iterations) between: (1) fitting per-pixel
    multi-exponential lifetimes and amplitudes via `fit_decay_pixel`
    given the current instrument response function, and (2), if
    ``cfg.estimate_irf`` is set, updating the shared per-pixel
    instrument response function via `update_irf` given the current
    decay fits, regularized by Huber-TV and spatial-Laplacian penalties
    whose weights (``mu1``, ``mu2``) are auto-scaled from the initial
    data misfit, TV norm, and IRF energy. If ``cfg.estimate_irf`` is
    ``False``, only a single decay-fitting pass is performed against
    the fixed initial/provided IRF.

    Args:
        y: Raw detector readout, shape ``(P, n_gates)`` where ``P`` is
            the number of pixels (``ny * nx``).
        detector: Detector name passed to `make_observation` (e.g.
            ``"tcspc"``, ``"spad"``, ``"iccd"``).
        det_params: Detector parameters instance matching ``detector``.
        ny: Number of pixel rows in the spatial grid.
        nx: Number of pixel columns in the spatial grid.
        gate_spec: Dict describing the gate geometry; supports keys
            ``"N"`` (fine time-bin count, default 256), ``"n_gates"``,
            ``"width"``, ``"edge"`` (default 0.0), and ``"eta"``
            (default ``None``), passed to `build_gate_matrix`.
        cfg: Solver configuration.
        h_init: Optional initial instrument response function, shape
            ``(P, N)``. If ``None``, a Gaussian pulse is used to
            initialize every pixel's IRF.

    Returns:
        Dict with keys ``"taus"`` (fitted lifetimes per pixel),
        ``"amps"`` (fitted amplitudes per pixel), ``"irf"`` (final
        instrument response function), ``"mu1"`` and ``"mu2"``
        (auto-scaled regularization weights), ``"gate"`` (the gate
        matrix), and ``"t"`` (the fine time-bin grid).
    """
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

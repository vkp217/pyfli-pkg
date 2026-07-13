"""Demo script exercising `solve_flim` on simulated multi-detector FLIM data.

Builds a synthetic two-exponential ground-truth decay and a spatially
varying instrument response function, simulates TCSPC, SPAD, and ICCD
detector readouts from it, and runs `solve_flim` in both a
reference-calibrated mode (known IRF) and a blind joint IRF/decay
estimation mode, printing recovered lifetimes/amplitudes and IRF
recovery error against the known ground truth.
"""

import numpy as np
from detector_weights import ICCDParams, SPADParams, TCSPCParams, make_observation
from flim_solver import (SolverConfig, solve_flim, build_gate_matrix,
                         decay_basis, cyclic_conv)

rng = np.random.default_rng(7)
T, N = 12.5, 256
ny, nx = 8, 8
P = ny * nx
t = np.linspace(0, T, N, endpoint=False)
dt = T / N
n_gates, width = 256, 1.5 * dt
G = build_gate_matrix(t, T, n_gates, width, edge=0.4 * dt)

tau_true = np.array([0.4, 2.2])
amp_true = np.array([0.7, 0.3])
f_true = amp_true @ decay_basis(tau_true, t, T)

H_true = np.zeros((P, N))
for k in range(P):
    iy, ix = divmod(k, nx)
    c = N // 8 + 0.6 * (ix - nx / 2) + 0.6 * (iy - ny / 2)
    g = np.exp(-0.5 * ((np.arange(N) - c) / 3.0) ** 2)
    H_true[k] = g / g.sum()

lam_clean = cyclic_conv(H_true, np.tile(f_true, (P, 1))) @ G.T


def make_counts(detector, params, budget):
    """Simulate raw detector counts for a given detector and photon budget.

    Rescales the noiseless gated signal `lam_clean` so each pixel's
    total rate matches ``budget``, then draws simulated readout noise
    matching the given detector's physical model: gamma-distributed
    intensifier gain plus Gaussian read noise on Poisson photoelectron
    counts for ICCD, binomial hit counts for SPAD, and Poisson photon
    counts for TCSPC.

    Args:
        detector: Detector name, one of ``"iccd"``, ``"spad"``, or
            ``"tcspc"``.
        params: Detector parameters instance matching ``detector``
            (`ICCDParams`, `SPADParams`, or `TCSPCParams`).
        budget: Target total photon rate per pixel, used to rescale
            `lam_clean` before simulating noise.

    Returns:
        Array of simulated raw detector counts, same shape as
        `lam_clean`, or ``None`` if ``detector`` is not recognized.
    """
    lam = lam_clean * budget / lam_clean.sum(1, keepdims=True)
    if detector == "iccd":
        npe = rng.poisson(lam)
        return rng.gamma(np.maximum(npe, 0) + 1e-9, params.G0) \
               + rng.normal(0, params.sigma_r, lam.shape)
    if detector == "spad":
        prob = 1 - np.exp(-lam / params.n_ex)
        return rng.binomial(int(params.n_ex), prob).astype(float)
    if detector == "tcspc":
        return rng.poisson(lam).astype(float)


def report(name, res):
    """Print recovered lifetimes/amplitudes and their error against ground truth.

    Args:
        name: Label identifying the detector/run, printed in the
            output.
        res: Result dict returned by `solve_flim`, containing ``taus``
            and ``amps`` per pixel.
    """
    te = res["taus"].mean(0)
    err = 100 * np.abs(te - tau_true) / tau_true
    am = (res["amps"] / res["amps"].sum(1, keepdims=True)).mean(0)
    print(f"  {name:6s}  tau={te.round(3)}  amp={am.round(3)}  "
          f"|  err% = {err.round(1)}")


gate_spec = dict(N=N, n_gates=n_gates, width=width, edge=0.4 * dt)
print("ground truth: tau =", tau_true, " amp =", amp_true)
print("\n--- Mode 1: reference-calibrated (IRF known, fit decay) ---")
cfg_ref = SolverConfig(T=T, n_models=2, tau_init=(0.6, 1.5),
                       estimate_irf=False, outer_iters=1, verbose=False)

detectors = [
    ("tcspc", TCSPCParams(n_ex=2e6), 2.0e5),
    ("spad",  SPADParams(n_ex=1e6),  2.0e5),
    ("iccd",  ICCDParams(G0=12.0, F2=2.0, sigma_r=8.0), 6.0e4),
]
for det, p, budget in detectors:
    y = make_counts(det, p, budget)
    res = solve_flim(y, det, p, ny, nx, gate_spec, cfg_ref, h_init=H_true.copy())
    report(det.upper(), res)

print("\n--- Mode 2: blind joint IRF + decay (ICCD), strong spatial prior ---")
y = make_counts("iccd", detectors[2][1], detectors[2][2])
cfg_blind = SolverConfig(T=T, n_models=2, tau_init=(0.5, 2.0),
                         estimate_irf=True, rho1=0.3, rho2=5.0,
                         outer_iters=10, irf_inner_iters=400, verbose=False)
res = solve_flim(y, "iccd", detectors[2][1], ny, nx, gate_spec, cfg_blind)
report("ICCD", res)
cen_t = (H_true * np.arange(N)).sum(1)
cen_e = (res["irf"] * np.arange(N)).sum(1)
print(f"  IRF centroid RMSE = {np.sqrt(np.mean((cen_t-cen_e)**2)):.2f} bins"
      f"  (auto mu1={res['mu1']:.3g}, mu2={res['mu2']:.3g})")

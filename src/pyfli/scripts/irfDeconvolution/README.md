# FLIM detector-specific deconvolution — weights, regularization, solver

Three files:

- `detector_weights.py` — detector-specific weight/transform pair for each sensor.
- `flim_solver.py` — cyclic-convolution + gate operators, VARPRO decay fit,
  projected-gradient IRF update with TV + spatial coupling, alternating loop.
- `demo_flim.py` — synthetic verification across all three detectors.

## 1. Specialized weight expressions

Every detector is reduced to weighted least squares in the **ideal-intensity
domain** λ (the photons/counts predicted by `λ = Gate @ (h ⊛ f)`). Each detector
supplies two functions: `to_lambda` (measurement → λ estimate) and
`lambda_weight` (inverse variance in the λ domain).

| Detector | Statistics | Var(measurement) | Weight in λ domain |
|---|---|---|---|
| TCSPC | Poisson (+ pile-up) | Var(N) = λ | `w = 1/λ`, ×`(n_ex−n)/n_ex` inflation under pile-up |
| SPAD (gated) | Binomial over n_ex cycles, p = 1−e^(−λ/n_ex) | n_ex·p(1−p) | `w = (n_ex − y) / (n_ex · y)` |
| ICCD (gated) | Compound Poisson–Gaussian (MCP gain) | F²·G₀²·λ + σ_r² | `w = 1 / (F²·λ + (σ_r/G₀)²)` |

Key physics encoded in the weights:
- TCSPC pile-up uses the Coates inversion `Λ = −n_ex·ln(1 − n/n_ex)` so the
  early-time bias is linearized before fitting.
- SPAD reuses the same dead-time/"one photon per gate" inversion as its
  variance-stabilizing transform.
- ICCD: the excess-noise factor `F² = 2` (exponential MCP single-electron gain)
  inflates shot noise, while read noise is suppressed by the gain `G₀`.
  A generalized Anscombe transform is also provided as an alternative VST.

## 2. Concrete regularization weights μ1 (TV), μ2 (spatial)

μ1 and μ2 are **auto-scaled from the data** so the portable knobs are
dimensionless ratios, stable across datasets, detectors, and photon budgets:

```
mu1 = rho1 * D0 / TV0      # TV penalty as a fraction of the initial data misfit
mu2 = rho2 * D0 / E_h      # spatial coupling normalized by IRF energy (non-degenerate)
```

Recommended defaults:

| Knob | Default | Meaning | Tune up when |
|---|---|---|---|
| `rho1` (TV) | **0.02** | edge/ringing suppression on h_k | IRF is noisy / rings |
| `rho2` (spatial) | **0.10** | neighbour-IRF coupling | array is undersampled / few gates |

For the 8×8 / 256-bin ICCD demo these resolve to concrete values around
`mu1 ≈ 1e2–2e3` and `mu2 ≈ 1e3–1e5` depending on photon budget — printed at
run time. For blind, undersampled regimes raise `rho2` to 1–5 (strong spatial
prior) and `rho1` to 0.1–0.3.

## 3. Verified behaviour

Reference-calibrated mode (IRF known from a scatter measurement, fit decay only)
recovers the bi-exponential lifetimes essentially exactly, which validates the
detector weights + forward model + VARPRO core:

```
ground truth: tau = [0.4 2.2]  amp = [0.7 0.3]
  TCSPC   tau=[0.398 2.188]   err% = [0.5 0.6]
  SPAD    tau=[0.396 2.183]   err% = [0.9 0.8]
  ICCD    tau=[0.378 2.079]   err% = [5.5 5.5]   (larger: MCP excess noise F^2=2 + read noise)
```

Fully-blind nonparametric per-pixel IRF estimation is **ill-posed** (≈18–20%
lifetime error in the demo) and needs one of: a reference scatter IRF, a
low-rank eigen-IRF basis, or strong spatial coupling. This matches the theory:
the convolution shift/scale ambiguity is only broken by the IRF constraints
(h ≥ 0, Σh = 1), the parametric causal decay model, and inter-pixel coupling.
The optional `pin_global_shift` guardrail removes the global shift ambiguity but
must be used with care (it can destabilize a two-component fit if mis-targeted).

## Usage

```python
from detector_weights import ICCDParams
from flim_solver import SolverConfig, solve_flim

cfg = SolverConfig(T=12.5, n_models=2, rho1=0.02, rho2=0.10,
                   estimate_irf=True)            # or False for reference mode
res = solve_flim(y, "iccd", ICCDParams(G0=12, F2=2, sigma_r=8),
                 ny, nx, gate_spec, cfg)
res["taus"], res["amps"], res["irf"]
```

The gate width enters via `gate_spec["width"]` and the smoothed-boxcar edge via
`gate_spec["edge"]`; both fold into the effective IRF `h_eff = h_opt ⊛ gate`.

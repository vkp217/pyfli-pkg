"""
main.py
=======
Demonstration script for the phasor_flim package.

Runs all six acquisition modes described in Michalet 2021
(DOI: 10.1063/5.0027834), prints configuration summaries, and saves
two publication-quality figures.

Usage
-----
    python main.py                    # saves figures to ./output/
    python main.py --outdir /tmp/     # custom output directory
"""

from __future__ import annotations
import argparse
import pathlib
from dataclasses import replace

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering; remove for interactive use
import matplotlib.pyplot as plt

from config import AcquisitionConfig, AcquisitionMode
from phasors import (
    phasor_continuous,
    phasor_discrete,
    phasor_gated_single,
    phasor_gated_N,
    phasor_truncated,
    phasor_offset,
    phasor_from_config,
)
from locus import build_locus, build_loci, tau_grid, universal_semicircle
from lifetimes import phase_lifetime, modulus_lifetime, lifetime_from_phasor
from plot import plot_discrete_N_sweep, plot_phasor, plot_locus_comparison


# ──────────────────────────────────────────────────────────────────────────────
# Default experiment configuration (80 MHz Ti:Sa laser, 1st harmonic)
# ──────────────────────────────────────────────────────────────────────────────

BASE = AcquisitionConfig(
    T_ns       = 12.5,    # 80 MHz laser
    harmonic   = 1,
    tau_min_ns = 0.05,
    tau_max_ns = 10.0,
    n_tau_pts  = 800,
)


def make_all_configs() -> list[AcquisitionConfig]:
    """Return one AcquisitionConfig per acquisition mode."""
    return [
        replace(BASE, mode=AcquisitionMode.CONTINUOUS),
        replace(BASE, mode=AcquisitionMode.DISCRETE,     N_bins=64),
        replace(BASE, mode=AcquisitionMode.GATED_SINGLE, gate_width_frac=0.5),
        replace(BASE, mode=AcquisitionMode.GATED_N,      gate_width_frac=0.5, N_gates=4),
        replace(BASE, mode=AcquisitionMode.TRUNCATED,    T_rec_frac=0.8),
        replace(BASE, mode=AcquisitionMode.OFFSET,       t0_frac=0.1),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1 — all-modes comparison
# ──────────────────────────────────────────────────────────────────────────────

def fig_comparison(outdir: pathlib.Path) -> None:
    cfgs = make_all_configs()
    fig, ax = plot_locus_comparison(
        cfgs,
        title="Phasor SEPL — all acquisition modes",
        suptitle_note="Michalet, AIP Advances 11, 035331 (2021)",
    )
    path = outdir / "phasor_all_modes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2 — discrete N sweep
# ──────────────────────────────────────────────────────────────────────────────

def fig_discrete_sweep(outdir: pathlib.Path) -> None:
    fig, ax = plot_discrete_N_sweep(
        BASE,
        N_values=[2, 4, 8, 16, 64, 256],
    )
    path = outdir / "phasor_discrete_N_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3 — individual mode panels
# ──────────────────────────────────────────────────────────────────────────────

def fig_individual_modes(outdir: pathlib.Path) -> None:
    cfgs = make_all_configs()
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "Individual SEPL curves — Michalet 2021 (DOI: 10.1063/5.0027834)",
        fontsize=13
    )
    for ax, cfg in zip(axes.flat, cfgs):
        plot_phasor(cfg, ax=ax, show_universal=True, show_ticks=True)

    path = outdir / "phasor_individual_modes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved → {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Console summary
# ──────────────────────────────────────────────────────────────────────────────

def print_summary() -> None:
    print("\n" + "═" * 60)
    print("  phasor_flim — acquisition mode summary")
    print("═" * 60)
    for cfg in make_all_configs():
        print()
        print(cfg.describe())

    print("\n" + "═" * 60)
    print("  Sample phasor coordinates at τ = 2.0 ns")
    print("═" * 60)
    
    tau_sample = np.array([2.0])
    for cfg in make_all_configs():
        g, s = phasor_from_config(tau_sample, cfg)
        tau_ph = phase_lifetime(g, s, cfg)
        tau_m  = modulus_lifetime(g, s, cfg)
        print(
            f"  {cfg.mode.name:<14}  g={float(np.squeeze(g)):.4f}  s={float(np.squeeze(s)):.4f}"
            f"  τ_φ={float(np.squeeze(tau_ph)):.3f} ns  τ_m={float(np.squeeze(tau_m)):.3f} ns"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="phasor_flim demo")
    parser.add_argument("--outdir", default="output", help="Output directory for figures")
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print_summary()

    print("\nGenerating figures …")
    fig_comparison(outdir)
    fig_discrete_sweep(outdir)
    fig_individual_modes(outdir)
    print("\nDone.")


if __name__ == "__main__":
    main()

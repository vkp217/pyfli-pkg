"""
plot.py
=======
Matplotlib helpers for rendering phasor plots and SEPL comparisons.

All functions return (fig, ax) so callers retain full control.

Color palette mirrors the widget colour scheme; each acquisition mode has
a distinct colour defined in PALETTE.
"""

from __future__ import annotations
from dataclasses import replace
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .config import AcquisitionConfig, AcquisitionMode
from .locus import build_locus, universal_semicircle


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (mode → hex)
# ──────────────────────────────────────────────────────────────────────────────

PALETTE: dict[AcquisitionMode, str] = {
    AcquisitionMode.CONTINUOUS:   "#3266ad",   # blue
    AcquisitionMode.DISCRETE:     "#1d9e75",   # teal
    AcquisitionMode.GATED_SINGLE: "#d85a30",   # coral
    AcquisitionMode.GATED_N:      "#ba7517",   # amber
    AcquisitionMode.TRUNCATED:    "#d4537e",   # pink
    AcquisitionMode.OFFSET:       "#7f77dd",   # purple
}

MODE_LABELS: dict[AcquisitionMode, str] = {
    AcquisitionMode.CONTINUOUS:   "Continuous (universal semicircle)",
    AcquisitionMode.DISCRETE:     "Discrete N-bins (TCSPC)",
    AcquisitionMode.GATED_SINGLE: "Single square gate",
    AcquisitionMode.GATED_N:      "N equidistant gates",
    AcquisitionMode.TRUNCATED:    "Truncated decay",
    AcquisitionMode.OFFSET:       "IRF offset",
}


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _setup_axes(ax: Axes, title: str = "") -> None:
    """Apply standard phasor-plot aesthetics to *ax*."""
    ax.set_xlim(-0.04, 1.08)
    ax.set_ylim(-0.04, 0.62)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("g  (real / cosine component)", fontsize=11)
    ax.set_ylabel("s  (imaginary / sine component)", fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, pad=8)
    ax.axhline(0, color="0.7", linewidth=0.5, zorder=0)
    ax.axvline(0, color="0.7", linewidth=0.5, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.4, zorder=0)


def _draw_universal_semicircle(ax: Axes, alpha: float = 0.35) -> None:
    """Overlay the universal semicircle as a dashed grey reference."""
    g_uc, s_uc = universal_semicircle()
    ax.plot(
        g_uc, s_uc,
        color="0.5", linewidth=1.2, linestyle="--",
        alpha=alpha, label="Universal semicircle (∞ bins)",
        zorder=1,
    )
    ax.plot(0.5, 0.0, marker="o", color="0.5", markersize=4, alpha=alpha, zorder=1)


def _draw_lifetime_ticks(
    ax: Axes,
    g: np.ndarray,
    s: np.ndarray,
    tau: np.ndarray,
    cfg: AcquisitionConfig,
    color: str = "#e24b4a",
    step: float | None = None,
) -> None:
    """
    Mark and label selected lifetime values on a SEPL curve.

    Tick values are chosen automatically based on tau_max_ns unless *step*
    is given explicitly.
    """
    if step is None:
        tm = cfg.tau_max_ns
        step = 0.5 if tm <= 5 else 1.0 if tm <= 20 else 5.0 if tm <= 50 else 10.0

    tick_taus = np.arange(step, cfg.tau_max_ns + step * 0.5, step)

    for tt in tick_taus:
        idx = np.searchsorted(tau, tt)
        if idx >= len(tau):
            continue
        gp, sp = g[idx], s[idx]
        ax.plot(gp, sp, marker="o", color=color, markersize=4, zorder=5)
        # Outward normal (roughly perpendicular to the curve at this point)
        label = f"{tt:.0f} ns" if tt >= 1.0 else f"{tt:.1f} ns"
        ax.annotate(
            label,
            xy=(gp, sp),
            xytext=(gp + 0.02, sp + 0.02),
            fontsize=8,
            color=color,
            zorder=6,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Primary public function
# ──────────────────────────────────────────────────────────────────────────────

def plot_phasor(
    cfg: AcquisitionConfig,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (7, 5),
    show_universal: bool = True,
    show_ticks: bool = True,
    show_endpoints: bool = True,
    color: str | None = None,
    label: str | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot the SEPL (Single-Exponential Phasor Locus) for a single config.

    Parameters
    ----------
    cfg             : AcquisitionConfig
    ax              : Axes, optional   — draw into an existing axes.
    figsize         : tuple            — figure size if *ax* is None.
    show_universal  : bool             — overlay the universal semicircle.
    show_ticks      : bool             — mark lifetime values on the SEPL.
    show_endpoints  : bool             — mark τ→0 and τ→∞ endpoints.
    color           : str, optional    — override default mode colour.
    label           : str, optional    — override default legend label.
    title           : str, optional    — axes title.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    _setup_axes(ax, title=title or f"Phasor SEPL — {cfg.mode.name}")

    if show_universal:
        _draw_universal_semicircle(ax)

    g, s, tau = build_locus(cfg)
    c = color or PALETTE.get(cfg.mode, "#3266ad")
    lbl = label or MODE_LABELS.get(cfg.mode, cfg.mode.name)
    ax.plot(g, s, color=c, linewidth=2.0, label=lbl, zorder=3)

    if show_ticks:
        _draw_lifetime_ticks(ax, g, s, tau, cfg, color=c)

    if show_endpoints:
        for idx, lbl_ep, ha in [(0, "τ→0", "left"), (-1, "τ→∞", "right")]:
            ax.plot(g[idx], s[idx], "o", color=c, markersize=6, zorder=6)
            ax.annotate(lbl_ep, xy=(g[idx], s[idx]),
                        xytext=(g[idx] + (0.03 if ha == "left" else -0.03), s[idx] + 0.02),
                        fontsize=8, color=c, ha=ha)

    # Frequency annotation
    freq_str = (
        f"ω = {cfg.omega:.3f} rad/ns  |  "
        f"f = {cfg.frequency_MHz:.1f} MHz  |  "
        f"n = {cfg.harmonic}"
    )
    ax.text(0.98, 0.03, freq_str, transform=ax.transAxes,
            fontsize=8, ha="right", va="bottom", color="0.5")

    ax.legend(fontsize=9, loc="upper right", framealpha=0.8)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# Multi-mode comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_locus_comparison(
    cfgs: Sequence[AcquisitionConfig],
    *,
    figsize: tuple[float, float] = (8, 5.5),
    show_universal: bool = True,
    show_ticks: bool = True,
    colors: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    title: str = "Phasor SEPL comparison — Michalet 2021",
    suptitle_note: str = "",
) -> tuple[Figure, Axes]:
    """
    Overlay multiple SEPL curves on a single phasor plot.

    Parameters
    ----------
    cfgs            : sequence of AcquisitionConfig
    figsize         : tuple
    show_universal  : bool
    show_ticks      : bool  — only for the first curve to keep the plot readable.
    colors          : sequence of hex strings (defaults to PALETTE).
    labels          : sequence of strings (defaults to MODE_LABELS).
    title           : str
    suptitle_note   : str   — small subtitle below the title.

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax, title=title)
    if suptitle_note:
        ax.set_title(f"{title}\n{suptitle_note}", fontsize=11)

    if show_universal:
        _draw_universal_semicircle(ax)

    for i, cfg in enumerate(cfgs):
        c   = (colors[i] if colors else None) or PALETTE.get(cfg.mode, "#3266ad")
        lbl = (labels[i] if labels else None) or MODE_LABELS.get(cfg.mode, cfg.mode.name)
        g, s, tau = build_locus(cfg)
        ax.plot(g, s, color=c, linewidth=1.8, label=lbl, zorder=3 + i)

        if show_ticks and i == 0:
            _draw_lifetime_ticks(ax, g, s, tau, cfg, color=c)

        ax.plot(g[0], s[0], "o", color=c, markersize=5, zorder=8)
        ax.plot(g[-1], s[-1], "s", color=c, markersize=5, zorder=8)

    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.85,
              title="Acquisition mode", title_fontsize=8)
    return fig, ax


# ──────────────────────────────────────────────────────────────────────────────
# Discrete SEPL — N sweep
# ──────────────────────────────────────────────────────────────────────────────

def plot_discrete_N_sweep(
    base_cfg: AcquisitionConfig,
    N_values: Sequence[int] = (4, 8, 16, 64, 256),
    *,
    figsize: tuple[float, float] = (8, 5.5),
) -> tuple[Figure, Axes]:
    """
    Show how the discrete SEPL converges to the universal semicircle as N → ∞.

    Parameters
    ----------
    base_cfg : AcquisitionConfig   — mode will be overridden to DISCRETE.
    N_values : sequence of int     — bin counts to sweep.
    figsize  : tuple

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _setup_axes(ax, title=f"Discrete SEPL convergence  (T={base_cfg.T_ns} ns, n={base_cfg.harmonic})")
    _draw_universal_semicircle(ax, alpha=0.5)

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(N_values) - 1)) for i in range(len(N_values))]

    for N, c in zip(N_values, colors):
        cfg_n = replace(base_cfg, mode=AcquisitionMode.DISCRETE, N_bins=N)  # noqa: F811
        g, s, _ = build_locus(cfg_n)
        ax.plot(g, s, color=c, linewidth=1.6, label=f"N = {N}")

    ax.legend(fontsize=9, title="Bins N", title_fontsize=9,
              loc="upper right", framealpha=0.85)
    return fig, ax

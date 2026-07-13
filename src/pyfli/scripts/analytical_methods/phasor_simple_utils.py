"""Shared constants and low-level drawing helpers for phasor plots.

Provides the reference lifetime tick marks, the parametric "universal
circle" (the semicircle/circle traced by single-exponential lifetimes in
G/S phasor space) and small matplotlib-axis helpers reused by
`PhasorAnalyzer` and `PhasorPlotsMixin` when rendering phasor diagrams.
"""

import numpy as np

_TAU_MARKS_NS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                           0.9, 1, 1.5, 2, 3, 5, 7, 10])
_UNIVERSAL_CIRCLE_CENTER = (0.5, 0.0)
_UNIVERSAL_CIRCLE_RADIUS = 0.5


def _universal_circle_xy(n_points: int = 500, half_circle: bool = False):
    """Compute (G, S) coordinates tracing the phasor universal circle.

    Args:
        n_points: Number of points used to sample the circle.
        half_circle: If True, sample only the upper half (0 to pi);
            otherwise sample the full circle (0 to 2*pi).

    Returns:
        tuple[np.ndarray, np.ndarray]: The G and S coordinate arrays of
        the circle, centered at ``_UNIVERSAL_CIRCLE_CENTER`` with radius
        ``_UNIVERSAL_CIRCLE_RADIUS``.
    """
    theta = np.linspace(0, np.pi if half_circle else 2 * np.pi, n_points)
    cx, cy = _UNIVERSAL_CIRCLE_CENTER
    r = _UNIVERSAL_CIRCLE_RADIUS
    return cx + r * np.cos(theta), cy + r * np.sin(theta)


def _draw_lifetime_ticks(ax, G_mark, S_mark,
                         tick_length: float = 0.02,
                         text_offset: float = 0.035,
                         color: str = "black",
                         lw: float = 2,
                         fontsize: int = 7,
                         show_units: bool = False):
    """Draw radial tick marks and labels for reference lifetimes on a phasor axis.

    For each lifetime in ``_TAU_MARKS_NS``, draws a short tick centered on
    its (G, S) position, oriented along the radial direction from the
    universal-circle center, with a text label placed just beyond the tick.

    Args:
        ax: Matplotlib axis to draw on.
        G_mark: G coordinates of the lifetime marks (one per entry in
            ``_TAU_MARKS_NS``).
        S_mark: S coordinates of the lifetime marks (one per entry in
            ``_TAU_MARKS_NS``).
        tick_length: Length of each tick mark, in phasor-plot units.
        text_offset: Radial offset of the label text beyond the tick.
        color: Color used for ticks and labels.
        lw: Line width of the tick marks.
        fontsize: Font size of the labels.
        show_units: If True, append " ns" to each label; otherwise show
            just the numeric value.
    """
    cx, cy = _UNIVERSAL_CIRCLE_CENTER
    for tau, Gm, Sm in zip(_TAU_MARKS_NS, G_mark, S_mark):
        normal = np.array([Gm - cx, Sm - cy])
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm

        tick_start = np.array([Gm, Sm]) - tick_length * normal / 2
        tick_end   = np.array([Gm, Sm]) + tick_length * normal / 2
        ax.plot([tick_start[0], tick_end[0]], [tick_start[1], tick_end[1]],
                color=color, lw=lw)

        label    = f"{tau:.1f} ns" if show_units else f"{tau:.1f}"
        text_pos = tick_end + text_offset * normal
        ax.text(text_pos[0], text_pos[1], label,
                color=color, fontsize=fontsize, ha="center")


def _style_phasor_ax(ax, title: str = "Phasor Diagram",
                     xlim=(-0.1, 1.1), ylim=(0.0, 0.6), half_circle: bool = True):
    """Apply consistent axis labels, limits and styling to a phasor plot.

    Sets the G/S axis labels, title, equal aspect ratio, axis limits,
    dashed gridlines, zero-reference lines and inward tick marks.

    Args:
        ax: Matplotlib axis to style.
        title: Title to set on the axis.
        xlim: Tuple of (min, max) limits for the G axis.
        ylim: Tuple of (min, max) limits for the S axis.
        half_circle: Unused directly here; kept for call-site symmetry
            with other phasor-plot helpers.
    """
    ax.set_xlabel("G")
    ax.set_ylabel("S")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.axvline(0, color="black", lw=1)
    ax.tick_params(direction="in", length=6, width=1)

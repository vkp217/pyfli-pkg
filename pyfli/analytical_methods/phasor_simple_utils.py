import numpy as np

_TAU_MARKS_NS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                           0.9, 1, 1.5, 2, 3, 5, 7, 10])
_UNIVERSAL_CIRCLE_CENTER = (0.5, 0.0)
_UNIVERSAL_CIRCLE_RADIUS = 0.5


def _universal_circle_xy(n_points: int = 500, half_circle: bool = False):
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

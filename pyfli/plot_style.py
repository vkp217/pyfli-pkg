"""
plot_style.py
-------------
Shared line/marker color palette and outside-axes legend placement for the
plotting functions in :mod:`pyfli.analysis` and :mod:`pyfli.data_vnp`.

Every categorical palette in those two packages is built from `dark_palette()`
(or is `DARK_PALETTE` itself), so retheming every plot that uses one means
editing `VMAX`/`SMIN` (or a call's `base=` seaborn palette name) here rather
than touching each plotting function.
"""

import colorsys
from typing import Any

import seaborn as sns

VMAX = 0.55
SMIN = 0.55

LEGEND_FONTSIZE = 8


def darken_color(
    color: tuple[float, ...], vmax: float = VMAX, smin: float = SMIN
) -> tuple[float, ...]:
    """Cap an RGB(A) color's brightness and floor its saturation so it reads as dark."""
    r, g, b = color[:3]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2, g2, b2 = colorsys.hsv_to_rgb(h, max(s, smin), min(v, vmax))
    return (r2, g2, b2, color[3]) if len(color) == 4 else (r2, g2, b2)


def dark_palette(
    n_colors: int, base: str = "colorblind", vmax: float = VMAX, smin: float = SMIN
) -> list[tuple[float, ...]]:
    """Return `n_colors` dark, mutually distinguishable colors derived from a
    seaborn base palette (any name/list accepted by `seaborn.color_palette`)."""
    return [
        darken_color(c, vmax=vmax, smin=smin)
        for c in sns.color_palette(base, n_colors=n_colors)
    ]


DARK_PALETTE: list[tuple[float, ...]] = dark_palette(10)


def legend_outside(
    ax: Any,
    loc: str = "upper left",
    bbox_to_anchor: tuple[float, float] = (1.02, 1.0),
    borderaxespad: float = 0.0,
    **kwargs: Any,
) -> Any:
    """Draw ax's legend outside the axes box (to the right, by default) instead
    of overlapping the plotted data."""
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("fontsize", LEGEND_FONTSIZE)
    return ax.legend(
        loc=loc, bbox_to_anchor=bbox_to_anchor, borderaxespad=borderaxespad, **kwargs
    )

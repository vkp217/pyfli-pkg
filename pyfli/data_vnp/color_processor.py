"""
Build Matplotlib colormap variants for FLI map visualization.

This module belongs to :mod:`pyfli.data_vnp` and is part of PyFLI visualization,
normalization, plotting, and mono-versus-bi-exponential comparison tools. Public API
includes classes :class:`ColorProcessor`.
"""

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


class ColorProcessor:
    """
    Create Matplotlib colormaps tailored for FLI maps. It can force the lowest value to
    black or clip a colormap range to improve visual contrast.
    """

    def lowest_zero(self, cmap_name: str = "jet") -> Any:
        """
        Run the lowest zero routine.

        Parameters
        ----------
        cmap_name : str
            Name of the Matplotlib colormap to transform.

        Returns
        -------
        Any
            Object produced by lowest zero.
        """
        original_cmap = plt.get_cmap(cmap_name)
        colors = original_cmap(np.linspace(0, 1, 256))
        colors[0] = [0, 0, 0, 1]
        return ListedColormap(colors)

    def clip_crange(
        self,
        cmap_name: str = "jet",
        low: float = 0.05,
        high: float = 0.75,
        n: int = 256,
    ) -> Any:
        """
        Run the clip crange routine.

        Parameters
        ----------
        cmap_name : str
            Name of the Matplotlib colormap to transform.
        low : float
            Lower normalized colormap or threshold bound.
        high : float
            Upper normalized colormap or threshold bound.
        n : int
            Number of samples, bins, gates, or plotted items.

        Returns
        -------
        Any
            Object produced by clip crange.
        """
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"Require 0 ≤ low < high ≤ 1, got low={low}, high={high}")
        base = plt.get_cmap(cmap_name)
        colors = base(np.linspace(low, high, n))
        return ListedColormap(
            colors, name=f"{cmap_name}_{int(low * 100)}_{int(high * 100)}"
        )

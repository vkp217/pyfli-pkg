"""
Build Matplotlib colormap variants for FLIM map visualization.

This module belongs to :mod:`pyfli.data_vnp` and is part of PyFLI visualization,
normalization, plotting, and mono-versus-bi-exponential comparison tools. Public API
includes classes :class:`ColorProcessor`.
"""

from __future__ import annotations
from typing import Any
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


class ColorProcessor:
    """
    Create Matplotlib colormaps tailored for FLIM maps. It can force the lowest value to
    black or clip a colormap range to improve visual contrast.
    """

    def lowest_zero(self, cmap_name: str = "jet") -> Any:
        """
        Handle lowest zero.

        Parameters
        ----------
        cmap_name : str
            Input value.

        Returns
        -------
        Any
            Return value.
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
        Handle clip crange.

        Parameters
        ----------
        cmap_name : str
            Input value.
        low : float
            Input value.
        high : float
            Input value.
        n : int
            Input value.

        Returns
        -------
        Any
            Return value.
        """
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"Require 0 ≤ low < high ≤ 1, got low={low}, high={high}")
        base = plt.get_cmap(cmap_name)
        colors = base(np.linspace(low, high, n))
        return ListedColormap(
            colors, name=f"{cmap_name}_{int(low * 100)}_{int(high * 100)}"
        )

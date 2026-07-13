"""Small helpers for building and adjusting matplotlib colormaps used
throughout the FLI/FLIM plotting tools.
"""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class Colorprocess:
    """Utility class for deriving custom `ListedColormap` variants.

    Wraps a couple of stateless colormap transforms (zero-highlighting and
    range clipping) behind an instantiable class so callers can chain calls
    such as ``Colorprocess().lowest_zero('jet')``.
    """

    def __init__(self):
        """Create a Colorprocess instance. Holds no state."""
        pass

    def lowest_zero(self, cmap_name='jet'):
        """Build a colormap with its lowest entry forced to opaque black.

        Samples 256 colors from the named base colormap and overrides the
        first entry (value 0) with black, which is useful for marking
        background / zero-intensity pixels distinctly in imshow plots.

        Args:
            cmap_name (str): Name of a matplotlib colormap to sample from.
                Defaults to 'jet'.

        Returns:
            matplotlib.colors.ListedColormap: A 256-color colormap whose
            lowest entry is `[0, 0, 0, 1]` (opaque black).
        """
        original_cmap = plt.get_cmap(cmap_name)
        colors = original_cmap(np.linspace(0, 1, 256))
        colors[0] = [0, 0, 0, 1]
        return ListedColormap(colors)

    def clip_crange(self, cmap_name='jet', low=0.05, high=0.75, n=256):
        """Build a colormap restricted to a sub-range of a base colormap.

        Samples `n` colors evenly between `low` and `high` (in the base
        colormap's normalized 0-1 domain), which effectively clips out the
        extreme ends of the palette.

        Args:
            cmap_name (str): Name of a matplotlib colormap to sample from.
                Defaults to 'jet'.
            low (float): Lower bound of the sampling range (0-1). Defaults
                to 0.05.
            high (float): Upper bound of the sampling range (0-1). Defaults
                to 0.75.
            n (int): Number of colors to sample. Defaults to 256.

        Returns:
            matplotlib.colors.ListedColormap: A colormap named
            ``f"{cmap_name}_{int(low*100)}_{int(high*100)}"`` built from the
            clipped color range.

        Raises:
            ValueError: If the bounds do not satisfy
                ``0.0 <= low < high <= 1.0``.
        """
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"Require 0 ≤ low < high ≤ 1, got low={low}, high={high}")
        base = plt.get_cmap(cmap_name)
        colors = base(np.linspace(low, high, n))
        return ListedColormap(colors, name=f"{cmap_name}_{int(low*100)}_{int(high*100)}")
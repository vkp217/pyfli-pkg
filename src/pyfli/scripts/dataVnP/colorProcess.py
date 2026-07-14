import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class Colorprocess:
    def __init__(self):
        pass

    def lowest_zero(self, cmap_name='jet'):
        original_cmap = plt.get_cmap(cmap_name)
        colors = original_cmap(np.linspace(0, 1, 256))
        colors[0] = [0, 0, 0, 1]
        return ListedColormap(colors)

    def clip_crange(self, cmap_name='jet', low=0.05, high=0.75, n=256):
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(f"Require 0 ≤ low < high ≤ 1, got low={low}, high={high}")
        base = plt.get_cmap(cmap_name)
        colors = base(np.linspace(low, high, n))
        return ListedColormap(colors, name=f"{cmap_name}_{int(low*100)}_{int(high*100)}")
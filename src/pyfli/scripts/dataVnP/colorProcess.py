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
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import gridspec

class DataViewer:
    def __init__(self, save_path=None):
        self.save_path = save_path
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

    def _apply_marker(self, ax, coord):
        if coord is not None:
            x, y = coord
            ax.scatter(y, x, color='red', s=40, marker='x', edgecolors='white', linewidths=1.5)

    def display_data(self, data_list, structure=(1, 1), coord=None, data_names=None, 
                     cmaps=None, v_ranges=None, figsize=None, normalize=True, yscale = 'linear'):
        """
        Unified 2D/3D visualization with specific normalization and height-aligned subplots.
        """
        num_plots = len(data_list)
        r, c = structure
        names = data_names or [f"Data {i+1}" for i in range(num_plots)]
        
        # Identify first 3D data for decay-based normalization
        first_3d_idx = next((i for i, d in enumerate(data_list) if d.ndim == 3), None)
        show_decay = coord is not None and first_3d_idx is not None
        
        # Define GridSpec: add an extra column for the decay plot if necessary
        n_cols = c + (1 if show_decay else 0)
        fig = plt.figure(figsize=figsize or (n_cols * 5, r * 4))
        gs = gridspec.GridSpec(r, n_cols, figure=fig)
        
        # Calculate ref_max: Based on the decay peak of the first 3D dataset at (x,y)
        ref_max = None
        if normalize and show_decay:
            x, y = coord
            ref_max = np.max(data_list[first_3d_idx][x, y, :])

        # Plot Images
        for i in range(num_plots):
            row, col = divmod(i, c)
            ax = fig.add_subplot(gs[row, col])
            
            data = data_list[i]
            img = np.sum(data, axis=2) if data.ndim == 3 else data
            
            # Normalization Logic
            if ref_max is not None:
                img = img / (ref_max + 1e-9)
            elif normalize is True:
                img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img) + 1e-9)

            cmap = cmaps[i] if cmaps and i < len(cmaps) else "viridis"
            vr = v_ranges[i] if v_ranges and i < len(v_ranges) else (None, None)
            
            im = ax.imshow(img, cmap=cmap, vmin=vr[0], vmax=vr[1])
            self._apply_marker(ax, coord)
            ax.set_title(names[i])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot Decay (aligned height)
        if show_decay:
            x, y = coord
            ax_decay = fig.add_subplot(gs[:, -1]) 
            for i, data in enumerate(data_list):
                if data.ndim == 3:
                    ax_decay.plot(data[x, y, :], label=names[i], lw=1.5)
            
            ax_decay.set(yscale = yscale, title=f"time-series @ ({x},{y})", xlabel="Time Bin", ylabel="Counts")
            ax_decay.legend(fontsize='small', loc='upper right')
            ax_decay.grid(True, which='both', alpha=0.3)

        plt.tight_layout()
        
        if self.save_path:
            plt.savefig(os.path.join(self.save_path, "combined_viz.png"), dpi=300, bbox_inches='tight')
        plt.show()
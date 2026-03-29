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


    def plot_fli_fit_summary(self, data, pixel=None, title="FLI Fit Summary"):
        """
        Dual-mode FLI visualization with clean layout.

        Fixes:
        - Summary placed BETWEEN (1,2) and (1,3)
        - Proper log scaling (no zero/negative issues)
        """

        # -------- Detect mode --------
        is_3d = pixel is not None

        # -------- Extract --------
        decay = data['raw_data']['decay']
        irf = data['raw_data']['irf']
        fit = data['TR_maps']['fit_map']
        residuals = data['TR_maps']['residuals_map']
        maps = data['results']['maps']

        # -------- Handle 1D vs 3D --------
        if is_3d:
            x, y = pixel
            decay_1d = decay[x, y, :]
            irf_1d = irf[x, y, :]
            fit_1d = fit[x, y, :]
            residuals_1d = residuals[x, y, :]

            def get_param(k):
                val = maps.get(k, None)
                return val[x, y] if isinstance(val, np.ndarray) else val
        else:
            decay_1d = decay
            irf_1d = irf
            fit_1d = fit
            residuals_1d = residuals

            def get_param(k):
                return maps.get(k, None)

        # -------- Safe log handling --------
        eps = 1e0
        decay_log = np.clip(decay_1d, eps, None)
        fit_log = np.clip(fit_1d, eps, None)
        irf_scaled = (irf_1d / np.max(irf_1d)) * np.max(decay_1d)
        irf_log = np.clip(irf_scaled, eps, None)

        # -------- Format parameters --------
        def fmt(v):
            try:
                return f"{float(v):.4f}"
            except:
                return "NA"

        text_str = (
            f"mono: {fmt(get_param('mono'))}\n"
            f"E: {fmt(get_param('E'))}\n"
            f"f: {fmt(get_param('f'))}\n"
            f"tau1: {fmt(get_param('tau1'))}\n"
            f"tau2: {fmt(get_param('tau2'))}\n"
            f"A1: {fmt(get_param('A1'))}\n"
            f"A2: {fmt(get_param('A2'))}\n"
            f"tau_mean: {fmt(get_param('tau_mean'))}"
        )

        # -------- Layout (NOW 4 columns: plot, plot, text, image) --------
        fig = plt.figure(figsize=(20, 5))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.6, 1], wspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax_text = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[3])

        x_axis = np.arange(len(decay_1d))

    # -------- (1,1) LOG --------
        ax1.scatter(x_axis, decay_log, s=20, color='black', alpha=0.7, marker='*', label='Decay (scatter)')
        ax1.plot(decay_log, color='blue', alpha=0.6, lw=1.2, label='Decay (line)')
        ax1.plot(irf_log, linestyle='--', color='orange', lw=1.5, label='IRF')
        ax1.plot(fit_log, color='green', lw=1.5, label='Fit')

        ax1.set_yscale('log')
        ax1.set_ylim(eps, np.max(decay_log) * 1.2)

        ax1.set_title('Log Scale')
        ax1.set_xlabel('Time/Bins')
        ax1.set_ylabel('Intensity')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        # -------- (1,2) LINEAR --------
        ax2.scatter(x_axis, decay_1d, s=20, color='black', alpha=0.7, marker='*', label='Decay (scatter)')
        ax2.plot(decay_1d, color='blue', alpha=0.6, lw=1.2, label='Decay (line)')
        ax2.plot(irf_scaled, linestyle='--', color='orange', lw=1.5, label='IRF')
        ax2.plot(fit_1d, color='green', lw=1.5, label='Fit')
        ax2.plot(residuals_1d, color='red', lw=1.5, label='Residuals')

        ax2.set_title('Linear Scale')
        ax2.set_xlabel('Time/Bins')
        ax2.set_ylabel('Intensity')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # -------- (TEXT PANEL - BETWEEN 1,2 and 1,3) --------
        ax_text.axis('off')
        ax_text.text(
            0.0, 1.0, text_str,
            fontsize=10,
            va='top',
            family='monospace'
        )
        ax_text.set_title('Fit Summary')

        # -------- (1,3) IMAGE --------
        if is_3d:
            img = np.sum(decay, axis=2)
            im = ax3.imshow(img)

            ax3.plot(y, x, 'rx', markersize=8, mew=2)
            ax3.set_title(f"Intensity Map\nPixel ({x},{y})")

            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.02)
        else:
            ax3.axis('off')

        # -------- Final --------
        fig.suptitle(title, fontsize=14)
        plt.show()
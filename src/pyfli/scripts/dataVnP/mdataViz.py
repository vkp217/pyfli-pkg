"""Visualization helpers for FLI/FLIM data: multi-panel spatial map
displays, per-pixel decay/fit summaries, and combined image/time-series
plots, with optional figure saving to disk.
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import gridspec

class DataViewer:
    """Renders grids of spatial maps and per-pixel decay/fit plots for
    FLI/FLIM datasets, with optional saving of the resulting figures.

    An instance can be configured with a save directory and figure name so
    that every plotting call also persists a PNG alongside displaying it.
    """

    def __init__(self, save_path=None, fig_name = None):
        """Create a DataViewer, optionally configured to save figures.

        Args:
            save_path (str, optional): Directory to save figures into. If
                given and it does not already exist, it is created.
            fig_name (str, optional): Base filename (without extension)
                used when saving figures. If not given, a method-specific
                default name is used instead.
        """
        self.fig_name = fig_name
        self.save_path = save_path
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

    def _apply_marker(self, ax, coord):
        if coord is not None:
            x, y = coord
            ax.scatter(y, x, color='red', s=40, marker='x', edgecolors='white', linewidths=1.5)

    def display_data(self, data_list, structure=(1, 1), coord=None, data_names=None,
                    cmaps=None, v_ranges=None, figsize=None, normalize=False, yscale='linear'):
        """Display a grid of spatial maps, with an optional decay-plot panel.

        Each entry in `data_list` is shown as an imshow panel (2-D arrays
        shown directly; 3-D arrays are summed along the last axis first).
        If `coord` is given and at least one array in `data_list` is 3-D, an
        extra column is added showing the time-series ("decay") at that
        pixel for every 3-D array, optionally normalized against the first
        3-D array's peak at that pixel. If the instance was constructed with
        a `save_path`, the resulting figure is also saved as a PNG there.

        Args:
            data_list (list[np.ndarray]): Arrays to display, each either 2-D
                (H, W) or 3-D (H, W, T).
            structure (tuple[int, int]): (rows, cols) grid layout for the
                image panels. Defaults to (1, 1).
            coord (tuple[int, int], optional): (x, y) pixel location used to
                mark each image panel and, if present, drive the extra
                decay/time-series panel.
            data_names (list[str], optional): Titles for each panel.
                Defaults to "Data 1", "Data 2", ...
            cmaps (list, optional): Per-panel colormap, one per entry in
                `data_list`. Falls back to "viridis" for missing entries.
            v_ranges (list[tuple], optional): Per-panel (vmin, vmax) for
                imshow. Falls back to (None, None) for missing entries.
            figsize (tuple, optional): Figure size. Defaults to a size
                scaled by the number of columns/rows.
            normalize (bool): If True, normalize the image panels — either
                against the reference pixel's peak (when `coord` and a 3-D
                array are present) or per-panel min/max otherwise. Defaults
                to False.
            yscale (str): Y-axis scale ('linear', 'log', ...) for the decay
                panel. Defaults to 'linear'.

        Returns:
            tuple: `(fig, img_axes, ax_decay)` — the created Figure, the
            list of image-panel Axes, and the decay-panel Axes (or None if
            no decay panel was drawn).
        """
        num_plots = len(data_list)
        r, c = structure
        names = data_names or [f"Data {i+1}" for i in range(num_plots)]        
        first_3d_idx = next((i for i, d in enumerate(data_list) if d.ndim == 3), None)
        show_decay = coord is not None and first_3d_idx is not None        
        n_cols = c + (1 if show_decay else 0)
        
        # FIX: Added layout="constrained" to figure initialization
        fig = plt.figure(figsize=figsize or (n_cols * 5, r * 4), layout="constrained")

        # Main grid
        gs = gridspec.GridSpec(r, n_cols, figure=fig, wspace=0.25, hspace=0.25)
        # Reference normalization
        ref_max = None
        if normalize and show_decay:
            x, y = coord
            ref_max = np.max(data_list[first_3d_idx][x, y, :])
        # IMAGE PANELS
        img_axes = []
        for i in range(num_plots):
            row, col = divmod(i, c)
            # Subgrid: [main axis | colorbar axis]
            subgs = gs[row, col].subgridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
            ax = fig.add_subplot(subgs[0])
            cax = fig.add_subplot(subgs[1])
            img_axes.append(ax)
            data = data_list[i]
            img = np.sum(data, axis=2) if data.ndim == 3 else data
            if ref_max is not None:
                img = img / (ref_max + 1e-9)
            elif normalize is True:
                img = (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img) + 1e-9)
            cmap = cmaps[i] if cmaps and i < len(cmaps) else "viridis"
            vr = v_ranges[i] if v_ranges and i < len(v_ranges) else (None, None)
            im = ax.imshow(img, cmap=cmap, vmin=vr[0], vmax=vr[1])
            self._apply_marker(ax, coord)
            ax.set_title(names[i])
            plt.colorbar(im, cax=cax)

        # Plot Panel (forcing to same size)
        ax_decay = None
        if show_decay:
            x, y = coord
            subgs = gs[:, -1].subgridspec(1, 2, width_ratios=[20, 1], wspace=0.05)
            ax_decay = fig.add_subplot(subgs[0])
            cax_dummy = fig.add_subplot(subgs[1])  # placeholder
            for i, data in enumerate(data_list):
                if data.ndim == 3:
                    ax_decay.plot(data[x, y, :], label=names[i], lw=1.5)
            ax_decay.set(
                yscale=yscale,
                title=f"Plots @ ({x},{y})",
                # xlabel="Time Bin",
                # ylabel="Counts"
            )
            ax_decay.legend(fontsize='small', loc='upper right')
            ax_decay.grid(True, which='both', alpha=0.3)
            cax_dummy.axis('off')

        if self.save_path:
            if self.fig_name:
                plt.savefig(os.path.join(self.save_path, self.fig_name +".png"),
                            dpi=300, bbox_inches='tight')
            else:
                plt.savefig(os.path.join(self.save_path, "combined_display.png"),
                            dpi=300, bbox_inches='tight')
        plt.show()
        return fig, img_axes, ax_decay


    def plot_pyfli_fit_summary(self, 
                                data,
                                pixel=None,
                                title="FLI Fit Summary",
                                mode=("decay", "irf", "fit", "residuals"),
                                esp = 1e0):
        """Plot a decay/IRF/fit/residuals summary panel for a pyfli fit result.

        Expects `data` in the specific dict structure produced by pyfli's
        image-processing / simulator pipeline (`data['raw_data']['decay']`,
        `data['raw_data']['irf']`, `data['TR_maps']['fit_map']`,
        `data['TR_maps']['residuals_map']`, `data['results']['maps']`).
        Renders log-scale and linear-scale traces of the selected `mode`
        curves side by side, a text panel listing the fitted parameter maps
        (evaluated at `pixel` if given), and — when `pixel` is given — an
        intensity-map panel marking that pixel.

        Args:
            data (dict): pyfli fit-result dictionary as described above.
            pixel (tuple[int, int], optional): (x, y) pixel to index into
                3-D data. If None, `data` is treated as already 1-D
                (single-pixel) and no image panel is drawn.
            title (str): Figure suptitle. Defaults to "FLI Fit Summary".
            mode (Iterable[str]): Subset of {"decay", "irf", "fit",
                "residuals"} selecting which curves to plot. Defaults to
                all four.
            esp (float): Clipping floor used before taking the log scale
                (avoids log(0)/log(negative)). Defaults to 1e0.

        Returns:
            tuple: `(fig, (ax1, ax2, ax_text, ax3))` — the Figure and its
            log-scale, linear-scale, text, and (if `pixel` given) image
            Axes; `ax3` is None when `pixel` is None.
        """
        # since simulator/pyfli img processing output is in specific disctionary
        # best to use in the simulator
        mode = set(mode)  # faster lookup
        is_3d = pixel is not None
        decay = data['raw_data']['decay']
        irf = data['raw_data']['irf']
        fit = data['TR_maps']['fit_map']
        residuals = data['TR_maps']['residuals_map']
        maps = data['results']['maps']
        if is_3d:
            x, y = pixel
            decay_1d = decay[x, y, :]
            irf_1d = irf[x, y, :]
            fit_1d = fit[x, y, :]
            residuals_1d = residuals[x, y, :]
        else:
            decay_1d = decay
            irf_1d = irf
            fit_1d = fit
            residuals_1d = residuals
        eps = esp
        decay_log = np.clip(decay_1d, eps, None)
        fit_log = np.clip(fit_1d, eps, None)
        irf_scaled = (irf_1d / np.max(irf_1d)) * np.max(decay_1d)
        irf_log = np.clip(irf_scaled, eps, None)
        def fmt(v):
            try:
                return f"{float(v):.4f}"
            except:
                return "NA"
        lines = []
        for k, v in maps.items():
            try:
                if is_3d and isinstance(v, np.ndarray):
                    val = v[x, y]
                else:
                    val = v
            except:
                val = v
            lines.append(f"{k}: {fmt(val)}")
        text_str = "\n".join(lines)
        if is_3d:
            fig = plt.figure(figsize=(20, 5))
            gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 0.6, 1], wspace=0.3)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax_text = fig.add_subplot(gs[2])
            ax3 = fig.add_subplot(gs[3])
        else:
            fig = plt.figure(figsize=(14, 5))
            gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.8], wspace=0.3)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[1])
            ax_text = fig.add_subplot(gs[2])
            ax3 = None
        x_axis = np.arange(len(decay_1d))
        #  (1,2) Log --------
        if "decay" in mode:
            ax1.scatter(x_axis, decay_log, s=20, color='black', marker='*', alpha=0.7, label='Decay (scatter)')
            ax1.plot(x_axis, decay_log, color='blue', lw=1.2, alpha=0.6, label='Decay (line)')
        if "irf" in mode:
            ax1.plot(x_axis, irf_log, linestyle='--', color='orange', lw=1.5, label='IRF')
        if "fit" in mode:
            ax1.plot(x_axis, fit_log, color='green', lw=1.5, label='Fit')
        if "residuals" in mode:
            ax1.plot(x_axis, np.clip(residuals_1d, eps, None), color='red', lw=1.2, label='Residuals')
        ax1.set_yscale('log')
        ax1.set_ylim(eps, np.max(decay_log) * 1.2)
        ax1.set_title('Log Scale')
        ax1.set_xlabel('Time/Bins')
        ax1.set_ylabel('Intensity')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        #  (1,2) LINEAR --------
        if "decay" in mode:
            ax2.scatter(x_axis, decay_1d, s=20, color='black', marker='*', alpha=0.7, label='Decay (scatter)')
            ax2.plot(x_axis, decay_1d, color='blue', lw=1.2, alpha=0.6, label='Decay (line)')
        if "irf" in mode:
            ax2.plot(x_axis, irf_scaled, linestyle='--', color='orange', lw=1.5, label='IRF')
        if "fit" in mode:
            ax2.plot(x_axis, fit_1d, color='green', lw=1.5, label='Fit')
        if "residuals" in mode:
            ax2.plot(x_axis, residuals_1d, color='red', lw=1.2, label='Residuals')
        ax2.set_title('Linear Scale')
        ax2.set_xlabel('Time/Bins')
        ax2.set_ylabel('Intensity')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # -------- TEXT PANEL --------
        ax_text.axis('off')
        ax_text.text(
            0.0, 1.0, text_str,
            fontsize=12,
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

        fig.suptitle(title, fontsize=14)
        plt.show()
        return fig, (ax1, ax2, ax_text, ax3)

    def plot_fli_px(self,
                    data_list,
                    pixel=None, # pixel: (x, y) → enables time-series plotting
                    title="FLI Data Viewer",
                    mode=None, # index in data_list which are to be added in the plot
                    mode2=None, # index in the data_list which has to be displayed
                    names=None,
                    esp = 1e0):
        """Plot selected datasets as images and, optionally, pixel time-series.

        `data_list` entries indexed by `mode` are plotted as log/linear
        time-series at `pixel` (3-D entries are indexed at `pixel`; other
        entries are plotted as-is). `data_list` entries indexed by `mode2`
        are shown as image panels (3-D entries summed along the last axis;
        other entries drawn with `ax.plot`), each optionally marked at
        `pixel`. If the instance was constructed with a `save_path`, the
        resulting figure is also saved as a PNG there.

        Args:
            data_list (list[np.ndarray]): Candidate datasets, each 2-D or
                3-D.
            pixel (tuple[int, int], optional): (x, y) pixel enabling the
                log/linear time-series panels. If None, only image panels
                are drawn.
            title (str): Figure suptitle. Defaults to "FLI Data Viewer".
            mode (list[int], optional): Indices into `data_list` to include
                in the time-series panels. Defaults to all indices.
            mode2 (list[int], optional): Indices into `data_list` to display
                as image panels. Defaults to `mode`.
            names (list[str], optional): Labels for the plotted/imaged
                datasets, shared between the time-series and image panels.
                Defaults to "Data {index}".
            esp (float): Clipping floor used before taking the log scale in
                the time-series panel. Defaults to 1e0.

        Returns:
            tuple: If `pixel` is given, `(fig, (ax_log, ax_lin, img_axes))`;
            otherwise `(fig, img_axes)`.
        """
        n_total = len(data_list)
        if mode is None:
            mode = list(range(n_total))
        if mode2 is None:
            mode2 = mode

        selected_plot = [data_list[i] for i in mode]
        selected_img = [data_list[i] for i in mode2]

        labels_plot = names if names else [f"Data {i}" for i in mode]
        labels_img = names if names else [f"Data {i}" for i in mode2]

        is_pixel = pixel is not None
        n_imgs = len(selected_img)
        if is_pixel:
            fig = plt.figure(figsize=(5 * (n_imgs + 2), 4), layout='constrained')
            gs = gridspec.GridSpec(1, n_imgs + 2, figure=fig)
            ax_log = fig.add_subplot(gs[0])
            ax_lin = fig.add_subplot(gs[1])
            img_axes = [fig.add_subplot(gs[i+2]) for i in range(n_imgs)]
        else:
            fig = plt.figure(figsize=(5 * n_imgs, 4), layout='constrained')
            gs = gridspec.GridSpec(1, n_imgs, figure=fig)
            img_axes = [fig.add_subplot(gs[i]) for i in range(n_imgs)]

        eps = esp
        if is_pixel:
            x, y = pixel
            for i, data in enumerate(selected_plot):
                label = labels_plot[i]
                if data.ndim == 3:
                    signal = data[x, y, :]
                else:
                    signal = data
                t = np.arange(len(signal))
                ax_log.plot(t, np.clip(signal, eps, None), lw=1.5, label=label)
                ax_lin.plot(t, signal, lw=1.5, label=label)

            ax_log.set_yscale('log')
            ax_log.set_title("Log Scale")
            ax_log.grid(True, alpha=0.3)
            ax_log.legend(fontsize=8)

            ax_lin.set_title("Linear Scale")
            ax_lin.grid(True, alpha=0.3)
            ax_lin.legend(fontsize=8)

        if is_pixel:
            x, y = pixel
        for i, data in enumerate(selected_img):
            label = labels_img[i]
            if data.ndim == 3:
                img = np.sum(data, axis=2)
                im = img_axes[i].imshow(img)
                if is_pixel:
                    img_axes[i].plot(y, x, 'rx', markersize=8, mew=2)
                plt.colorbar(im, ax=img_axes[i], fraction=0.046, pad=0.02)
            else:
                img_axes[i].plot(data)
            img_axes[i].set_title(label)
        fig.suptitle(title, fontsize=14)

        if self.save_path:
            fname = (self.fig_name if self.fig_name else "fli_px_display") + ".png"
            plt.savefig(os.path.join(self.save_path, fname),
                        dpi=300, bbox_inches='tight')

        plt.show()
        if is_pixel:
            return fig, (ax_log, ax_lin, img_axes)
        return fig, img_axes
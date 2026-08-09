"""
Render FLIM maps, fit summaries, and selected-pixel traces.

This module belongs to :mod:`pyfli.data_vnp` and is part of PyFLI visualization,
normalization, plotting, and mono-versus-bi-exponential comparison tools. Public API
includes classes :class:`DataViewer`.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib import gridspec


class DataViewer:
    """
    Render fitted parameter maps, raw decay traces, IRF traces, and selected-pixel
    summaries. The class is aimed at interactive inspection and publication-style
    diagnostic figures.

    Parameters
    ----------
    save_path : str | None
        Output path used when saving generated masks, figures, or data.
    fig_name : str | None
        Figure name or output stem used by plotting helpers.
    """

    def __init__(
        self, save_path: str | None = None, fig_name: str | None = None
    ) -> None:
        self.fig_name = fig_name
        self.save_path = save_path
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)

    def _apply_marker(self, ax: Any, coord: Any) -> None:
        """
        Apply marker.

        Parameters
        ----------
        ax : Any
            Matplotlib axes object on which the plot is drawn.
        coord : Any
            Pixel coordinate highlighted or summarized by the plot.

        Returns
        -------
        None
            No object is returned; the function apply marker.
        """
        if coord is not None:
            x, y = coord
            ax.scatter(
                y, x, color="red", s=40, marker="x", edgecolors="white", linewidths=1.5
            )

    def display_data(
        self,
        data_list: np.ndarray,
        structure: tuple[int, ...] = (1, 1),
        coord: Any | None = None,
        data_names: np.ndarray | None = None,
        cmaps: np.ndarray | None = None,
        v_ranges: np.ndarray | None = None,
        figsize: np.ndarray | None = None,
        normalize: bool = False,
        yscale: str = "linear",
    ) -> tuple[Any, ...]:
        """
        Display data.

        Parameters
        ----------
        data_list : np.ndarray
            List of data arrays displayed by the viewer.
        structure : tuple[int, ...]
            Layout structure that controls how data arrays are displayed.
        coord : Any | None
            Pixel coordinate highlighted or summarized by the plot.
        data_names : np.ndarray | None
            Labels assigned to displayed data arrays.
        cmaps : np.ndarray | None
            Matplotlib colormap used by the visualization.
        v_ranges : np.ndarray | None
            Per-map display ranges used by the viewer.
        figsize : np.ndarray | None
            Figure size passed to Matplotlib.
        normalize : bool
            Whether data are normalized before comparison or plotting.
        yscale : str
            Scale used for the y-axis.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing displayed data handles and summary values.
        """
        num_plots = len(data_list)
        r, c = structure
        names = data_names or [f"Data {i + 1}" for i in range(num_plots)]
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
            ax_decay.legend(fontsize="small", loc="upper right")
            ax_decay.grid(True, which="both", alpha=0.3)
            cax_dummy.axis("off")

        if self.save_path:
            if self.fig_name:
                plt.savefig(
                    os.path.join(self.save_path, self.fig_name + ".png"),
                    dpi=300,
                    bbox_inches="tight",
                )
            else:
                plt.savefig(
                    os.path.join(self.save_path, "combined_display.png"),
                    dpi=300,
                    bbox_inches="tight",
                )
        plt.show()
        return fig, img_axes, ax_decay

    def plot_pyfli_fit_summary(
        self,
        data: np.ndarray,
        pixel: np.ndarray | None = None,
        title: str = "FLI Fit Summary",
        mode: tuple[str, ...] = ("decay", "irf", "fit", "residuals"),
        esp: float = 1e0,
    ) -> Any:
        # since simulator/pyfli img processing output is in specific disctionary
        # best to use in the simulator
        """
        Plot pyfli fit summary.

        Parameters
        ----------
        data : np.ndarray
            Data array or mapping processed by the routine.
        pixel : np.ndarray | None
            Selected pixel coordinate.
        title : str
            Title displayed on the generated plot.
        mode : tuple[str, ...]
            Mode selector used by the fitting, loading, or plotting routine.
        esp : float
            Small epsilon used to stabilize divisions and comparisons.

        Returns
        -------
        Any
            Object produced by plot pyfli fit summary.
        """
        mode = set(mode)  # faster lookup
        is_3d = pixel is not None
        decay = data["raw_data"]["decay"]
        irf = data["raw_data"]["irf"]
        fit = data["results"]["TR_maps"]["fit_map"]
        residuals = data["results"]["TR_maps"]["residual_map"]
        maps = data["results"]["maps"]
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

        def fmt(v: np.ndarray) -> Any:
            """
            Run the fmt routine.

            Parameters
            ----------
            v : np.ndarray
                Vector or matrix evaluated by the simplex projection.

            Returns
            -------
            Any
                Object produced by fmt.
            """
            try:
                return f"{float(v):.4f}"
            except Exception:
                return "NA"

        lines = []
        for k, v in maps.items():
            try:
                if is_3d and isinstance(v, np.ndarray):
                    val = v[x, y]
                else:
                    val = v
            except Exception:
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
            ax1.scatter(
                x_axis,
                decay_log,
                s=20,
                color="black",
                marker="*",
                alpha=0.7,
                label="Decay (scatter)",
            )
            ax1.plot(
                x_axis, decay_log, color="blue", lw=1.2, alpha=0.6, label="Decay (line)"
            )
        if "irf" in mode:
            ax1.plot(
                x_axis, irf_log, linestyle="--", color="orange", lw=1.5, label="IRF"
            )
        if "fit" in mode:
            ax1.plot(x_axis, fit_log, color="green", lw=1.5, label="Fit")
        if "residuals" in mode:
            ax1.plot(
                x_axis,
                np.clip(residuals_1d, eps, None),
                color="red",
                lw=1.2,
                label="Residuals",
            )
        ax1.set_yscale("log")
        ax1.set_ylim(eps, np.max(decay_log) * 1.2)
        ax1.set_title("Log Scale")
        ax1.set_xlabel("Time/Bins")
        ax1.set_ylabel("Intensity")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        #  (1,2) LINEAR --------
        if "decay" in mode:
            ax2.scatter(
                x_axis,
                decay_1d,
                s=20,
                color="black",
                marker="*",
                alpha=0.7,
                label="Decay (scatter)",
            )
            ax2.plot(
                x_axis, decay_1d, color="blue", lw=1.2, alpha=0.6, label="Decay (line)"
            )
        if "irf" in mode:
            ax2.plot(
                x_axis, irf_scaled, linestyle="--", color="orange", lw=1.5, label="IRF"
            )
        if "fit" in mode:
            ax2.plot(x_axis, fit_1d, color="green", lw=1.5, label="Fit")
        if "residuals" in mode:
            ax2.plot(x_axis, residuals_1d, color="red", lw=1.2, label="Residuals")
        ax2.set_title("Linear Scale")
        ax2.set_xlabel("Time/Bins")
        ax2.set_ylabel("Intensity")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)

        # -------- TEXT PANEL --------
        ax_text.axis("off")
        ax_text.text(0.0, 1.0, text_str, fontsize=12, va="top", family="monospace")
        ax_text.set_title("Fit Summary")

        # -------- (1,3) IMAGE --------
        if is_3d:
            img = np.sum(decay, axis=2)
            im = ax3.imshow(img)
            ax3.plot(y, x, "rx", markersize=8, mew=2)
            ax3.set_title(f"Intensity Map\nPixel ({x},{y})")
            plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.02)

        fig.suptitle(title, fontsize=14)
        plt.show()
        return fig, (ax1, ax2, ax_text, ax3)

    # Fixed, CVD-validated categorical order — assigned by series identity
    # (never cycled/re-ranked), so "decay" is always the same hue across calls.
    _SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7")
    _INK = "#0b0b0b"
    _MUTED = "#898781"
    _AXIS = "#c3c2b7"
    _GRID = "#e1e0d9"

    def plot_fli_px(
        self,
        data_list: np.ndarray,
        pixel: np.ndarray | None = None,  # pixel: (x, y) → enables time-series plotting
        title: str = "FLI Data Viewer",
        mode: str | None = None,  # index in data_list which are to be added in the plot
        mode2: Any | None = None,  # index in the data_list which has to be displayed
        names: Any | None = None,
        esp: float = 1e0,
        cmap: str = "viridis",
    ) -> tuple[Any, ...]:
        """
        Plot FLI px.

        Parameters
        ----------
        data_list : np.ndarray
            List of data arrays displayed by the viewer.
        pixel : np.ndarray | None
            Selected pixel coordinate.
        title : str
            Title displayed on the generated plot.
        mode : str | None
            Mode selector used by the fitting, loading, or plotting routine.
        mode2 : Any | None
            Secondary display mode used by the pixel viewer.
        names : Any | None
            Dataset names used in summaries and plots.
        esp : float
            Small epsilon used to stabilize divisions and comparisons.
        cmap : str
            Colormap used for the image panels.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing the pixel plot handles and extracted pixel values.
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

        rc = {
            "axes.edgecolor": self._AXIS,
            "axes.labelcolor": self._INK,
            "axes.linewidth": 1.0,
            "axes.titlecolor": self._INK,
            "axes.titleweight": "normal",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.color": self._MUTED,
            "ytick.color": self._MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": self._GRID,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }

        with plt.rc_context(rc):
            if is_pixel:
                fig = plt.figure(figsize=(5 * (n_imgs + 2), 4), layout="constrained")
                gs = gridspec.GridSpec(1, n_imgs + 2, figure=fig)
                ax_log = fig.add_subplot(gs[0])
                ax_lin = fig.add_subplot(gs[1])
                img_axes = [fig.add_subplot(gs[i + 2]) for i in range(n_imgs)]
            else:
                fig = plt.figure(figsize=(5 * n_imgs, 4), layout="constrained")
                gs = gridspec.GridSpec(1, n_imgs, figure=fig)
                img_axes = [fig.add_subplot(gs[i]) for i in range(n_imgs)]

            eps = esp
            if is_pixel:
                x, y = pixel
                for i, data in enumerate(selected_plot):
                    label = labels_plot[i]
                    color = self._SERIES_COLORS[i % len(self._SERIES_COLORS)]
                    if data.ndim == 3:
                        signal = data[x, y, :]
                    else:
                        signal = data
                    t = np.arange(len(signal))
                    ax_log.plot(t, np.clip(signal, eps, None), lw=2, color=color)
                    ax_lin.plot(t, signal, lw=2, color=color, label=label)

                for ax, scale_title in (
                    (ax_log, "Log Scale"),
                    (ax_lin, "Linear Scale"),
                ):
                    ax.set_title(scale_title)
                    ax.set_xlabel("Time Bin")
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(True, alpha=0.6, lw=0.6)
                    ax.set_axisbelow(True)

                ax_log.set_yscale("log")
                ax_log.set_ylabel("Counts (log)")
                ax_lin.set_ylabel("Counts")
                if len(selected_plot) > 1:
                    ax_lin.legend(loc="best")

            for i, data in enumerate(selected_img):
                label = labels_img[i]
                ax = img_axes[i]
                if data.ndim == 3:
                    img = np.sum(data, axis=2)
                    im = ax.imshow(img, cmap=cmap)
                    if is_pixel:
                        ax.plot(
                            y,
                            x,
                            marker="x",
                            markersize=10,
                            markeredgewidth=2.5,
                            color="white",
                            zorder=5,
                        )
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                    cbar.set_label("Total Counts", size=9)
                    cbar.ax.tick_params(labelsize=8)
                    ax.set_xlabel("X (px)")
                    ax.set_ylabel("Y (px)")
                else:
                    ax.plot(data, lw=2, color=self._SERIES_COLORS[0])
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.grid(True, alpha=0.6, lw=0.6)
                    ax.set_axisbelow(True)
                ax.set_title(label)

            fig.suptitle(title, fontsize=14, fontweight="bold", color=self._INK)

            if self.save_path:
                fname = (self.fig_name if self.fig_name else "fli_px_display") + ".png"
                plt.savefig(
                    os.path.join(self.save_path, fname), dpi=300, bbox_inches="tight"
                )

            plt.show()

        if is_pixel:
            return fig, (ax_log, ax_lin, img_axes)
        return fig, img_axes

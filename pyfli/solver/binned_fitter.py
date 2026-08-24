"""
Bin image cubes spatially and fit the binned FLI data.

This module belongs to :mod:`pyfli.solver` and is part of PyFLI least-squares, maximum-
likelihood, CPU, GPU, binned, and global FLI fitting routines. Public API includes
classes :class:`FLIBinner` and :class:`BinnedFLIFitter`.
"""

from typing import Any

# solver/binned_fitter.py
import numpy as np

from pyfli import logging


class FLIBinner:
    """
    Apply spatial binning to FLI image and IRF cubes. It reduces noise by aggregating
    neighboring pixels before fitting.

    Parameters
    ----------
    bin_radius : int
        Radius of the spatial binning neighborhood in pixels.
    """

    def __init__(self, bin_radius: int = 1) -> None:
        self.bin_radius = bin_radius
        self.binned_img = None
        self.binned_irf = None

    def apply_binning(
        self, image_cube: np.ndarray, irf_cube: np.ndarray
    ) -> tuple[Any, ...]:
        """
        Performs spatial binning using constant padding to maintain
        original image dimensions.
        """
        H, W, T = image_cube.shape
        n = self.bin_radius
        window_size = 2 * n + 1

        # 1. Pad spatially (H, W) but not temporally (T)
        pad_width = ((n, n), (n, n), (0, 0))
        img_pad = np.pad(image_cube, pad_width, mode="constant", constant_values=0)
        irf_pad = np.pad(irf_cube, pad_width, mode="constant", constant_values=0)

        # 2. Initialize output arrays with same size as original
        self.binned_img = np.zeros_like(image_cube, dtype=np.float32)
        self.binned_irf = np.zeros_like(irf_cube, dtype=np.float32)

        logging.info(
            f"Applying spatial binning: Radius={n} ({window_size}x{window_size} window)"
        )

        # 3. Fast vectorised summation using window offsets.
        # dr shifts along rows (axis 0), dc along columns (axis 1).
        for dr in range(window_size):
            for dc in range(window_size):
                self.binned_img += img_pad[dr : dr + H, dc : dc + W, :]
                self.binned_irf += irf_pad[dr : dr + H, dc : dc + W, :]

        return self.binned_img, self.binned_irf

    def get_binned_data(self) -> tuple[Any, ...]:
        """Returns the binned cubes for manual inspection."""
        return self.binned_img, self.binned_irf


class BinnedFLIFitter:
    """
    Fit spatially binned FLI data with an existing processor. It wraps binning, mask
    propagation, processor dispatch, and result saving for binned datasets.

    Parameters
    ----------
    processor_instance : Any
        Optional processor reused for pixel-level fitting or reconstruction.
    bin_radius : int
        Radius of the spatial binning neighborhood in pixels.
    """

    def __init__(self, processor_instance: Any, bin_radius: int = 1) -> None:
        self.processor = processor_instance
        self.bin_radius = bin_radius
        self.freq = processor_instance.freq

    def fit(
        self,
        b_img: np.ndarray,
        b_irf: np.ndarray,
        mask: np.ndarray | None = None,
        data_name: str = "Binned_Dataset",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Unified entry point using Duck-Typing.
        Accepts PRE-BINNED data cubes.
        """
        # 1. Setup variables
        dataset = None
        proc = self.processor

        # Safely extract estimator, defaulting to 'least_squares' if not provided
        estimator = kwargs.pop("estimator", "least_squares")

        # 2. Dynamic Engine Dispatch
        if hasattr(proc, "process_image"):
            logging.info(f"Engine: CPU Parallel Processor (via {type(proc).__name__})")
            kwargs["estimator"] = estimator.lower()
            dataset = proc.process_image(
                image_cube=b_img,
                irf_cube=b_irf,
                mask=mask,
                data_name=data_name,
                **kwargs,
            )

        elif hasattr(proc, "fit_image"):
            logging.info(
                f"Engine: GPU Vectorized Processor (via {type(proc).__name__})"
            )
            kwargs["mode"] = estimator.upper()
            kwargs.pop("n_jobs", None)  # Clean up CPU-specific args
            dataset = proc.fit_image(
                image_cube=b_img,
                irf_cube=b_irf,
                mask=mask,
                data_name=data_name,
                **kwargs,
            )

        else:
            raise TypeError(
                "The provided processor_instance is not a recognized CPU or GPU FLI Processor."
            )

        # 3. Metadata Injection
        if dataset and "results" in dataset:
            dataset["name"] = f"{data_name}_Binned_R{self.bin_radius}"
            dataset["bin_radius"] = (
                self.bin_radius
            )  # top-level; NOT inside maps (maps holds 2D arrays only)

        return dataset

    def save_results(self, dataset: np.ndarray, folder: str = "results") -> None:
        """Pass-through to the underlying processor's optimized save logic."""
        if dataset is None:
            logging.warning("No dataset provided to save.")
            return
        self.processor.save_results(dataset, folder)

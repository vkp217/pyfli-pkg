# pyfli/simulator/image_generator_common.py

"""
Shared per-ROI mask loading, simulator selection, and pixel-loop logic for the
full-image FLI dataset generators.

``combined/sim_image_generator.py`` (:class:`~pyfli.simulator.combined.sim_image_generator.FLIImageGenerator`,
wrapping :class:`~pyfli.simulator.combined.main_factory.MacroSimulator`/
:class:`~pyfli.simulator.combined.main_factory.TCSPCSimulator`) and
``separate/sim_model_image_generator.py`` (:class:`~pyfli.simulator.separate.sim_model_image_generator.FLIModelImageGenerator`,
wrapping :class:`~pyfli.simulator.separate.main_factory_gen.ContinuousSimulator`/
:class:`~pyfli.simulator.separate.main_factory_gen.PhotonCountSimulator`) implement
identical mask-loading, per-ROI simulator dispatch, and pixel-loop logic; they differ
only in which pair of simulator classes they dispatch between, and in whether
parameter maps are recorded for the background ROI (0). Subclasses declare those
differences as class attributes; this module owns the actual logic so it is defined
exactly once.
"""

import itertools
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from pyfli import logging

from .sim_helper import irf_picker


class BaseFLIImageGenerator:
    """
    Shared ``__init__``/``generate_image`` logic for FLIImageGenerator/
    FLIModelImageGenerator.

    Subclasses set the following class attributes:

    continuous_cls : type
        Simulator class used for ROIs whose effective ``sensor_type`` is
        ``"continuous"`` (``MacroSimulator`` / ``ContinuousSimulator``).
    discrete_cls : type
        Simulator class used for ROIs whose effective ``sensor_type`` is
        anything else (``TCSPCSimulator`` / ``PhotonCountSimulator``).
    include_background_roi_in_maps : bool
        Whether parameter maps are recorded for pixels in the background ROI
        (ROI value 0), in addition to any labeled ROI. Ignored (treated as
        ``True``) whenever no ``roi_mask_path`` was supplied, since in that
        case ROI 0 is not "background" — it's the only region there is.
    """

    continuous_cls: type
    discrete_cls: type
    include_background_roi_in_maps: bool = True

    def __init__(
        self,
        irf_data: np.ndarray,
        intensity_image_path: str | None = None,
        roi_mask_path: str | None = None,
        roi_params: Any | None = None,
        image_shape: tuple[int, ...] = (32, 32),
        method: str = "continuous",
        verbose: bool = True,
        bool_mask: np.ndarray | None = None,
    ) -> None:
        self.method = method.lower()
        self.irf_data = irf_data
        self.verbose = verbose
        self.bool_mask = (
            np.asarray(bool_mask, dtype=bool) if bool_mask is not None else None
        )
        # Without an explicit roi_mask_path every pixel is ROI 0 by
        # construction (see below) — that's "the only region", not
        # "background to exclude", so the exclusion policy only applies
        # when the caller actually supplied a multi-region ROI mask.
        self._record_background_roi = (
            self.include_background_roi_in_maps or roi_mask_path is None
        )

        # Loading the intensity Mask
        if intensity_image_path:
            img = Image.open(intensity_image_path)
            if img.mode in ("P", "PA"):
                img = img.convert("RGBA")
            arr = np.array(img)
            # Binary foreground/background mask: any pixel with a nonzero
            # value (any nonzero channel for color input, any nonzero value
            # for grayscale input at any bit depth) is foreground; pure
            # zero/black is background. An already-binary source image
            # passes through unchanged, since re-binarizing it is a no-op.
            nonzero = arr.any(axis=-1) if arr.ndim == 3 else (arr != 0)
            self.intensity_mask = nonzero.astype(float)
            self.shape = self.intensity_mask.shape
        else:
            self.intensity_mask = np.ones(image_shape)
            self.shape = image_shape

        # loading the ROI Mask (multi-cluster mask)
        if roi_mask_path:
            mask_img = Image.open(roi_mask_path).convert("L")
            self.roi_mask = np.array(
                mask_img.resize((self.shape[1], self.shape[0]), Image.NEAREST)
            ).astype(int)
        else:
            self.roi_mask = np.zeros(self.shape, dtype=int)

        # Initialize ROI Simulators
        dummy_irf = irf_picker(irf_data)
        # dummy_irf = irf_data[0, 0, :] if irf_data.ndim == 3 else irf_data
        self.roi_sims = {}
        unique_rois = np.unique(self.roi_mask)
        default_sensor_type = (
            "continuous" if self.method == "continuous" else "discrete"
        )

        for idx, roi_val in enumerate(unique_rois):
            cfg = (
                roi_params[idx].copy() if (roi_params and idx < len(roi_params)) else {}
            )
            sensor_type = cfg.pop("sensor_type", default_sensor_type)
            cfg.pop("method", None)
            SimClass = (
                self.continuous_cls
                if sensor_type.lower() == "continuous"
                else self.discrete_cls
            )
            self.roi_sims[roi_val] = SimClass(dummy_irf, sensor_type=sensor_type, **cfg)

    def generate_image(self) -> dict[Any, Any]:
        """Simulates every pixel and assembles the full FLI dataset.

        Iterates over all ``(i, j)`` pixels, selects the simulator assigned
        to that pixel's ROI (swapping in a per-pixel normalized IRF slice
        when ``irf_data`` is 3-D), runs it, and accumulates the results
        (scaled by the intensity mask) into pre-allocated decay/fit/IRF
        cubes and parameter maps. If ``bool_mask`` was provided, it is
        applied as a final multiplicative mask.

        Returns:
            dict: ``{"raw_data": {"decay": <H,W,T>, "irf": <H,W,T>},
            "results": {"maps": {<param_name>: <H,W> ...}, "TR_maps":
            {"fit_map": <H,W,T>, "residual_map": decay_cube - fit_cube}}}``.

        Raises:
            ValueError: If ``bool_mask`` was provided but its shape does
                not match the image shape ``(H, W)``.
        """
        h, w = self.shape
        total_pixels = h * w

        # determining time-axis length
        first_roi = next(iter(self.roi_sims))
        sample = self.roi_sims[first_roi]()
        t_len = sample["raw_data"]["decay"].size

        # Pre-allocate
        decay_cube = np.zeros((h, w, t_len), dtype=np.float32)
        fit_cube = np.zeros((h, w, t_len), dtype=np.float32)
        irf_cube = np.zeros((h, w, t_len), dtype=np.float32)

        param_maps: dict[Any, np.ndarray] = {}

        if self.verbose:
            logging.info(
                f"Generating {self.method.upper()} FLI Image [{h}x{w}x{t_len}]..."
            )

        pixel_iterator = itertools.product(range(h), range(w))

        # --- tqdm OUTSIDE THE LOOP ---
        with tqdm(
            total=total_pixels,
            desc="Simulating Pixels",
            unit="px",
            disable=not self.verbose,
            leave=False,
        ) as pbar:
            for i, j in pixel_iterator:
                roi_val = self.roi_mask[i, j]
                sim = self.roi_sims[roi_val]

                # Pixel-wise IRF handling
                if self.irf_data.ndim == 3:
                    current_irf = self.irf_data[i, j, :]
                    irf_sum = current_irf.sum()
                    norm_irf = current_irf / irf_sum if irf_sum > 0 else current_irf
                    sim.engine.irf = norm_irf
                else:
                    norm_irf = sim.engine.irf

                # Run Simulation
                pixel_data = sim()
                m = self.intensity_mask[i, j]

                decay_cube[i, j, :] = pixel_data["raw_data"]["decay"] * m
                fit_cube[i, j, :] = pixel_data["results"]["TR_maps"]["fit_map"] * m
                irf_cube[i, j, :] = norm_irf

                if self._record_background_roi or roi_val != 0:
                    for k, v in pixel_data["results"]["maps"].items():
                        if k not in param_maps:
                            param_maps[k] = np.full((h, w), np.nan, dtype=np.float32)
                        param_maps[k][i, j] = v

                pbar.update(1)

        if self.bool_mask is not None:
            if self.bool_mask.shape != (h, w):
                raise ValueError(
                    f"bool_mask shape {self.bool_mask.shape} does not match image shape {(h, w)}."
                )
            m3 = self.bool_mask[
                :, :, np.newaxis
            ]  # (H, W, 1) for broadcasting over time axis
            decay_cube = decay_cube * m3
            fit_cube = fit_cube * m3
            irf_cube = irf_cube * m3
            for k in param_maps:
                param_maps[k] = param_maps[k] * self.bool_mask

        return {
            "raw_data": {"decay": decay_cube, "irf": irf_cube},
            "results": {
                "maps": param_maps,
                "TR_maps": {"fit_map": fit_cube, "residual_map": decay_cube - fit_cube},
            },
        }

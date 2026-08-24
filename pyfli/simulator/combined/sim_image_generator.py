"""
Generate FLI/FLIM image cubes from intensity images, ROI masks, and simulator settings.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. Public API
includes classes :class:`FLIImageGenerator`.

Shared mask loading, simulator selection, and pixel-loop logic live in
:mod:`pyfli.simulator.image_generator_common`.
"""

from ..image_generator_common import BaseFLIImageGenerator
from .main_factory import MacroSimulator, TCSPCSimulator


class FLIImageGenerator(BaseFLIImageGenerator):
    """
    Run the fliimage generator routine.
    simulator settings. It bridges pixel-level simulation with image-shaped datasets
    used by solvers and visualizers.

    Parameters
    ----------
    irf_data : np.ndarray
        Instrument response data used to convolve or simulate decays.
    intensity_image : str | np.ndarray | None
        Optional color or grayscale image (any bit depth) — a path to a PNG/
        TIFF/etc. file, or an already-loaded array — used to derive a
        per-pixel binary mask: any pixel with a nonzero value (any nonzero
        color channel, or a nonzero grayscale value) is foreground (1.0);
        pure zero/black is background (0.0). An already-binary source
        passes through unchanged. If omitted, a mask of ones with shape
        ``image_shape`` is used (no masking).
    roi_mask : str | np.ndarray | None
        Optional ROI label mask — a path to a grayscale/label image file, or
        an already-loaded integer-labeled array — where values 0, 1, 2, ...
        identify different ROIs. Resized (nearest-neighbor) to match the
        intensity mask shape if needed. If omitted, all pixels belong to
        ROI 0.
    roi_params : Any | None
        Parameters defining ROI shape, position, and intensity properties.
    image_shape : tuple[int, ...]
        Height and width of the generated image.
    method : str
        Algorithm or model-selection method to use.
    verbose : bool
        If ``True``, report progress and diagnostic messages during processing.
    bool_mask : str | np.ndarray | None
        Boolean mask selecting pixels included in the analysis — a path to
        an image file (binarized the same way as ``intensity_image``), or
        an already-loaded boolean/numeric array.
    """

    continuous_cls = MacroSimulator
    discrete_cls = TCSPCSimulator
    include_background_roi_in_maps = True

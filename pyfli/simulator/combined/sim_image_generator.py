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
    intensity_image_path : str | None
        Optional path to a color or grayscale image (any bit depth) used to
        derive a per-pixel binary mask: any pixel with a nonzero value (any
        nonzero color channel, or a nonzero grayscale value) is foreground
        (1.0); pure zero/black is background (0.0). An already-binary
        source image passes through unchanged. If omitted, a mask of ones
        with shape ``image_shape`` is used (no masking).
    roi_mask_path : str | None
        Filesystem path used by this workflow.
    roi_params : Any | None
        Parameters defining ROI shape, position, and intensity properties.
    image_shape : tuple[int, ...]
        Height and width of the generated image.
    method : str
        Algorithm or model-selection method to use.
    verbose : bool
        If ``True``, report progress and diagnostic messages during processing.
    bool_mask : np.ndarray | None
        Boolean mask selecting pixels included in the analysis.
    """

    continuous_cls = MacroSimulator
    discrete_cls = TCSPCSimulator
    include_background_roi_in_maps = True

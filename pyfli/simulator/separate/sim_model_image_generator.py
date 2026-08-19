# simulator/sim_model_image_generator.py

"""Full-image FLI dataset generation with per-ROI simulator configuration.

Provides ``FLIModelImageGenerator``, which builds a 2-D image of simulated
FLI pixel decays by assigning a ``ContinousEqSim`` (ICCD) or
``PhotonCountSim`` (photon-counting) simulator instance to each region of
interest (ROI) in an optional label mask, and simulating every pixel
individually.

Shared mask loading, simulator selection, and pixel-loop logic live in
:mod:`pyfli.simulator.image_generator_common`.
"""

from ..image_generator_common import BaseFLIImageGenerator
from .main_factory_gen import ContinuousSimulator, PhotonCountSimulator


class FLIModelImageGenerator(BaseFLIImageGenerator):
    """Generates a full 2-D FLI image dataset with per-ROI simulator configs.

    Loads an optional intensity mask (to scale per-pixel photon budget) and
    an optional ROI label mask (to assign different simulator parameters to
    different regions), instantiates one ``ContinousEqSim``/``PhotonCountSim``
    per ROI value, and simulates every pixel to build stacked decay/IRF/fit
    cubes and parameter maps.
    """

    continuous_cls = ContinuousSimulator
    discrete_cls = PhotonCountSimulator
    include_background_roi_in_maps = False

    def __init__(
        self,
        irf_data,
        intensity_image_path=None,
        roi_mask_path=None,
        roi_params=None,
        image_shape=(32, 32),
        method="continuous",
        verbose=True,
        bool_mask=None,
    ):
        """Loads masks and instantiates one simulator per ROI.

        Args:
            irf_data: IRF data (1-D trace or 3-D IRF stack). If 3-D, a
                per-pixel IRF slice is used during simulation; otherwise a
                single IRF (picked via ``irf_picker``) is shared across
                pixels.
            intensity_image_path: Optional path to a color or grayscale
                image (any bit depth) used to derive a per-pixel binary
                mask: any pixel with a nonzero value (any nonzero color
                channel, or a nonzero grayscale value) is foreground (1.0);
                pure zero/black is background (0.0). An already-binary
                source image passes through unchanged. If omitted, a mask
                of ones with shape ``image_shape`` is used (no masking).
            roi_mask_path: Optional path to a grayscale/label image where
                pixel values 0, 1, 2, ... identify different ROIs; resized
                (nearest-neighbor) to match the intensity mask shape. If
                omitted, all pixels belong to ROI 0.
            roi_params: Optional list of per-ROI config dicts (indexed by
                ROI value) forwarded as ``**cfg`` to the simulator
                constructor for that ROI; may include a ``sensor_type``
                key. ROI values without a matching entry get an empty
                config.
            image_shape: Fallback ``(H, W)`` image shape when no intensity
                image is provided.
            method: Simulation method string (case-insensitive); stored
                lower-cased as ``self.method``. Used only as the default
                ``sensor_type`` for ROIs whose config doesn't set one:
                ``'continuous'`` defaults to ``'continuous'`` (selecting
                ``ContinousEqSim``); any other value defaults to
                ``'discrete'`` (selecting ``PhotonCountSim``). Each ROI's
                effective ``sensor_type`` (explicit or defaulted) is what
                actually decides which of the two classes is used for
                that ROI.
            verbose: If True, prints progress info and shows the tqdm
                progress bar during ``generate_image``.
            bool_mask: Optional boolean array of shape ``image_shape``
                used to zero out pixels outside the mask in the final
                output cubes/maps.
        """
        super().__init__(
            irf_data=irf_data,
            intensity_image_path=intensity_image_path,
            roi_mask_path=roi_mask_path,
            roi_params=roi_params,
            image_shape=image_shape,
            method=method,
            verbose=verbose,
            bool_mask=bool_mask,
        )

# simulator/sim_image_generator.py

"""Full-image FLI dataset generation with per-ROI simulator configuration.

Provides ``FLIImageGenerator``, which builds a 2-D image of simulated FLI
pixel decays by assigning a ``Macro_sim`` (ICCD) or ``TCSPC_sim``
(photon-counting) simulator instance to each region of interest (ROI) in an
optional label mask, and simulating every pixel individually.
"""

import numpy as np
from PIL import Image
from tqdm import tqdm
import itertools
from .main_factory import Macro_sim, TCSPC_sim
from .sim_helper import irf_picker

class FLIImageGenerator:
    """Generates a full 2-D FLI image dataset with per-ROI simulator configs.

    Loads an optional intensity mask (to scale per-pixel photon budget) and
    an optional ROI label mask (to assign different simulator parameters to
    different regions), instantiates one ``Macro_sim``/``TCSPC_sim`` per
    ROI value, and simulates every pixel to build stacked decay/IRF/fit
    cubes and parameter maps.
    """

    def __init__(self,
                 irf_data,
                 intensity_image_path=None,
                 roi_mask_path=None,
                 roi_params=None,
                 image_shape=(32, 32),
                 method='ICCD',
                 verbose=True,
                 bool_mask=None
                 ):
        """Loads masks and instantiates one simulator per ROI.

        Args:
            irf_data: IRF data (1-D trace or 3-D IRF stack). If 3-D, a
                per-pixel IRF slice is used during simulation; otherwise a
                single IRF (picked via ``irf_picker``) is shared across
                pixels.
            intensity_image_path: Optional path to a grayscale image used
                as a per-pixel intensity scaling mask (values normalized to
                [0, 1]); if omitted, a mask of ones with shape
                ``image_shape`` is used.
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
                lower-cased as ``self.method``. ``'ICCD'`` selects
                ``Macro_sim`` with sensor_type ``'ICCD'``; any other value
                selects ``TCSPC_sim`` with sensor_type ``'PHOTON_COUNTER'``.
            verbose: If True, prints progress info and shows the tqdm
                progress bar during ``generate_image``.
            bool_mask: Optional boolean array of shape ``image_shape``
                used to zero out pixels outside the mask in the final
                output cubes/maps.
        """
        self.method = method.lower()
        self.irf_data = irf_data
        self.verbose = verbose
        self.bool_mask = np.asarray(bool_mask, dtype=bool) if bool_mask is not None else None
        
        # Loading the intensity Mask
        if intensity_image_path:
            img = Image.open(intensity_image_path).convert('L')
            self.intensity_mask = np.array(img).astype(float) / 255.0
            self.shape = self.intensity_mask.shape
        else:
            self.intensity_mask = np.ones(image_shape)
            self.shape = image_shape

        # loading the ROI Mask (multi-cluster mask)
        if roi_mask_path:
            mask_img = Image.open(roi_mask_path).convert('L')
            self.roi_mask = np.array(mask_img.resize((self.shape[1], self.shape[0]), 
                                     Image.NEAREST)).astype(int)
        else:
            self.roi_mask = np.zeros(self.shape, dtype=int)

        # Initialize ROI Simulators
        dummy_irf = irf_picker(irf_data)
        # dummy_irf = irf_data[0, 0, :] if irf_data.ndim == 3 else irf_data
        self.roi_sims = {}
        unique_rois = np.unique(self.roi_mask)
        SimClass = Macro_sim if self.method == 'iccd' else TCSPC_sim

        for roi_val in unique_rois:
            cfg = roi_params[roi_val].copy() if (roi_params and roi_val < len(roi_params)) else {}
            default_sensor = 'ICCD' if self.method == 'iccd' else 'PHOTON_COUNTER'
            sensor_type = cfg.pop('sensor_type', default_sensor)
            cfg.pop('method', None) 
            self.roi_sims[roi_val] = SimClass(dummy_irf, sensor_type=sensor_type, **cfg)

    def generate_image(self):
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
        
        param_keys = sample["results"]["maps"].keys()
        param_maps = {k: np.zeros((h, w), dtype=np.float32) for k in param_keys}

        if self.verbose:
            print(f"Generating {self.method.upper()} FLI Image [{h}x{w}x{t_len}]...")

        pixel_iterator = itertools.product(range(h), range(w))
        
        # --- tqdm OUTSIDE THE LOOP ---
        with tqdm(total=total_pixels, 
                  desc="Simulating Pixels", 
                  unit="px",
                  disable=not self.verbose,
                  leave=False) as pbar:
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
                
                for k in param_keys:
                    param_maps[k][i, j] = pixel_data["results"]["maps"][k]
                
                # Manually update the bar
                pbar.update(1)
        
        if self.bool_mask is not None:
            if self.bool_mask.shape != (h, w):
                raise ValueError(
                    f"bool_mask shape {self.bool_mask.shape} does not match image shape {(h, w)}."
                )
            m3 = self.bool_mask[:, :, np.newaxis]   # (H, W, 1) for broadcasting over time axis
            decay_cube   = decay_cube   * m3
            fit_cube     = fit_cube     * m3
            irf_cube     = irf_cube     * m3
            for k in param_maps:
                param_maps[k] = param_maps[k] * self.bool_mask

        return {
            "raw_data": {"decay": decay_cube,
                         "irf": irf_cube},
            "results": {"maps": param_maps,
                        "TR_maps": {"fit_map": fit_cube,
                        "residual_map": decay_cube - fit_cube}}
                    }
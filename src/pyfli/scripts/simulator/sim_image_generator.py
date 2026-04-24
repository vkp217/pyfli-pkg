# simulator/sim_image_generator.py
import numpy as np
from PIL import Image
from tqdm import tqdm
import itertools  
from .main_factory import Macro_sim, TCSPC_sim
from .sim_helper import irf_picker

class FLIImageGenerator:
    def __init__(self, 
                 irf_data, 
                 intensity_image_path=None, 
                 roi_mask_path=None, 
                 roi_params=None, 
                 image_shape=(32, 32), 
                 method='ICCD',
                 verbose = True
                 ):
        self.method = method.lower()        
        self.irf_data = irf_data
        self.verbose = verbose
        
        # Loading the intensity Mask
        if intensity_image_path:
            img = Image.open(intensity_image_path).convert('L')
            self.intensity_mask = np.array(img).astype(float) / 255.0
            self.shape = self.intensity_mask.shape
        else:
            self.intensity_mask = np.ones(image_shape)
            self.shape = image_shape

        # loading the ROI Mask (multi-color mask)
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
        SimClass = Macro_sim if self.method == 'ICCD' else TCSPC_sim
        # SimClass = TCSPC_sim if self.method == 'tcspc' else Macro_sim
        
        for roi_val in unique_rois:
            cfg = roi_params[roi_val].copy() if (roi_params and roi_val < len(roi_params)) else {}
            default_sensor = 'ICCD' if self.method == 'ICCD' else 'SPAD'
            sensor_type = cfg.pop('sensor_type', default_sensor)
            cfg.pop('method', None) 
            self.roi_sims[roi_val] = SimClass(dummy_irf, sensor_type=sensor_type, **cfg)

    def generate_image(self):
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
                fit_cube[i, j, :] = pixel_data["TR_maps"]["fit_map"] * m
                irf_cube[i, j, :] = norm_irf 
                
                for k in param_keys:
                    param_maps[k][i, j] = pixel_data["results"]["maps"][k]
                
                # Manually update the bar
                pbar.update(1)
        
        return {
            "raw_data": {"decay": decay_cube, "irf": irf_cube},
            "results": {"maps": param_maps},
            "TR_maps": {"fit_map": fit_cube, "residuals_map": decay_cube - fit_cube}
        }
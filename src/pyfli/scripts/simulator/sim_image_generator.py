# simulator/sim_image_generator.py
import numpy as np
from PIL import Image
from tqdm import tqdm
from .main_factory import Macro_sim, TCSPC_sim

class FLIImageGenerator:
    def __init__(self, 
                 irf_data, 
                 intensity_image_path=None, 
                 roi_mask_path=None, 
                 roi_params=None, 
                 image_shape=(32, 32), 
                 method='analytical'):
        """
        Generates a synthetic FLI image. Supports 1D (uniform) or 3D (pixel-wise) IRFs.
        """
        self.method = method.lower()        
        self.irf_data = irf_data # Store the full IRF data (1D or 3D)
        
        # 1. Load Intensity Mask
        if intensity_image_path:
            img = Image.open(intensity_image_path).convert('L')
            self.intensity_mask = np.array(img).astype(float) / 255.0
            self.shape = self.intensity_mask.shape
        else:
            self.intensity_mask = np.ones(image_shape)
            self.shape = image_shape

        # 2. Load ROI Mask
        if roi_mask_path:
            mask_img = Image.open(roi_mask_path).convert('L')
            self.roi_mask = np.array(mask_img.resize((self.shape[1], self.shape[0]), 
                                     Image.NEAREST)).astype(int)
        else:
            self.roi_mask = np.zeros(self.shape, dtype=int)

        # 3. Initialize ROI Simulators
        # We use a dummy IRF slice for init; we will swap it pixel-wise in generate_image
        dummy_irf = irf_data[0, 0, :] if irf_data.ndim == 3 else irf_data
        
        self.roi_sims = {}
        unique_rois = np.unique(self.roi_mask)
        SimClass = TCSPC_sim if self.method == 'tcspc' else Macro_sim
        
        for roi_val in unique_rois:
            cfg = roi_params[roi_val] if (roi_params and roi_val < len(roi_params)) else {}
            sensor_type = cfg.get('sensor_type', 'ICCD' if self.method == 'analytical' else 'SPAD')
            self.roi_sims[roi_val] = SimClass(dummy_irf, sensor_type=sensor_type, **cfg)

    def generate_image(self):
        h, w = self.shape
        
        # Determine time-axis length
        first_roi = next(iter(self.roi_sims))
        sample = self.roi_sims[first_roi]()
        t_len = sample["raw_data"]["decay"].size
        
        # Pre-allocate
        decay_cube = np.zeros((h, w, t_len), dtype=np.float32)
        fit_cube = np.zeros((h, w, t_len), dtype=np.float32)
        irf_cube = np.zeros((h, w, t_len), dtype=np.float32)
        
        param_keys = sample["results"]["maps"].keys()
        param_maps = {k: np.zeros((h, w), dtype=np.float32) for k in param_keys}

        print(f"Generating {self.method.upper()} FLI Image [{h}x{w}x{t_len}] with Pixel-wise IRF...")
        
        for i in tqdm(range(h), desc="Rows"):
            for j in range(w):
                roi_val = self.roi_mask[i, j]
                sim = self.roi_sims[roi_val]
                
                # --- PIXEL-WISE IRF SWAP ---
                # If IRF is 3D, extract the specific pixel's IRF
                if self.irf_data.ndim == 3:
                    current_irf = self.irf_data[i, j, :]
                    # Normalize and update the internal engine of the simulator
                    norm_irf = np.nan_to_num(current_irf / current_irf.sum())
                    sim.engine.irf = norm_irf
                else:
                    norm_irf = sim.engine.irf # Already normalized during init
                
                # Run Simulation for this pixel
                pixel_data = sim()
                m = self.intensity_mask[i, j]
                
                decay_cube[i, j, :] = pixel_data["raw_data"]["decay"] * m
                fit_cube[i, j, :] = pixel_data["TR_maps"]["fit_map"] * m
                irf_cube[i, j, :] = norm_irf # Store the used IRF for this pixel
                
                for k in param_keys:
                    param_maps[k][i, j] = pixel_data["results"]["maps"][k]
        
        return {
            "raw_data": {
                "decay": decay_cube, 
                "irf": irf_cube 
            },
            "results": {
                "maps": param_maps
            },
            "TR_maps": {
                "fit_map": fit_cube, 
                "residuals_map": decay_cube - fit_cube
            }
        }
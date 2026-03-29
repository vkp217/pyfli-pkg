# simulator/sim_image_generator.py
import numpy as np
from PIL import Image
from tqdm import tqdm
from .main_factory import Macro_sim, TCSPC_sim

class FLIImageGenerator:
    def __init__(self, irf_data, intensity_image_path=None, roi_mask_path=None, 
                 roi_params=None, image_shape=(32, 32), method='analytical'):
        self.method = method.lower()
        
        # Load Masks
        if intensity_image_path:
            self.intensity_mask = np.array(Image.open(intensity_image_path).convert('L')).astype(float)
            self.shape = self.intensity_mask.shape
        else:
            self.intensity_mask = np.ones(image_shape)
            self.shape = image_shape

        if roi_mask_path:
            mask_img = Image.open(roi_mask_path).convert('L')
            self.roi_mask = np.array(mask_img.resize((self.shape[1], self.shape[0]), Image.NEAREST)).astype(int)
        else:
            self.roi_mask = np.zeros(self.shape, dtype=int)

        # Initialize ROI Simulators
        self.roi_sims = {}
        unique_rois = np.unique(self.roi_mask)
        SimClass = TCSPC_sim if self.method == 'tcspc' else Macro_sim
        
        for idx, roi_val in enumerate(unique_rois):
            cfg = roi_params[idx] if (roi_params and idx < len(roi_params)) else {}
            self.roi_sims[roi_val] = SimClass(irf_data, **cfg)

    def generate_image(self):
        h, w = self.shape
        # Get a sample to determine time-axis length
        sample = self.roi_sims[next(iter(self.roi_sims))]()
        t_len = sample["raw_data"]["decay"].size
        
        dataset = {
            "raw_data": {"decay": np.zeros((h, w, t_len)), "irf": np.zeros((h, w, t_len))},
            "results": {"maps": {k: np.zeros((h, w)) for k in sample["results"]["maps"].keys()}},
            "TR_maps": {"fit_map": np.zeros((h, w, t_len)), "residuals_map": np.zeros((h, w, t_len))}
        }

        for i in range(h):
            for j in range(w):
                roi_val = self.roi_mask[i, j]
                pixel = self.roi_sims[roi_val]()
                m = self.intensity_mask[i, j]
                
                # Assign with intensity weighting
                dataset["raw_data"]["decay"][i, j] = pixel["raw_data"]["decay"] * m
                dataset["raw_data"]["irf"][i, j] = pixel["raw_data"]["irf"]
                dataset["TR_maps"]["fit_map"][i, j] = pixel["TR_maps"]["fit_map"] * m
                dataset["TR_maps"]["residuals_map"][i, j] = (pixel["raw_data"]["decay"] - pixel["TR_maps"]["fit_map"]) * m
                
                for k, v in pixel["results"]["maps"].items():
                    dataset["results"]["maps"][k][i, j] = v
        
        return dataset
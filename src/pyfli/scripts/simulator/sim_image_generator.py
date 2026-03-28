import numpy as np
from PIL import Image
from tqdm import tqdm
from .main_factory import Macro_sim, TCSPC_sim

class FLIImageGenerator:
    def __init__(self, irf_data, intensity_image_path=None, roi_mask_path=None, 
                 roi_params=None, image_shape=(32, 32), method='analytical'):
        
        self.method = method.lower()
        self.irf_data = irf_data
        
        # 1. Load Intensity/Spatial Information
        if intensity_image_path:
            img = Image.open(intensity_image_path).convert('L')
            self.intensity_mask = np.array(img).astype(np.float64)
            self.shape = self.intensity_mask.shape
        else:
            self.intensity_mask = np.ones(image_shape)
            self.shape = image_shape

        # 2. Load ROI Mask
        if roi_mask_path:
            mask_img = Image.open(roi_mask_path).convert('L')
            self.roi_mask = np.array(mask_img.resize((self.shape[1], self.shape[0]), Image.NEAREST)).astype(np.int32)
        else:
            self.roi_mask = np.zeros(self.shape, dtype=np.int32)

        # 3. Initialize Internal Simulators per ROI
        self.roi_sims = {}
        unique_rois = np.unique(self.roi_mask)
        SimClass = TCSPC_sim if self.method == 'tcspc' else Macro_sim
        
        for idx, roi_val in enumerate(unique_rois):
            params = roi_params[idx] if (roi_params and idx < len(roi_params)) else {}
            # Pass IRF and config to the chosen simulator
            self.roi_sims[roi_val] = SimClass(self.irf_data, **params)

        # Initialize storage based on a sample run
        sample = self.roi_sims[unique_rois[0]]()
        n_t = sample["s_t"].size

        self.decay_image = np.zeros((*self.shape, n_t))
        self.tau_mean_map = np.zeros(self.shape)
        self.f_map = np.zeros(self.shape)

    def generate_image(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                roi_val = self.roi_mask[i, j]
                pixel_data = self.roi_sims[roi_val]()

                # Apply spatial intensity mask if needed
                self.decay_image[i, j, :] = pixel_data["s_t"] * self.intensity_mask[i, j]
                
                # Extract ground truth maps
                self.tau_mean_map[i, j] = (pixel_data['tau1'] * pixel_data['f'] + 
                                           pixel_data['tau2'] * (1 - pixel_data['f']))
                self.f_map[i, j] = pixel_data['f']

        return self.decay_image, self.tau_mean_map
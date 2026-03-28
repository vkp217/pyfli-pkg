#  solver/flicpuFitter.py 
import numpy as np
import h5py
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from .globalFitter import GlobalFLIFitter

class Fli_CPUProcessor:
    def __init__(self, freq, fitter_class):
        """
        freq: [freq_laser, freq_acquisition]
        fitter_class: The BaseFLIFitter, MLEFLIFitter, or GlobalFLIFitter class
        """
        self.freq = freq
        self.fitter_class = fitter_class

    def _fit_task(self, y_data, irf_p, coords, model_type, estimator, p0, bounds, kwargs):
        """Internal task for Parallel execution."""
        y_data = y_data.astype(np.float32)
        fitter = self.fitter_class(self.freq, y_data, irf_p)
        
        res = list(fitter.fit_with_estimator(
            estimator_type=estimator, 
            model_type=model_type, 
            p0=p0, 
            bounds=bounds, 
            **kwargs
        ))
        
        popt, perr = res[0], res[1]

        if model_type == 'bi-exponential' and popt[2] > popt[3]:
            avg_pre = popt[1] * popt[2] + (1.0 - popt[1]) * popt[3]
            popt[2], popt[3] = popt[3], popt[2]
            popt[1] = 1.0 - popt[1]
            if perr is not None: perr[2], perr[3] = perr[3], perr[2]
            
            avg_post = popt[1] * popt[2] + (1.0 - popt[1]) * popt[3]
            if not np.isclose(avg_pre, avg_post, atol=1e-9):
                print(f"Warning at {coords}: Swap failed sanity check.")
            res[0], res[1] = popt, perr

        fit_curve = fitter.model_fit(fitter.t, popt, model_type=model_type).astype(np.float32)
        residual = (y_data - fit_curve).astype(np.float32)
        return coords, res, fit_curve, residual

    def process_image(self, image_cube, irf_cube, mask=None, data_name="FLIM_Dataset", 
                      model_type='bi-exponential', estimator='least_squares', 
                      p0=None, bounds=None, n_jobs=-1, backend='loky', **kwargs):
        """
        Processes image. Automatically detects if mask is binary or multi-label clusters.
        """
        H, W, T = image_cube.shape
        
        # 1. Detect Multi-label Clustering
        if mask is not None and np.max(mask) > 1 and issubclass(self.fitter_class, GlobalFLIFitter):
            print(f"Multi-label mask detected. Switching to Global Cluster Fitting...")
            g_fitter = self.fitter_class(self.freq, image_cube[0,0,:], irf_cube[0,0,:])
            return g_fitter.process_clusters(image_cube, irf_cube, mask, estimator=estimator, 
                                             model_type=model_type, **kwargs)

        # 2. Standard Pixel-wise Processing
        if mask is None: mask = np.sum(image_cube, axis=2) > 20
        final_coords = [(r, c) for r, c in np.argwhere(mask) if np.sum(image_cube[r, c, :]) > 0]
        
        if not final_coords: return None

        tasks = (delayed(self._fit_task)(image_cube[r,c,:], irf_cube[r,c,:], (r,c), 
                 model_type, estimator, p0, bounds, kwargs) for r, c in final_coords)

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            tqdm(tasks, total=len(final_coords), desc="Fitting Pixels", unit="px"))

        # --- Reconstruction Logic (p_maps, e_maps, etc.) ---
        internal_popt_len = 5 if model_type == 'bi-exponential' else 3
        p_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)
        e_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)
        r2_map, chi2_map, red_chi2_map, conv_map = [np.zeros((H, W), dtype=np.float32) for _ in range(4)]
        fit_map, res_map = np.zeros((H, W, T)), np.zeros((H, W, T))

        for coords, res, f_curve, r_curve in results:
            r, c = coords
            p_maps[r, c, :], e_maps[r, c, :] = res[0], res[1]
            r2_map[r, c], chi2_map[r, c], red_chi2_map[r, c], conv_map[r, c] = res[2], res[3], res[4], res[6]
            fit_map[r, c, :], res_map[r, c, :] = f_curve, r_curve

        # Parameter Mapping
        S = p_maps[..., 0]
        if model_type == 'bi-exponential':
            a1 = p_maps[..., 1]
            param_maps = {'Area_map': S, 'alpha1_map': a1, 'tau1_map': p_maps[..., 2], 
                          'tau2_map': p_maps[..., 3], 'B_map': p_maps[..., 4]}
        else:
            param_maps = {'Area_map': S, 'tau_map': p_maps[..., 1], 'B_map': p_maps[..., 2]}
        
        param_maps.update({'R2_map': r2_map, 'reduced_chi2_map': red_chi2_map, 'convergence_map': conv_map})
        return {'name': data_name, 'results': {'maps': param_maps, 'TR_maps': {'fit_map': fit_map, 'residual_map': res_map}}}

    def process_spatial_global(self, image_cube, irf_cube, window_size=3, step=1, model_type='bi-exponential'):
        """
        Applies spatial global fitting in NxN neighborhoods.
        Useful for high-noise data where local pixels share the same fluorophore environment.
        """
        if not issubclass(self.fitter_class, GlobalFLIFitter):
            raise TypeError("Spatial Global fitting requires GlobalFLIFitter class.")
            
        H, W, T = image_cube.shape
        pad = window_size // 2
        # Initialize result maps for shared lifetimes
        tau1_map = np.zeros((H, W))
        tau2_map = np.zeros((H, W))
        a1_map = np.zeros((H, W))
        
        g_fitter = self.fitter_class(self.freq, image_cube[0,0,:], irf_cube[0,0,:])
        
        # Iterate through image with sliding window
        for r in tqdm(range(pad, H - pad, step), desc="Spatial Global Fitting"):
            for c in range(pad, W - pad, step):
                # Extract NxN neighborhood
                pixel_group = image_cube[r-pad:r+pad+1, c-pad:c+pad+1, :].reshape(-1, T)
                
                # Fit the neighborhood globally
                popt, success = g_fitter.global_mle(pixel_group, model_type=model_type)
                
                if success:
                    # Shared params are at the start of popt: [a1, t1, t2]
                    a1_map[r, c] = popt[0]
                    tau1_map[r, c] = popt[1]
                    tau2_map[r, c] = popt[2]
                    
        return {'tau1_map': tau1_map, 'tau2_map': tau2_map, 'alpha1_map': a1_map}

    def save_results(self, dataset, folder="results"):
        """Saves the structured dataset to HDF5."""
        if dataset is None: return
        if not os.path.exists(folder): os.makedirs(folder)
        h5_path = os.path.join(folder, f"{dataset['name']}_results.h5")
        with h5py.File(h5_path, "w") as f:
            res_grp = f.create_group("results")
            maps_grp = res_grp.create_group("maps")
            for k, v in dataset['results']['maps'].items():
                maps_grp.create_dataset(k, data=v, compression="gzip")
        print(f"Dataset saved to: {h5_path}")

    def load_map(self, h5_path, map_name='tau1_map'):
        """Utility to reload a specific slice for visualization."""
        with h5py.File(h5_path, 'r') as f:
            if map_name in f['results/maps']:
                return f[f'results/maps/{map_name}'][()]
            else:
                print(f"Map {map_name} not found.")
                return None
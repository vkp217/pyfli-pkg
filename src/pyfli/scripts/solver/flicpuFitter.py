# solver/flicpuFitter.py
import numpy as np
import h5py
import os
from joblib import Parallel, delayed
from tqdm import tqdm
# from .globalFitter import GlobalFLIFitter

class Fli_CPUProcessor:
    def __init__(self, freq, fitter_class):
        """
        freq: [freq_laser, freq_acquisition]
        fitter_class: The BaseFLIFitter, MLEFLIFitter, or GlobalFLIFitter class
        """
        self.freq = freq
        self.fitter_class = fitter_class

    def _fit_task(self, y_data, irf_p, coords, model_type, estimator, p0, bounds, kwargs):
        """
        Internal task for Parallel execution with foolproof error handling.
        Returns a 'health' status: 1 for success, 0 for failure.
        """
        # Convert to float32 for memory efficiency
        y_data = y_data.astype(np.float32)
        irf_p = irf_p.astype(np.float32)
        
        try:
            # Initialize the specific fitter instance for this pixel
            fitter = self.fitter_class(self.freq, y_data, irf_p)
            
            # Execute fit - Registry logic handles the offset/bkg mapping internally
            res = fitter.fit_with_estimator(
                estimator_type=estimator, 
                model_type=model_type, 
                p0=p0, 
                bounds=bounds, 
                **kwargs
            )
            
            # Extract popt for curve generation (res[0] is popt)
            popt = res[0]
            
            # Generate the fit curve and residuals for the full acquisition time
            fit_curve = fitter.model_fit(fitter.t, popt, model_type=model_type).astype(np.float32)
            residual = (y_data - fit_curve).astype(np.float32)
            
            health = 1 # Pixel fit successfully
            return coords, res, fit_curve, residual, health

        except Exception as e:
            # Generate 'dummy' results to maintain array shapes on failure
            n_params = 5 if model_type == 'bi-exponential' else 3
            dummy_res = (
                np.zeros(n_params), # popt
                np.zeros(n_params), # perr
                0.0,                # r_sq
                0.0,                # stat
                0.0,                # red_stat
                0.0,                # ssr
                0                   # converged
            )
            dummy_curve = np.zeros_like(y_data)
            health = 0 # Pixel failed
            return coords, dummy_res, dummy_curve, dummy_curve, health

    def process_image(self, image_cube, irf_cube, mask=None, data_name="FLIM_Dataset", 
                      model_type='bi-exponential', estimator='least_squares', 
                      p0=None, bounds=None, n_jobs=-1, backend='loky', **kwargs):
        """
        Processes image. Automatically detects if mask is binary or multi-label clusters.
        Generates a 'pixel_health_map' to identify problematic fits.
        """
        H, W, T = image_cube.shape
        
        # 1. Detect Multi-label Clustering for Global Fitting
        if mask is not None and np.max(mask) > 1 and issubclass(self.fitter_class, GlobalFLIFitter):
            print(f"Multi-label mask detected. Switching to Global Cluster Fitting...")
            g_fitter = self.fitter_class(self.freq, image_cube[0,0,:], irf_cube[0,0,:])
            return g_fitter.process_clusters(image_cube, irf_cube, mask, estimator=estimator, 
                                             model_type=model_type, **kwargs)

        # 2. Standard Pixel-wise Processing
        if mask is None: 
            mask = np.sum(image_cube, axis=2) > 20 # Basic auto-threshold
            
        final_coords = [(r, c) for r, c in np.argwhere(mask) if np.sum(image_cube[r, c, :]) > 0]
        
        if not final_coords: 
            print("No valid pixels found in mask.")
            return None

        # Prepare parallel tasks
        tasks = (delayed(self._fit_task)(image_cube[r,c,:], irf_cube[r,c,:], (r,c), 
                 model_type, estimator, p0, bounds, kwargs) for r, c in final_coords)

        # Execute
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            tqdm(tasks, total=len(final_coords), desc=f"Fitting Pixels ({estimator})", unit="px"))

        # --- Reconstruction Logic ---
        internal_popt_len = 5 if model_type == 'bi-exponential' else 3
        p_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)
        e_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)
        
        # Result tuple indices: 0:popt, 1:perr, 2:r_sq, 3:stat, 4:red_stat, 5:ssr, 6:conv
        r2_map = np.zeros((H, W), dtype=np.float32)
        stat_map = np.zeros((H, W), dtype=np.float32)
        red_stat_map = np.zeros((H, W), dtype=np.float32)
        conv_map = np.zeros((H, W), dtype=np.float32)
        pixel_health_map = np.zeros((H, W), dtype=np.float32) # 1=Good, 0=Problem
        
        fit_map = np.zeros((H, W, T), dtype=np.float32)
        res_map = np.zeros((H, W, T), dtype=np.float32)

        for coords, res, f_curve, r_curve, health in results:
            r, c = coords
            pixel_health_map[r, c] = health
            p_maps[r, c, :] = res[0]
            e_maps[r, c, :] = res[1]
            r2_map[r, c] = res[2]
            stat_map[r, c] = res[3]
            red_stat_map[r, c] = res[4]
            conv_map[r, c] = res[6]
            fit_map[r, c, :] = f_curve
            res_map[r, c, :] = r_curve

        # Build final parameter dictionary
        S_map = p_maps[..., 0]
        if model_type == 'bi-exponential':
            param_maps = {
                'Area_map': S_map, 
                'alpha1_map': p_maps[..., 1], 
                'tau1_map': p_maps[..., 2], 
                'tau2_map': p_maps[..., 3], 
                'offset_map': p_maps[..., 4] # Corrected 'B_map' to 'offset_map'
            }
        else:
            param_maps = {
                'Area_map': S_map, 
                'tau_map': p_maps[..., 1], 
                'offset_map': p_maps[..., 2] # Corrected 'B_map' to 'offset_map'
            }
        
        # Add statistical and health maps
        param_maps.update({
            'R2_map': r2_map, 
            'chi2_map': stat_map,
            'reduced_stat_map': red_stat_map, 
            'convergence_map': conv_map,
            'pixel_health_map': pixel_health_map # The requested "Problem" vs "Good" map
        })
        
        return {
            'name': data_name, 
            'results': {
                'maps': param_maps, 
                'error_maps': e_maps,
                'TR_maps': {'fit_map': fit_map, 'residual_map': res_map}
            }
        }

    # def process_spatial_global(self, image_cube, irf_cube, window_size=3, step=1, model_type='bi-exponential'):
    #     """
    #     Applies spatial global fitting in NxN neighborhoods.
    #     """
    #     if not issubclass(self.fitter_class, GlobalFLIFitter):
    #         raise TypeError("Spatial Global fitting requires GlobalFLIFitter class.")
            
    #     H, W, T = image_cube.shape
    #     pad = window_size // 2
        
    #     tau1_map = np.zeros((H, W))
    #     tau2_map = np.zeros((H, W))
    #     a1_map = np.zeros((H, W))
        
    #     # Initialize a global fitter instance
    #     g_fitter = self.fitter_class(self.freq, image_cube[0,0,:], irf_cube[0,0,:])
        
    #     for r in tqdm(range(pad, H - pad, step), desc="Spatial Global Fitting"):
    #         for c in range(pad, W - pad, step):
    #             # Extract neighborhood and flatten
    #             pixel_group = image_cube[r-pad:r+pad+1, c-pad:c+pad+1, :].reshape(-1, T)
                
    #             # Fit the neighborhood globally
    #             popt, perr, rsq, stat, red_stat, ssr, success = g_fitter.fit_with_estimator(
    #                 model_type=model_type, pixel_group=pixel_group
    #             )
                
    #             if success:
    #                 a1_map[r, c] = popt[0]
    #                 tau1_map[r, c] = popt[1]
    #                 tau2_map[r, c] = popt[2]
                    
    #     return {'tau1_map': tau1_map, 'tau2_map': tau2_map, 'alpha1_map': a1_map}

    def save_results(self, dataset, folder="results"):
        """Saves the structured dataset to HDF5 with compression."""
        if dataset is None: return
        if not os.path.exists(folder): os.makedirs(folder)
        
        h5_path = os.path.join(folder, f"{dataset['name']}_results.h5")
        with h5py.File(h5_path, "w") as f:
            res_grp = f.create_group("results")
            
            # Save Parameter Maps
            maps_grp = res_grp.create_group("maps")
            for k, v in dataset['results']['maps'].items():
                maps_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            
            # Save Error Maps separately
            err_grp = res_grp.create_group("error_maps")
            err_grp.create_dataset("errors", data=dataset['results']['error_maps'], compression="gzip", compression_opts=4)

            # Save Full Time-Resolved Curves
            tr_grp = res_grp.create_group("TR_maps")
            for k, v in dataset['results']['TR_maps'].items():
                tr_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)
                
        print(f"Analysis complete. Results saved to: {h5_path}")

    def load_map(self, h5_path, map_name='tau1_map'):
        """Utility to reload a specific slice for visualization."""
        with h5py.File(h5_path, 'r') as f:
            if f'results/maps/{map_name}' in f:
                return f[f'results/maps/{map_name}'][()]
            else:
                print(f"Map {map_name} not found in {h5_path}")
                return None
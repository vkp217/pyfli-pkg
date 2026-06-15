import numpy as np
import h5py
import os
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from .globalFitter import GlobalFLIFitter as _GlobalFLIFitter
except ImportError:
    _GlobalFLIFitter = None

class Fli_CPUProcessor:
    def __init__(self, freq, fitter_class):
        self.freq = freq
        self.fitter_class = fitter_class

    def _fit_task(self, y_data, irf_p, coords, model_type, estimator, p0, bounds, kwargs):
        y_data = y_data.astype(np.float32)
        irf_p = irf_p.astype(np.float32)
        
        try:
            shift_method = kwargs.get('shift_method', 'zero_pad')
            fit_kwargs = {k: v for k, v in kwargs.items() if k != 'shift_method'}

            fitter = self.fitter_class(self.freq, y_data, irf_p, shift_method=shift_method)

            res = fitter.fit_with_estimator(
                estimator_type=estimator,
                model_type=model_type,
                p0=p0,
                bounds=bounds,
                **fit_kwargs
            )

            popt = res[0]

            fit_curve = fitter.model_fit(fitter.t, popt, model_type=model_type).astype(np.float32)
            residual = (y_data - fit_curve).astype(np.float32)

            health = 1
            return coords, res, fit_curve, residual, health

        except Exception as e:
            n_params = 6 if model_type == 'bi-exponential' else 4
            dummy_res = (
                np.zeros(n_params),
                np.zeros(n_params),
                0.0,
                0.0,
                0.0,
                0.0,
                0
            )
            dummy_curve = np.zeros_like(y_data)
            health = 0
            return coords, dummy_res, dummy_curve, dummy_curve, health

    def process_image(self, image_cube, irf_cube, mask=None, data_name="FLIM_Dataset", 
                      model_type='bi-exponential', estimator='least_squares', 
                      p0=None, bounds=None, n_jobs=-1, backend='loky', **kwargs):
        H, W, T = image_cube.shape
        
        if mask is not None and np.max(mask) > 1 and _GlobalFLIFitter is not None and issubclass(self.fitter_class, _GlobalFLIFitter):
            print(f"Multi-label mask detected. Switching to Global Cluster Fitting...")
            g_fitter = self.fitter_class(self.freq, image_cube[0,0,:], irf_cube[0,0,:])
            return g_fitter.process_clusters(image_cube, irf_cube, mask, estimator=estimator, 
                                             model_type=model_type, **kwargs)

        if mask is None: 
            mask = np.sum(image_cube, axis=2) > 20
            
        final_coords = [(r, c) for r, c in np.argwhere(mask) if np.sum(image_cube[r, c, :]) > 0]
        
        if not final_coords: 
            print("No valid pixels found in mask.")
            return None

        tasks = (delayed(self._fit_task)(image_cube[r,c,:], irf_cube[r,c,:], (r,c), 
                 model_type, estimator, p0, bounds, kwargs) for r, c in final_coords)

        results = Parallel(n_jobs=n_jobs, backend=backend)(
            tqdm(tasks, total=len(final_coords), desc=f"Fitting Pixels ({estimator})", unit="px"))

        internal_popt_len = 6 if model_type == 'bi-exponential' else 4
        p_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)
        e_maps = np.zeros((H, W, internal_popt_len), dtype=np.float32)
        
        r2_map = np.zeros((H, W), dtype=np.float32)
        stat_map = np.zeros((H, W), dtype=np.float32)
        red_stat_map = np.zeros((H, W), dtype=np.float32)
        conv_map = np.zeros((H, W), dtype=np.float32)
        pixel_health_map = np.zeros((H, W), dtype=np.float32)
        
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

        S_map = p_maps[..., 0]
        if model_type == 'bi-exponential':
            tau1_m, tau2_m = p_maps[..., 2], p_maps[..., 3]
            param_maps = {
                'Area_map': S_map,
                'alpha1_map': p_maps[..., 1],
                'tau1_map': tau1_m,
                'tau2_map': tau2_m,
                'offset_map': p_maps[..., 4],
                'fret_efficiency_map': np.where(tau2_m > 0, 1.0 - tau1_m / tau2_m, 0.0).astype(np.float32),
                'h_shift_map': p_maps[..., 5].astype(np.float32),
            }
        else:
            param_maps = {
                'Area_map': S_map,
                'tau_map': p_maps[..., 1],
                'offset_map': p_maps[..., 2],
                'h_shift_map': p_maps[..., 3].astype(np.float32),
            }
        
        param_maps.update({
            'R2_map': r2_map, 
            'chi2_map': stat_map,
            'reduced_stat_map': red_stat_map, 
            'convergence_map': conv_map,
            'pixel_health_map': pixel_health_map
        })
        
        return {
            'name': data_name, 
            'results': {
                'maps': param_maps, 
                'error_maps': e_maps,
                'TR_maps': {'fit_map': fit_map, 'residual_map': res_map}
            }
        }

    def save_results(self, dataset, folder="results"):
        if dataset is None: return
        if not os.path.exists(folder): os.makedirs(folder)
        
        h5_path = os.path.join(folder, f"{dataset['name']}_results.h5")
        with h5py.File(h5_path, "w") as f:
            res_grp = f.create_group("results")
            
            maps_grp = res_grp.create_group("maps")
            for k, v in dataset['results']['maps'].items():
                maps_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            
            err_grp = res_grp.create_group("error_maps")
            err_grp.create_dataset("errors", data=dataset['results']['error_maps'], compression="gzip", compression_opts=4)

            tr_grp = res_grp.create_group("TR_maps")
            for k, v in dataset['results']['TR_maps'].items():
                tr_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)
                
        print(f"Analysis complete. Results saved to: {h5_path}")

    def load_map(self, h5_path, map_name='tau1_map'):
        with h5py.File(h5_path, 'r') as f:
            if f'results/maps/{map_name}' in f:
                return f[f'results/maps/{map_name}'][()]
            else:
                print(f"Map {map_name} not found in {h5_path}")
                return None
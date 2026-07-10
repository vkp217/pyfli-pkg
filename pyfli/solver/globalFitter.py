# solver/globalFitter.py
import numpy as np
import torch
import time
import re
from tqdm import tqdm
from tabulate import tabulate
from .comparison import FittingComparator
from .base_static import resolve_params_and_bounds

class GlobalFLIFitter:
    def __init__(self, freq, base_fitter_class, mle_fitter_class, processor_instance=None):
        self.freq = freq
        self.BaseClass = base_fitter_class
        self.MLEClass = mle_fitter_class
        self.processor = processor_instance
        
        self.comparator = FittingComparator(
            freq=self.freq, 
            base_fitter_class=self.BaseClass, 
            mle_fitter_class=self.MLEClass
        )
        
        self.T_acq = 1000.0 / freq[1]
        self.cluster_data = {}

    def make_clusters(self, image_cube, irf_cube, cluster_mask, min_cluster_size=10):
        """Extracts cluster-specific data and stores spatial coordinates for reconstruction."""
        self.cluster_data = {}
        cluster_ids = np.unique(cluster_mask)[np.unique(cluster_mask) != 0]
        
        for cid in cluster_ids:
            coords = np.argwhere(cluster_mask == cid)
            if len(coords) < min_cluster_size: 
                continue
            
            self.cluster_data[f'cluster_{cid}'] = {
                'decay': image_cube[coords[:, 0], coords[:, 1], :],
                'irf': irf_cube[coords[:, 0], coords[:, 1], :],
                'coords': coords,
                'id': cid
            }
        return self.cluster_data
    
    def super_pixel_fitting(self, cluster_strategy='snr_weighted', estimator='least_squares',
                            model_type='bi-exponential', p0=None, bounds=None):
        """Performs high-SNR super-pixel fitting and triggers comparison plots."""
        super_pixel_data = {}
        super_pixel_params = {}
        master_table_data = []

        if not self.cluster_data:
            print("No cluster data found. Run make_clusters first.")
            return {}, {}

        print(f"\n--- Fitting Super-Pixels (Strategy: {cluster_strategy}) ---")
        
        for c_key, data in tqdm(self.cluster_data.items(), desc="Super-Pixel Progress"):
            decay, irf = data['decay'], data['irf']

            if cluster_strategy == 'sum':
                sp_y, sp_irf = np.sum(decay, axis=0), np.sum(irf, axis=0)
            elif cluster_strategy == 'mean':
                sp_y, sp_irf = np.mean(decay, axis=0), np.mean(irf, axis=0)
            else: # SNR Weighted
                w = np.sum(decay, axis=1) + 1e-9
                w /= np.sum(w)
                sp_y = np.sum(decay * w[:, np.newaxis], axis=0)
                sp_irf = np.sum(irf * w[:, np.newaxis], axis=0)

            super_pixel_data[c_key] = [sp_y, sp_irf]

            # Trigger Super-Pixel Visualization
            self.comparator.compare_selected(
                methods=[estimator], y_data=sp_y, irf_data=sp_irf, 
                model_type=model_type, p0=p0, bounds=bounds, 
                yscale='log', plot=True
            )

            f_class = self.MLEClass if any(m in estimator.lower() for m in ['poisson', 'mle', 'pearson']) else self.BaseClass
            fitter_inst = f_class(self.freq, sp_y, sp_irf)
            
            start_t = time.time()
            res = fitter_inst.fit_with_estimator(
                estimator_type=estimator, model_type=model_type, p0=p0, bounds=bounds
            )
            elapsed = (time.time() - start_t) * 1000

            popt, r2, stat, red_stat = res[0], res[2], res[3], res[4]
            success = "YES" if res[6] == 1 else "NO"
            super_pixel_params[c_key] = popt
            
            p_str = f"A:{popt[0]:.1f}, α:{popt[1]:.2f}, τ1:{popt[2]:.2f}, τ2:{popt[3]:.2f}, B:{popt[4]:.1f}" if model_type == 'bi-exponential' else f"A:{popt[0]:.1f}, τ:{popt[1]:.2f}, B:{popt[2]:.1f}"
            cat = "MLE" if any(m in estimator.lower() for m in ['poisson', 'pearson', 'neyman']) else "NLSF"
            master_table_data.append([c_key, estimator.upper(), cat, success, f"{elapsed:.2f} ms", f"{r2:.4f}", f"{stat:.2f}", f"{red_stat:.4f}", p_str])

        self._print_master_table(master_table_data)
        return super_pixel_data, super_pixel_params

    def process_clusters(self, image_cube, irf_cube, mask=None, gi_tol=0.2, **kwargs):
        """
        global_inference=True: Super-pixel values are seeds (p0), bounds are wide.
        global_inference=False: Super-pixel values are seeds (p0), lifetimes constrained +/- gi_tol.
        """
        H, W, T = image_cube.shape
        results = {}
        model_type = kwargs.get('model_type', 'bi-exponential')
        estimator = kwargs.get('estimator', 'least_squares')
        global_inf = kwargs.get('global_inference', True)
        data_name = kwargs.get('data_name', 'Global_Cluster')

        passed_p0 = kwargs.pop('p0', None)
        passed_bounds = kwargs.pop('bounds', None)

        sp_data, sp_params = self.super_pixel_fitting(
            estimator=estimator, model_type=model_type, 
            p0=passed_p0, bounds=passed_bounds,
            cluster_strategy=kwargs.get('cluster_strategy', 'snr_weighted')
        )

        proc = self.processor() if isinstance(self.processor, type) else self.processor

        for c_key, c_info in self.cluster_data.items():
            cid, coords = c_info['id'], c_info['coords']
            popt_sp = sp_params.get(c_key)
            
            # Failsafe: check if SP fitting actually returned valid parameters
            if popt_sp is None or np.any(np.isnan(popt_sp)): 
                continue

            c_img = np.zeros((H, W, T), dtype=np.float32)
            c_irf = np.zeros((H, W, T), dtype=np.float32)
            c_img[coords[:, 0], coords[:, 1], :] = c_info['decay']
            c_irf[coords[:, 0], coords[:, 1], :] = c_info['irf']

            # Seeding Logic
            local_p0 = None
            local_bounds = None 

            # Build cluster-specific mask: only the pixels that belong to this cluster.
            # Using the original mask here would cause zero-filled non-cluster pixels
            # to be fitted, producing garbage parameters that dominate the result.
            c_mask = np.zeros((H, W), dtype=bool)
            c_mask[coords[:, 0], coords[:, 1]] = True
            if mask is not None:
                c_mask = c_mask & mask

            if model_type == 'bi-exponential':
                # popt_sp = [S, alpha1, tau1, tau2, v_shift, h_shift]
                local_p0 = {
                    'amp':    popt_sp[0],
                    'alpha1': popt_sp[1],
                    'tau1':   popt_sp[2],
                    'tau2':   popt_sp[3],
                    'v_shift': popt_sp[4],
                    'h_shift': popt_sp[5] if len(popt_sp) > 5 else 0.0,
                }
                if not global_inf:
                    local_bounds = {
                        'tau1': [max(popt_sp[2] * (1 - gi_tol), 1e-3),
                                 popt_sp[2] * (1 + gi_tol)],
                        'tau2': [max(popt_sp[3] * (1 - gi_tol), 1e-3),
                                 popt_sp[3] * (1 + gi_tol)],
                    }
            else:
                # popt_sp = [S, tau, v_shift, h_shift]
                local_p0 = {
                    'amp':    popt_sp[0],
                    'tau':    popt_sp[1],
                    'v_shift': popt_sp[2],
                    'h_shift': popt_sp[3] if len(popt_sp) > 3 else 0.0,
                }
                if not global_inf:
                    local_bounds = {
                        'tau': [max(popt_sp[1] * (1 - gi_tol), 1e-3),
                                popt_sp[1] * (1 + gi_tol)],
                    }

            dataset = None
            if hasattr(proc, 'process_image'):
                kwargs['estimator'] = estimator.lower()
                dataset = proc.process_image(image_cube=c_img, irf_cube=c_irf, mask=c_mask, p0=local_p0, bounds=local_bounds, **kwargs)
            elif hasattr(proc, 'fit_image'):
                kwargs['mode'] = estimator.upper()
                dataset = proc.fit_image(image_cube=c_img, irf_cube=c_irf, mask=c_mask, p0=local_p0, bounds=local_bounds, **kwargs)

            if dataset and 'results' in dataset:
                dataset['name'] = f"{data_name}_Cluster_{cid}"
                results[str(cid)] = dataset
                
        return results, sp_data, sp_params

    def stitch_results(self, cluster_results, H, W, T, model_type='bi-exponential'):
        """Combines cluster-wise datasets into global maps with corrected TR naming."""
        _z2 = lambda: np.zeros((H, W), dtype=np.float32)
        stitched_maps = {
            'alpha1_map' if model_type == 'bi-exponential' else 'tau_map': _z2(),
            'tau1_map':            _z2() if model_type == 'bi-exponential' else None,
            'tau2_map':            _z2() if model_type == 'bi-exponential' else None,
            'tau_mean_map':        _z2() if model_type == 'bi-exponential' else None,
            'fret_efficiency_map': _z2() if model_type == 'bi-exponential' else None,
            'photon_count_map':    _z2(),
            'v_shift_map':         _z2(),
            'h_shift_map':         _z2(),
            'chi2_map':            _z2(),
            'R2_map':              _z2(),
            'reduced_chi2_map':    _z2(),
            'convergence_map':     _z2(),
            'pixel_health_map':    _z2(),
        }
        
        stitched_tr = {
            'fit_map': np.zeros((H, W, T), dtype=np.float32),
            'residual_map': np.zeros((H, W, T), dtype=np.float32)
        }

        for cid, dataset in cluster_results.items():
            c_key = f'cluster_{cid}'
            if c_key not in self.cluster_data: continue
            
            coords = self.cluster_data[c_key]['coords']
            r_idx, c_idx = coords[:, 0], coords[:, 1]
            
            res_data = dataset.get('results', {})
            maps = res_data.get('maps', {})
            tr_maps = res_data.get('TR_maps', {})

            # Map parameter values
            for key in stitched_maps.keys():
                if key in maps and stitched_maps[key] is not None:
                    stitched_maps[key][r_idx, c_idx] = maps[key][r_idx, c_idx]
            
            # Map TR data using corrected naming keys
            if 'fit_map' in tr_maps:
                stitched_tr['fit_map'][r_idx, c_idx, :] = tr_maps['fit_map'][r_idx, c_idx, :]
            if 'residual_map' in tr_maps:
                stitched_tr['residual_map'][r_idx, c_idx, :] = tr_maps['residual_map'][r_idx, c_idx, :]

        return {
            'name':   "Global_Stitched_Result",
            'method': 'GlobalFit',
            'results': {
                'maps':    {k: v for k, v in stitched_maps.items() if v is not None},
                'TR_maps': stitched_tr,
            }
        }

    def _print_master_table(self, data):
        headers = ["Cluster", "Method", "Type", "Conv", "Time", "R2", "Chi2", "Red. Chi2", "Parameters"]
        print("\n" + "═"*165 + "\nCONSOLIDATED CLUSTER BENCHMARK\n" + "═"*165)
        print(tabulate(data, headers=headers, tablefmt="fancy_grid", numalign="center", stralign="center"))
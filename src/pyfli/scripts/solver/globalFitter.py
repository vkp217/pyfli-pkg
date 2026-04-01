# solver/globalFitter.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares

class GlobalFLIFitter:
    def __init__(self, freq, fitter_class):
        """
        freq: [freq_laser, freq_acquisition]
        fitter_class: The BaseFLIFitter or MLEFLIFitter class (not an instance)
        """
        self.freq = freq
        self.fitter_class = fitter_class
        self.t = None
        self.model_fit_func = None

    def _fit_super_pixel(self, y_data, irf_p, model_type, estimator, p0, bounds, fit_range=None, **kwargs):
        """Fits the aggregate cluster decay to extract high-SNR shared parameters."""
        fitter = self.fitter_class(self.freq, y_data.astype(np.float32), irf_p)
        
        if fit_range is not None:
            fitter.set_fit_range(*fit_range)
        
        # Clean kwargs for Scipy compatibility
        fit_kwargs = kwargs.copy()
        for key in ['global_inference', 'cluster_strategy', 'min_cluster_size']:
            fit_kwargs.pop(key, None)
        
        res = list(fitter.fit_with_estimator(
            estimator_type=estimator, 
            model_type=model_type, 
            p0=p0, 
            bounds=bounds, 
            **fit_kwargs
        ))
        
        popt, perr = res[0], res[1]

        # Force tau1 < tau2 ordering for bi-exponential consistency
        if model_type == 'bi-exponential' and popt[2] > popt[3]:
            popt[2], popt[3] = popt[3], popt[2]
            popt[1] = 1.0 - popt[1]
            if perr is not None: 
                perr[2], perr[3] = perr[3], perr[2]
            res[0], res[1] = popt, perr

        self.t = fitter.t 
        self.model_fit_func = fitter.model_fit
        
        fit_curve = self.model_fit_func(self.t, popt, model_type=model_type).astype(np.float32)
        residual = (y_data - fit_curve).astype(np.float32)
        
        return res, fit_curve, residual

    def process_clusters(self, image_cube, irf_cube, cluster_mask, estimator='poisson', 
                         model_type='bi-exponential', cluster_strategy='mean', 
                         min_cluster_size=5, p0=None, bounds=None, fit_range=None, **kwargs):
        
        self.global_inference = kwargs.get('global_inference', False)
        H, W, T = image_cube.shape
        param_len = 5 if model_type == 'bi-exponential' else 3
        
        p_maps = np.zeros((H, W, param_len), dtype=np.float32)
        fit_map = np.zeros((H, W, T), dtype=np.float32)
        res_map = np.zeros((H, W, T), dtype=np.float32)
        conv_map = np.zeros((H, W), dtype=np.float32)
        chi2_map = np.zeros((H, W), dtype=np.float32)
        red_chi2_map = np.zeros((H, W), dtype=np.float32)
        pixel_health_map = np.zeros((H, W), dtype=np.float32) # 1=GOOD, 0=PROBLEM
        
        cluster_ids = np.unique(cluster_mask)
        cluster_ids = cluster_ids[cluster_ids != 0]

        for cid in cluster_ids:
            coords = np.argwhere(cluster_mask == cid)
            if len(coords) < min_cluster_size: continue
            
            pixel_group = image_cube[coords[:, 0], coords[:, 1], :]
            irf_group = irf_cube[coords[:, 0], coords[:, 1], :]
            
            if np.sum(pixel_group) == 0: continue

            try:
                super_y = np.mean(pixel_group, axis=0) if cluster_strategy == 'mean' else np.sum(pixel_group, axis=0)
                super_irf = np.mean(irf_group, axis=0) if cluster_strategy == 'mean' else np.sum(irf_group, axis=0)
                
                res, f_super, _ = self._fit_super_pixel(super_y, super_irf, model_type, estimator, p0, bounds, fit_range=fit_range, **kwargs)
                popt_super = res[0]
                
                # Shared lifetimes/fractions
                shared = popt_super[1:4] if model_type == 'bi-exponential' else [popt_super[1]]
                
                # Diagnostic reporting
                p0_sum = p0 if p0 is not None else popt_super
                bnd_sum = bounds if bounds is not None else ([0]*param_len, [np.inf]*param_len)
                self._print_fit_summary(p0_sum, bnd_sum, popt_super, model_type)

                # Determine fit indices
                irf_peak = np.argmax(super_irf)
                indices = np.arange(fit_range[0], min(fit_range[1], T)) if fit_range else np.arange(irf_peak, T)
                is_mle = 'mle' in self.fitter_class.__name__.lower()
                
                for i, (r, c) in enumerate(coords):
                    try:
                        popt, success = self._solve_local(pixel_group[i], super_irf, shared, model_type, 
                                                         indices, estimator, is_mle, bounds, fit_range=fit_range, **kwargs)
                        
                        p_maps[r, c, :] = popt
                        conv_map[r, c] = 1 if success else 0
                        pixel_health_map[r, c] = 1 # Mark as GOOD
                        
                        f_px = self.model_fit_func(self.t, popt, model_type=model_type).astype(np.float32)
                        fit_map[r, c, :] = f_px
                        res_px = (pixel_group[i] - f_px)
                        res_map[r, c, :] = res_px
                        
                        c2 = np.sum((res_px[indices]**2) / np.clip(pixel_group[i, indices], 1, None))
                        chi2_map[r, c] = c2
                        red_chi2_map[r, c] = c2 / (len(indices) - param_len)
                    except Exception:
                        pixel_health_map[r, c] = 0 # Mark as PROBLEM
                        continue

            except Exception as e:
                print(f"Critical error fitting Cluster {cid}: {e}")
                pixel_health_map[coords[:, 0], coords[:, 1]] = 0
                continue
            
        return {
            'results': {
                'maps': self._format_maps(p_maps, model_type, conv_map, chi2_map, red_chi2_map, pixel_health_map), 
                'TR_maps': {'fit_map': fit_map, 'residual_map': res_map}
            }
        }

    def _solve_local(self, pixel_data, irf, shared, model_type, indices, estimator, is_mle, bounds=None, fit_range=None, **kwargs):
        s_g, b_g = np.max(pixel_data), np.mean(pixel_data[:5])
        temp_fitter = self.fitter_class(self.freq, pixel_data, irf)
        
        if fit_range is not None:
            temp_fitter.set_fit_range(*fit_range)
        
        local_kwargs = kwargs.copy()
        for key in ['global_inference', 'cluster_strategy', 'min_cluster_size']:
            local_kwargs.pop(key, None)
        
        max_iterations = local_kwargs.pop('maxiter', 500)

        if model_type == 'bi-exponential':
            def map_params(params): return [params[0], shared[0], shared[1], shared[2], params[1]]
        else:
            def map_params(params): return [params[0], shared[0], params[1]]

        if self.global_inference:
            p0_full = [s_g, shared[0], shared[1], shared[2], b_g] if model_type == 'bi-exponential' else [s_g, shared[0], b_g]
            res = temp_fitter.fit_with_estimator(estimator_type=estimator, model_type=model_type, p0=p0_full, 
                                                bounds=bounds, maxiter=max_iterations, **local_kwargs)           
            return res[0], (res[6] == 1)
        
        else:
            if is_mle:
                def objective(params): return temp_fitter.poisson_log_likelihood(map_params(params), model_type)
                res = minimize(objective, x0=[s_g, b_g], bounds=[(0, None), (0, s_g)], 
                               method='L-BFGS-B', options={'maxiter': max_iterations})
                popt, success = map_params(res.x), res.success
            else:              
                def residuals(params):
                    full_p = map_params(params)
                    fit_curve = temp_fitter.model_fit(temp_fitter.t, full_p, model_type=model_type)
                    return (pixel_data[indices] - fit_curve[indices]).astype(np.float64)

                res = least_squares(residuals, x0=[s_g, b_g], bounds=([0, 0], [np.inf, s_g]), max_nfev=max_iterations)
                popt, success = map_params(res.x), res.status > 0
                
            return popt, success

    def _print_fit_summary(self, p0, bounds, popt, model_type):
        names = ['Amp', 'alpha1', 'tau1', 'tau2', 'BG'] if model_type == 'bi-exponential' else ['Amp', 'tau', 'BG']
        print(f"\nCluster shared parameters extracted:")
        print(f"{'Param':<8} | {'P0 Guess':<10} | {'Lower B':<10} | {'Upper B':<10} | {'Final Fit':<10}")
        print("-" * 65)
        for i, name in enumerate(names):
            if i < len(popt):
                print(f"{name:<8} | {p0[i]:<10.4f} | {bounds[0][i]:<10.4f} | {bounds[1][i]:<10.4f} | {popt[i]:<10.4f}")

    def _format_maps(self, p_maps, model_type, conv_map, chi2_map, red_chi2_map, health_map):
        S = p_maps[..., 0]
        res = {
            'Area_map': S, 
            'offset_map': p_maps[..., -1], # Renamed for library compatibility
            'convergence_map': conv_map, 
            'pixel_health_map': health_map, # New foolproof health indicator
            'Chi2_map': chi2_map, 
            'Red_Chi2_map': red_chi2_map
        }
        if model_type == 'bi-exponential':
            res.update({
                'alpha1_map': p_maps[..., 1], 
                'tau1_map': p_maps[..., 2], 
                'tau2_map': p_maps[..., 3],
                'alpha2_map': 1.0 - p_maps[..., 1],
                'Int_A1_map': S * p_maps[..., 1],
                'Int_A2_map': S * (1.0 - p_maps[..., 1])
            })
        else:
            res.update({'tau_map': p_maps[..., 1]})
        return res
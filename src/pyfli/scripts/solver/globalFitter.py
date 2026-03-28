import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, least_squares


class GlobalFLIFitter:
    def __init__(self, freq, fitter_class):
        """
        freq: [freq_laser, freq_acquisition]
        fitter_class: The BaseFLIFitter, or MLEFLIFitter class (not an instance)
        """
        self.freq = freq
        self.fitter_class = fitter_class

    def _fit_super_pixel(self, y_data, irf_p, model_type, estimator, p0, bounds, fit_range=None, **kwargs):
        """Fits the aggregate cluster decay to extract high-SNR shared parameters."""
        fitter = self.fitter_class(self.freq, y_data.astype(np.float32), irf_p)
        
        # --- ADDED: Set fit range if provided ---
        if fit_range is not None:
            fitter.set_fit_range(*fit_range)
        
        # --- FIX 1: Clean kwargs for Scipy compatibility ---
        fit_kwargs = kwargs.copy()
        for key in ['global_inference', 'cluster_strategy', 'min_cluster_size']:
            fit_kwargs.pop(key, None)
        
        res = list(fitter.fit_with_estimator(
            estimator_type=estimator, 
            model_type=model_type, 
            p0=p0, 
            bounds=bounds, 
            **fit_kwargs # Use cleaned version
        ))
        
        popt, perr = res[0], res[1]

        if model_type == 'bi-exponential' and popt[2] > popt[3]:
            avg_pre = popt[1] * popt[2] + (1.0 - popt[1]) * popt[3]
            popt[2], popt[3] = popt[3], popt[2]
            popt[1] = 1.0 - popt[1]
            if perr is not None: perr[2], perr[3] = perr[3], perr[2]
            
            avg_post = popt[1] * popt[2] + (1.0 - popt[1]) * popt[3]
            if not np.isclose(avg_pre, avg_post, atol=1e-9):
                print("Warning: Super-pixel swap failed sanity check.")
            res[0], res[1] = popt, perr

        self.t = fitter.t 
        fit_curve = fitter.model_fit(self.t, popt, model_type=model_type).astype(np.float32)
        residual = (y_data - fit_curve).astype(np.float32)
        self.model_fit = fitter.model_fit 
        
        return res, fit_curve, residual

    def process_clusters(self, image_cube, irf_cube, cluster_mask, estimator='poisson', 
                         model_type='bi-exponential', cluster_strategy='mean', 
                         min_cluster_size=5, p0=None, bounds=None, fit_range=None, **kwargs):
        
        # Store global_inference state for the local loop
        self.global_inference = kwargs.get('global_inference', False)
        
        H, W, T = image_cube.shape
        param_len = 5 if model_type == 'bi-exponential' else 3
        
        p_maps = np.zeros((H, W, param_len), dtype=np.float32)
        fit_map = np.zeros((H, W, T), dtype=np.float32)
        res_map = np.zeros((H, W, T), dtype=np.float32)
        conv_map = np.zeros((H, W), dtype=np.float32)
        chi2_map = np.zeros((H, W), dtype=np.float32)
        red_chi2_map = np.zeros((H, W), dtype=np.float32)
        
        cluster_ids = np.unique(cluster_mask)[np.unique(cluster_mask) != 0]

        for cid in cluster_ids:
            coords = np.argwhere(cluster_mask == cid)
            pixel_group = image_cube[coords[:, 0], coords[:, 1], :]
            irf_group = irf_cube[coords[:, 0], coords[:, 1], :]
            
            if np.sum(pixel_group) == 0: continue

            super_y = np.mean(pixel_group, axis=0) if cluster_strategy == 'mean' else np.sum(pixel_group, axis=0)
            super_irf = np.mean(irf_group, axis=0) if cluster_strategy == 'mean' else np.sum(irf_group, axis=0)
            
            # Pass fit_range to super pixel fit
            res, f_super, _ = self._fit_super_pixel(super_y, super_irf, model_type, estimator, p0, bounds, fit_range=fit_range, **kwargs)
            popt_super = res[0]
            shared = popt_super[1:4] if model_type == 'bi-exponential' else [popt_super[1]]
            
            p0_sum = p0 if p0 is not None else popt_super
            bnd_sum = bounds if bounds is not None else ([0]*param_len, [np.inf]*param_len)
            self._print_fit_summary(p0_sum, bnd_sum, popt_super, model_type)
            self._plot_diagnostics(f"Cluster {cid}", estimator, super_y, f_super, (res[4] if len(res) > 4 else 0))

            irf_peak = np.argmax(super_irf)
            indices = np.arange(irf_peak, T)
            is_mle = 'mle' in self.fitter_class.__name__.lower()
            
            for i, (r, c) in enumerate(coords):
                # Pass fit_range to local solve
                popt, success = self._solve_local(pixel_group[i], super_irf, shared, model_type, 
                                                  indices, estimator, is_mle, bounds, fit_range=fit_range, **kwargs)
                
                p_maps[r, c, :] = popt
                conv_map[r, c] = 1 if success else 0
                f_px = self.model_fit(self.t, popt, model_type=model_type).astype(np.float32)
                fit_map[r, c, :] = f_px
                res_px = (pixel_group[i] - f_px)
                res_map[r, c, :] = res_px
                
                c2 = np.sum((res_px[indices]**2) / np.clip(pixel_group[i, indices], 1, None))
                chi2_map[r, c] = c2
                red_chi2_map[r, c] = c2 / (len(indices) - param_len)
            
        return {
            'results': {
                'maps': self._format_maps(p_maps, model_type, conv_map, chi2_map, red_chi2_map), 
                'TR_maps': {'fit_map': fit_map, 'residual_map': res_map}
            }
        }

    def _solve_local(self, pixel_data, irf, shared, model_type, indices, estimator, is_mle, bounds=None, fit_range=None, **kwargs):
        """
        Merged local solver: Strictly fixes lifetimes and handles 
        optimizer-specific keyword arguments.
        """
        s_g, b_g = np.max(pixel_data), np.mean(pixel_data[:5])
        temp_fitter = self.fitter_class(self.freq, pixel_data, irf)
        
        # --- ADDED: Set fit range for local temp_fitter ---
        if fit_range is not None:
            temp_fitter.set_fit_range(*fit_range)
        
        # --- CLEANUP KWARGS ---
        local_kwargs = kwargs.copy()
        local_kwargs.pop('global_inference', None)
        local_kwargs.pop('cluster_strategy', None)
        local_kwargs.pop('min_cluster_size', None)        

        max_iterations = local_kwargs.pop('maxiter', 500)

        if model_type == 'bi-exponential':
            def map_params(params):
                return [params[0], shared[0], shared[1], shared[2], params[1]]
        else:
            def map_params(params):
                return [params[0], shared[0], params[1]]

        if self.global_inference:
            p0_full = list(shared) + [b_g]
            # Add maxiter back for fit_with_estimator which expects it
            local_kwargs['maxiter'] = max_iterations 
            res = temp_fitter.fit_with_estimator(
                estimator_type=estimator, 
                model_type=model_type, 
                p0=p0_full, 
                bounds=bounds,
                **local_kwargs
            )           
            return res[0], (res[6] == 1)
        
        else:
            # MODE B: Strictly Fixed
            if is_mle:
                def objective(params):
                    return temp_fitter.poisson_log_likelihood(map_params(params), model_type)
                res = minimize(objective, x0=[s_g, b_g], bounds=[(0, None), (0, s_g)], 
                               method='L-BFGS-B', options={'maxiter': max_iterations})
                popt = map_params(res.x)
                success = res.success
            else:              
                def residuals(params):
                    full_p = map_params(params)
                    fit_curve = temp_fitter.model_fit(temp_fitter.t, full_p, model_type=model_type)
                    return (pixel_data[indices] - fit_curve[indices]).astype(np.float64)

                res = least_squares(residuals, x0=[s_g, b_g], bounds=([0, 0], [np.inf, s_g]), 
                                    max_nfev=max_iterations, **local_kwargs)
                popt = map_params(res.x)
                success = res.status > 0
                
            return popt, success

    def _print_fit_summary(self, p0, bounds, popt, model_type):
        names = ['Amp', 'alpha1', 'tau1', 'tau2', 'BG'] if model_type == 'bi-exponential' else ['Amp', 'tau', 'BG']
        print(f"\n{'Param':<8} | {'P0 Guess':<10} | {'Lower B':<10} | {'Upper B':<10} | {'Final Fit':<10}")
        print("-" * 65)
        for i, name in enumerate(names):
            if i < len(popt):
                print(f"{name:<8} | {p0[i]:<10.4f} | {bounds[0][i]:<10.4f} | {bounds[1][i]:<10.4f} | {popt[i]:<10.4f}")

    def _plot_diagnostics(self, title, est, decay, fit, stat):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))
        for i, scale in enumerate(['log', 'linear']):
            if scale == 'log':
                ax[i].set_yscale('log')
            ax[i].plot(self.t, decay, 'k.', alpha=0.3)
            ax[i].plot(self.t, fit, 'r-')
            ax[i].plot(self.t, decay - fit, 'g-', alpha=0.3)
            ax[i].set_title(f"{scale.capitalize()}: {title} ({stat:.2f})")
        plt.tight_layout()
        plt.show()

    def _format_maps(self, p_maps, model_type, conv_map, chi2_map, red_chi2_map):
        S = p_maps[..., 0]
        res = {'Area_map': S, 'B_map': p_maps[..., -1], 'convergence_map': conv_map, 
               'chi2_map': chi2_map, 'red_chi2_map': red_chi2_map}
        if model_type == 'bi-exponential':
            res.update({'alpha1_map': p_maps[..., 1], 'tau1_map': p_maps[..., 2], 'tau2_map': p_maps[..., 3]})
        else:
            res.update({'tau_map': p_maps[..., 1]})
        return res
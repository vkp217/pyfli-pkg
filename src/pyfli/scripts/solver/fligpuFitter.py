# solver/fligpuFitter.py
import torch
import numpy as np
import h5py
import os
import time
from tqdm import tqdm
from .base_static import resolve_params_and_bounds, moment_based_guess

class Fli_GPUProcessor:
    def __init__(self, freq, fitter_class=None, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.freq = freq
        self.fitter_class = fitter_class
        self.T_acq = 1000.0 / freq[1]
        self.T_laser = 1000.0 / freq[0]
        print(f"Using Device: {self.device}")

    def _transform_params(self, raw_p, model_type):
        """Transform unconstrained optimizer values to physical parameters."""
        S = torch.exp(raw_p[:, 0:1]) 
        b = torch.exp(raw_p[:, -1:]) 
        
        if model_type == 'bi-exponential':
            a1 = torch.sigmoid(raw_p[:, 1:2])
            t1 = torch.exp(raw_p[:, 2:3])
            # Enforce tau1 <= tau2 hierarchy via positive offset
            t2 = t1 + torch.exp(raw_p[:, 3:4]) 
            return torch.cat([S, a1, t1, t2, b], dim=1)
        else:
            tau = torch.exp(raw_p[:, 1:2])
            return torch.cat([S, tau, b], dim=1)

    def _model_kernel(self, params, t, irf, model_type):
        """Vectorized Decay Model with FFT Convolution."""
        if model_type == 'mono-exponential':
            S, tau, b = params[:, 0:1], params[:, 1:2], params[:, 2:3]
            decay = (S / tau) * torch.exp(-t / tau)
        else:
            S, a1, t1, t2, b = params[:, 0:1], params[:, 1:2], params[:, 2:3], params[:, 3:4], params[:, 4:5]
            decay = S * ((a1 / t1) * torch.exp(-t / t1) + ((1.0 - a1) / t2) * torch.exp(-t / t2))
        
        T = t.shape[-1]
        n_fft = 2 * T
        decay_fft = torch.fft.rfft(decay, n=n_fft)
        irf_fft = torch.fft.rfft(irf, n=n_fft)
        convolved = torch.fft.irfft(decay_fft * irf_fft, n=n_fft)[..., :T]
        return convolved + b

    def _compute_crlb_errors(self, p_phys, t, irf, model_type):
        """Vectorized Jacobian-based CRLB error estimation."""
        p_phys = p_phys.detach().clone().requires_grad_(True)
        def model_func(p): return self._model_kernel(p, t, irf, model_type)
        
        # Compute Jacobian via autograd
        jac = torch.autograd.functional.jacobian(model_func, p_phys, vectorize=True)
        # Reshape: (batch, time, params)
        jac = torch.diagonal(jac, dim1=0, dim2=2).permute(2, 0, 1)

        with torch.no_grad():
            pred = model_func(p_phys)
            W = 1.0 / torch.clamp(pred, min=1.0)
            jt_w = jac.transpose(1, 2) * W.unsqueeze(1)
            fim = torch.bmm(jt_w, jac)
            eps = 1e-10 * torch.eye(fim.shape[-1], device=self.device)
            try:
                cov = torch.inverse(fim + eps)
                return torch.sqrt(torch.abs(torch.diagonal(cov, dim1=1, dim2=2)))
            except RuntimeError:
                return torch.zeros_like(p_phys)

    def fit_image(self, image_cube, irf_cube, mask=None, mode='MLE', 
                  model_type='bi-exponential', max_iter=150, CRLB=False, 
                  data_name="Torch_Fit", **kwargs):
        start_time = time.time()
        H, W, T = image_cube.shape
        t_axis = torch.linspace(0, self.T_acq, T, device=self.device)
        
        irf_tensor = torch.tensor(irf_cube, device=self.device, dtype=torch.float32).reshape(-1, T)
        irf_norm = irf_tensor / torch.clamp(irf_tensor.sum(dim=1, keepdim=True), 1e-9)
        
        if mask is None:
            mask = np.sum(image_cube, axis=2) > 20
        
        valid_idx = np.where(mask.flatten())[0]
        if len(valid_idx) == 0:
            print("No valid pixels found.")
            return None

        flat_data = torch.tensor(image_cube.reshape(-1, T)[valid_idx], device=self.device, dtype=torch.float32)
        flat_irf = irf_norm[valid_idx]
        
        # Initial Guessing
        p_guess = self._get_sophisticated_guess(flat_data, flat_irf, model_type)
        raw_p = torch.zeros_like(p_guess)
        with torch.no_grad():
            raw_p[:, 0] = torch.log(torch.clamp(p_guess[:, 0], min=1e-3))
            raw_p[:, -1] = torch.log(torch.clamp(p_guess[:, -1], min=1e-6))
            if model_type == 'bi-exponential':
                raw_p[:, 1] = torch.logit(torch.clamp(p_guess[:, 1], 0.01, 0.99))
                raw_p[:, 2] = torch.log(torch.clamp(p_guess[:, 2], min=0.1))
                raw_p[:, 3] = torch.log(torch.clamp(p_guess[:, 3] - p_guess[:, 2], min=0.1))
            else:
                raw_p[:, 1] = torch.log(torch.clamp(p_guess[:, 1], min=0.1))

        raw_p.requires_grad_(True)
        pixel_health_map = np.ones(H * W, dtype=np.float32)

        # Per-pixel photon count used to equalize gradient magnitudes across
        # pixels with very different brightness levels (bright pixels otherwise
        # dominate the shared LBFGS history and give poor fits to dim pixels).
        n_photons = flat_data.sum(dim=1, keepdim=True).clamp(min=1.0)

        # Objective Functions
        if mode == 'LSE':
            weights = 1.0 / torch.sqrt(torch.clamp(flat_data, min=1.0))
            def objective_fn(p_raw):
                p_phys = self._transform_params(p_raw, model_type)
                pred = self._model_kernel(p_phys, t_axis, flat_irf, model_type)
                per_px = torch.sum(((pred - flat_data) * weights)**2, dim=1, keepdim=True)
                return (per_px / n_photons).sum()
        else:
            def objective_fn(p_raw):
                p_phys = self._transform_params(p_raw, model_type)
                pred = self._model_kernel(p_phys, t_axis, flat_irf, model_type)
                pred = torch.clamp(pred, min=1e-9)
                per_px = 2 * torch.sum(
                    pred - flat_data + flat_data * torch.log(torch.clamp(flat_data, 1e-9) / pred),
                    dim=1, keepdim=True)
                return (per_px / n_photons).sum()

        optimizer = torch.optim.LBFGS([raw_p], lr=1, max_iter=max_iter, history_size=10, line_search_fn="strong_wolfe")

        print(f"--- GPU {mode} Processing ({len(valid_idx)} pixels) ---")
        pbar = tqdm(total=max_iter, desc=f"Optimizing {mode}")
        
        def closure():
            optimizer.zero_grad()
            loss = objective_fn(raw_p)
            if torch.isnan(loss):
                return torch.tensor(0.0, device=self.device, requires_grad=True)
            loss.backward()
            pbar.update(1)
            return loss

        try:
            optimizer.step(closure)
            pbar.update(max_iter - pbar.n)
        except Exception as e:
            print(f"Optimization interrupted: {e}")
            pixel_health_map[valid_idx] = 0
            
        pbar.close()

        with torch.no_grad():
            p_final = self._transform_params(raw_p, model_type)
            fit_flat = self._model_kernel(p_final, t_axis, flat_irf, model_type)
            res_flat = flat_data - fit_flat
            dof = max(T - p_final.shape[1], 1)

            # Raw chi2 (consistent with CPU fitter's stat_map)
            chi2_raw_flat = torch.sum((res_flat**2) / torch.clamp(flat_data, 1.0), dim=1)
            chi2_red_flat = chi2_raw_flat / dof

            # R² per pixel
            ss_tot = torch.sum((flat_data - flat_data.mean(dim=1, keepdim=True))**2, dim=1)
            ss_res = torch.sum(res_flat**2, dim=1)
            r2_flat = 1.0 - ss_res / torch.clamp(ss_tot, min=1e-9)

            perr_flat = torch.zeros_like(p_final)
            if CRLB:
                perr_flat = self._compute_crlb_errors(p_final, t_axis, flat_irf, model_type)

        # Mapping back to image dimensions
        full_popt = np.zeros((H * W, p_final.shape[1]))
        full_perr = np.zeros((H * W, p_final.shape[1]))
        full_fit  = np.zeros((H * W, T))
        full_res  = np.zeros((H * W, T))
        full_chi2_raw = np.zeros(H * W)
        full_chi2_red = np.zeros(H * W)
        full_r2       = np.zeros(H * W)

        full_popt[valid_idx]     = p_final.detach().cpu().numpy()
        full_perr[valid_idx]     = perr_flat.detach().cpu().numpy()
        full_fit[valid_idx]      = fit_flat.detach().cpu().numpy()
        full_res[valid_idx]      = res_flat.detach().cpu().numpy()
        full_chi2_raw[valid_idx] = chi2_raw_flat.detach().cpu().numpy()
        full_chi2_red[valid_idx] = chi2_red_flat.detach().cpu().numpy()
        full_r2[valid_idx]       = r2_flat.detach().cpu().numpy()

        print(f"Fit Finished in {time.time() - start_time:.2f}s")

        # Per-pixel health: flag pixels where reduced chi2 > 5 or a lifetime
        # parameter has converged to a bound (both indicate a poor fit).
        health_mask = np.zeros(H * W)
        health_mask[valid_idx] = 1.0
        tau_lo, tau_hi = 1e-4, self.T_laser
        p_np = p_final.detach().cpu().numpy()
        chi2_red_np = full_chi2_red[valid_idx]
        if model_type == 'bi-exponential':
            at_bound = (
                (p_np[:, 2] <= tau_lo * 1.01) | (p_np[:, 2] >= tau_hi * 0.99) |
                (p_np[:, 3] <= tau_lo * 1.01) | (p_np[:, 3] >= tau_hi * 0.99)
            )
        else:
            at_bound = (p_np[:, 1] <= tau_lo * 1.01) | (p_np[:, 1] >= tau_hi * 0.99)
        health_mask[valid_idx] = np.where(at_bound | (chi2_red_np > 5.0), 0.0, 1.0)

        return self._reconstruct_dataset(
            full_popt.reshape(H, W, -1), full_perr.reshape(H, W, -1),
            full_fit.reshape(H, W, T), full_res.reshape(H, W, T),
            full_chi2_raw.reshape(H, W), full_chi2_red.reshape(H, W),
            full_r2.reshape(H, W), health_mask.reshape(H, W),
            model_type, mode, data_name
        )

    def _reconstruct_dataset(self, p_maps, e_maps, fit_map, res_map,
                             chi2_raw, chi2_reduced, r2_map, health_map,
                             model_type, mode, name):
        """Standardized Dictionary Reconstruction compatible with Fli_CPUProcessor."""
        S = p_maps[..., 0]
        common = {
            'chi2_map':        chi2_raw,
            'reduced_stat_map': chi2_reduced,
            'R2_map':          r2_map,
            'pixel_health_map': health_map,
        }
        if model_type == 'bi-exponential':
            maps = {
                'Area_map': S, 'alpha1_map': p_maps[..., 1],
                'tau1_map': p_maps[..., 2], 'tau2_map': p_maps[..., 3],
                'offset_map': p_maps[..., 4],
                **common,
            }
        else:
            maps = {
                'Area_map': S, 'tau_map': p_maps[..., 1],
                'offset_map': p_maps[..., 2],
                **common,
            }
        return {
            'name': name,
            'method': f"GPU_{mode}",
            'results': {
                'maps': maps,
                'error_maps': e_maps,
                'TR_maps': {
                    'fit_map':      fit_map.astype(np.float32),
                    'residual_map': res_map.astype(np.float32),
                }
            }
        }

    def _get_sophisticated_guess(self, data, irf, model_type):
        guesses = []
        cpu_data = data.detach().cpu().numpy()
        T = cpu_data.shape[-1]
        t_axis = np.linspace(0, self.T_acq, T)

        for i in range(cpu_data.shape[0]):
            current_decay = cpu_data[i]
        
            p0_safe, _ = resolve_params_and_bounds(
                user_p0=None, 
                user_bounds=None, 
                model_type=model_type, 
                t=t_axis, 
                decay=current_decay, 
                T_laser=self.T_laser, 
                guess_plugin=moment_based_guess, 
                T_acq=self.T_acq
            )
            
            guesses.append(p0_safe)
        return torch.tensor(np.array(guesses), device=self.device, dtype=torch.float32)

    def save_results(self, dataset, folder="results"):
        """Saves with standardized maps/TR_maps structure."""
        if not os.path.exists(folder): os.makedirs(folder)
        h5_path = os.path.join(folder, f"{dataset['name']}_GPU_results.h5")
        
        with h5py.File(h5_path, "w") as f:
            f.attrs['method'] = dataset['method']
            res_grp = f.create_group("results")
            
            maps_grp = res_grp.create_group("maps")
            for k, v in dataset['results']['maps'].items():
                maps_grp.create_dataset(k, data=v.astype(np.float32), compression="gzip")
            
            err_grp = res_grp.create_group("error_maps")
            err_grp.create_dataset("errors", data=dataset['results']['error_maps'], compression="gzip")
            
            tr_grp = res_grp.create_group("TR_maps")
            for k, v in dataset['results']['TR_maps'].items():
                tr_grp.create_dataset(k, data=v, compression="gzip")
                
        print(f"Dataset successfully saved to: {h5_path}")
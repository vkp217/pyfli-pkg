# solver/fligpuFitter.py
import torch
import numpy as np
import h5py
import os
import time
from tqdm import tqdm

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
            t2 = t1 + torch.exp(raw_p[:, 3:4]) # Enforce tau1 <= tau2
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

    def _compute_crlb_errors(self, p_phys, t, irf, model_type, mode):
        """Vectorized Jacobian-based CRLB error estimation."""
        p_phys = p_phys.detach().clone().requires_grad_(True)
        def model_func(p): return self._model_kernel(p, t, irf, model_type)
        
        jac = torch.autograd.functional.jacobian(model_func, p_phys, vectorize=True)
        jac = torch.diagonal(jac, dim1=0, dim2=2).permute(2, 0, 1)

        with torch.no_grad():
            pred = model_func(p_phys)
            # W matches the weighting logic of the specific mode
            W = 1.0 / torch.clamp(pred, min=1.0)
            jt_w = jac.transpose(1, 2) * W.unsqueeze(1)
            fim = torch.bmm(jt_w, jac)
            eps = 1e-10 * torch.eye(fim.shape[-1], device=self.device)
            cov = torch.inverse(fim + eps)
            return torch.sqrt(torch.abs(torch.diagonal(cov, dim1=1, dim2=2)))

    def fit_image(self, image_cube, irf_cube, mask=None, mode='MLE', 
                  model_type='bi-exponential', max_iter=150, CRLB=False):
        start_time = time.time()
        H, W, T = image_cube.shape
        t_axis = torch.linspace(0, self.T_acq, T, device=self.device)
        
        irf_tensor = torch.tensor(irf_cube, device=self.device, dtype=torch.float32).reshape(-1, T)
        irf_norm = irf_tensor / torch.clamp(irf_tensor.sum(dim=1, keepdim=True), 1e-9)
        
        if mask is None:
            mask = np.sum(image_cube, axis=2) > 20
        
        valid_idx = np.where(mask.flatten())[0]
        flat_data = torch.tensor(image_cube.reshape(-1, T)[valid_idx], device=self.device, dtype=torch.float32)
        flat_irf = irf_norm[valid_idx]
        
        # Initial Guess Setup
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
        
        # Define Mode-Specific Objective Functions
        if mode == 'LSE':
            # Logic: Minimize sum( ((model - data) * weights)^2 )
            # weights = 1/sqrt(data)
            weights = 1.0 / torch.sqrt(torch.clamp(flat_data, min=1.0))
            def objective_fn(p_raw):
                p_phys = self._transform_params(p_raw, model_type)
                pred = self._model_kernel(p_phys, t_axis, flat_irf, model_type)
                return torch.sum(((pred - flat_data) * weights)**2)
        else:
            # Standard Poisson MLE
            def objective_fn(p_raw):
                p_phys = self._transform_params(p_raw, model_type)
                pred = self._model_kernel(p_phys, t_axis, flat_irf, model_type)
                pred = torch.clamp(pred, min=1e-9)
                return 2 * torch.sum(pred - flat_data + flat_data * torch.log(torch.clamp(flat_data, 1e-9) / pred))

        optimizer = torch.optim.LBFGS([raw_p], lr=1, max_iter=max_iter, history_size=10, line_search_fn="strong_wolfe")

        print(f"--- GPU {mode} Processing ({len(valid_idx)} pixels) ---")
        pbar = tqdm(total=max_iter, desc=f"Optimizing {mode}")
        
        def closure():
            optimizer.zero_grad()
            loss = objective_fn(raw_p)
            loss.backward()
            pbar.update(1)
            return loss

        optimizer.step(closure)
        pbar.close()

        with torch.no_grad():
            p_final = self._transform_params(raw_p, model_type)
            fit_flat = self._model_kernel(p_final, t_axis, flat_irf, model_type)
            res_flat = flat_data - fit_flat
            dof = T - p_final.shape[1]
            chi2_flat = torch.sum((res_flat**2) / torch.clamp(flat_data, 1.0), dim=1) / dof
            
            perr_flat = torch.zeros_like(p_final)
            if CRLB:
                perr_flat = self._compute_crlb_errors(p_final, t_axis, flat_irf, model_type, mode)

        # Mapping back to image dimensions
        full_popt = np.zeros((H * W, p_final.shape[1]))
        full_perr = np.zeros((H * W, p_final.shape[1]))
        full_fit = np.zeros((H * W, T))
        full_res = np.zeros((H * W, T))
        full_chi2 = np.zeros(H * W)

        full_popt[valid_idx] = p_final.cpu().numpy()
        full_perr[valid_idx] = perr_flat.cpu().numpy()
        full_fit[valid_idx] = fit_flat.cpu().numpy()
        full_res[valid_idx] = res_flat.cpu().numpy()
        full_chi2[valid_idx] = chi2_flat.cpu().numpy()

        print(f"Fit Finished in {time.time() - start_time:.2f}s")
        return self._reconstruct_dataset(
            full_popt.reshape(H, W, -1), full_perr.reshape(H, W, -1), 
            full_fit.reshape(H, W, T), full_res.reshape(H, W, T),
            full_chi2.reshape(H, W), model_type, mode
        )

    def _reconstruct_dataset(self, p_maps, e_maps, fit_cube, res_cube, chi2_map, model_type, mode):
        """Restored exact dictionary structure from BaseFLIFitter."""
        S = p_maps[..., 0]
        if model_type == 'bi-exponential':
            a1 = p_maps[..., 1]
            maps = {
                'Area_map': S, 'Area_err': e_maps[..., 0],
                'alpha1_map': a1, 'alpha1_err': e_maps[..., 1],
                'tau1_map': p_maps[..., 2], 'tau1_err': e_maps[..., 2],
                'tau2_map': p_maps[..., 3], 'tau2_err': e_maps[..., 3],
                'Bkg_map': p_maps[..., 4], 'Bkg_err': e_maps[..., 4],
                'alpha2_map': 1.0 - a1,
                'Int_A1_map': S * a1, 'Int_A2_map': S * (1.0 - a1),
                'Chi2_map': chi2_map
            }
        else:
            maps = {
                'Area_map': S, 'Area_err': e_maps[..., 0],
                'tau_map': p_maps[..., 1], 'tau_err': e_maps[..., 1],
                'Bkg_map': p_maps[..., 2], 'Bkg_err': e_maps[..., 2],
                'Chi2_map': chi2_map
            }
        return {
            'method': f"GPU_{mode}", 
            'results': {
                'maps': maps,
                'fit_cube': fit_cube,
                'residual_cube': res_cube
            }
        }

    def _get_sophisticated_guess(self, data, irf, model_type):
        guesses = []
        for i in range(data.shape[0]):
            f = self.fitter_class(self.freq, data[i].cpu().numpy(), irf[i].cpu().numpy())
            guesses.append(f.initial_guess(model_type))
        return torch.tensor(np.array(guesses), device=self.device, dtype=torch.float32)

    def save_results(self, dataset, folder="results", data_name="Torch_Fit"):
        if not os.path.exists(folder): os.makedirs(folder)
        h5_path = os.path.join(folder, f"{data_name}_results.h5")
        with h5py.File(h5_path, "w") as f:
            res_grp = f.create_group("results")
            maps_grp = res_grp.create_group("maps")
            for k, v in dataset['results']['maps'].items():
                maps_grp.create_dataset(k, data=v.astype(np.float32), compression="gzip")
            res_grp.create_dataset("fit_cube", data=dataset['results']['fit_cube'], compression="gzip")
            res_grp.create_dataset("residual_cube", data=dataset['results']['residual_cube'], compression="gzip")
        print(f"Saved: {h5_path}")

# import jax
# import jax.numpy as jnp
# from jax import jit, vmap, grad, hessian
# import numpy as np
# import h5py
# import os
# import time
# from functools import partial

# class Fli_GPUProcessor:
#     def __init__(self, freq, fitter_class=None):
#         self.freq = freq
#         self.fitter_class = fitter_class
#         self.T_acq = 1000.0 / freq[1]
#         self.T_laser = 1000.0 / freq[0]

#     @partial(jit, static_argnums=(0, 4))
#     def _model_kernel(self, params, t, irf, model_type):
#         """Standard FLIM decay model logic."""
#         if model_type == 'mono-exponential':
#             S, tau, b = params
#             decay = (S / tau) * jnp.exp(-t / tau)
#         else:
#             S, a1, t1, t2, b = params
#             decay = S * ((a1 / t1) * jnp.exp(-t / t1) + ((1 - a1) / t2) * jnp.exp(-t / t2))
        
#         T = t.shape[0]
#         decay_fft = jnp.fft.rfft(decay, n=2*T)
#         irf_fft = jnp.fft.rfft(irf, n=2*T)
#         convolved = jnp.fft.irfft(decay_fft * irf_fft, n=2*T)[:T]
#         return convolved + b

#     def _loss_factory(self, mode='MLE'):
#         """Supports Poisson MLE or Weighted Least Squares."""
#         def lse_loss(params, t, irf, data, model_type):
#             pred = self._model_kernel(params, t, irf, model_type)
#             weights = jnp.clip(data, 1.0) 
#             return jnp.sum(((data - pred)**2) / weights)

#         def mle_loss(params, t, irf, data, model_type):
#             pred = self._model_kernel(params, t, irf, model_type)
#             pred = jnp.clip(pred, 1e-9)
#             return 2 * jnp.sum(pred - data + data * jnp.log(jnp.clip(data, 1e-9) / pred))

#         return mle_loss if mode == 'MLE' else lse_loss

#     def _get_sophisticated_guess(self, data_array, irf_array, model_type):
#         """Pulls physics-based initial guesses from BaseFLIFitter."""
#         guesses = []
#         for i in range(data_array.shape[0]):
#             temp_fitter = self.fitter_class(self.freq, np.array(data_array[i]), np.array(irf_array[i]))
#             guesses.append(temp_fitter.initial_guess(model_type))
#         return jnp.array(guesses)

#     def _compute_full_uncertainty(self, p_all, t_axis, irf_all, d_all, model_type, mode, batch_hess_fn):
#         """Calculates CRLB-based parameter standard deviations."""
#         H_all = batch_hess_fn(p_all, t_axis, irf_all, d_all)
#         eps = 1e-8 * jnp.eye(p_all.shape[1])
#         cov_matrices = jnp.linalg.inv(H_all + eps)
#         return jnp.sqrt(jnp.abs(jnp.diagonal(cov_matrices, axis1=1, axis2=2)))

#     def fit_image(self, image_cube, irf_cube, mask=None, mode='MLE', 
#                   model_type='bi-exponential', lr_init=0.005, epochs=500):
#         """
#         Executes full-image GPU fitting, Chi-squared calculation, and residual mapping.
#         """
#         start_time = time.time()
#         H, W, T = image_cube.shape
#         t_axis = jnp.linspace(0, self.T_acq, T)
        
#         # IRF Normalization
#         irf_sums = np.sum(irf_cube, axis=2, keepdims=True)
#         irf_cube_norm = np.divide(irf_cube, irf_sums, out=np.zeros_like(irf_cube), where=irf_sums!=0)
        
#         if mask is None:
#             mask = np.sum(image_cube, axis=2) > 20
        
#         valid_idx = np.where(mask.flatten())[0]
#         flat_data = jnp.array(image_cube.reshape(-1, T)[valid_idx])
#         flat_irf = jnp.array(irf_cube_norm.reshape(-1, T)[valid_idx])
        
#         loss_fn = self._loss_factory(mode)

#         # --- LAMBDA WRAPPERS TO FIX JAX TYPEERROR ---
#         # These closures keep 'self' and 'model_type' out of the vmap input stream
#         grad_single = jit(lambda p, t, i, d: grad(loss_fn)(p, t, i, d, model_type))
#         hess_single = jit(lambda p, t, i, d: hessian(loss_fn)(p, t, i, d, model_type))
#         model_single = jit(lambda p, t, i: self._model_kernel(p, t, i, model_type))

#         grad_fn = vmap(grad_single, in_axes=(0, None, 0, 0))
#         hess_fn = vmap(hess_single, in_axes=(0, None, 0, 0))
#         model_vmap = vmap(model_single, in_axes=(0, None, 0))
#         # ---------------------------------------------

#         print(f"--- JAX GPU Full-Image Fit ({mode}) ---")
#         p_all = self._get_sophisticated_guess(flat_data, flat_irf, model_type)
        
#         # Optimization Loop
#         decay_rate = 0.02
#         for epoch in range(epochs):
#             current_lr = lr_init / (1.0 + decay_rate * epoch)
#             grads = grad_fn(p_all, t_axis, flat_irf, flat_data)
#             p_all = p_all - current_lr * grads
            
#             p_all = jnp.clip(p_all, a_min=1e-6)
#             if model_type == 'bi-exponential':
#                 # alpha1, tau1, tau2 constraints
#                 p_all = p_all.at[:, 1].set(jnp.clip(p_all[:, 1], 0.0, 1.0))
#                 p_all = p_all.at[:, 2].set(jnp.clip(p_all[:, 2], 0.1, self.T_laser))
#                 p_all = p_all.at[:, 3].set(jnp.clip(p_all[:, 3], 0.1, self.T_laser))

#         # 3. Compute Fit, Residuals, and Chi-Squared
#         fit_flat = model_vmap(p_all, t_axis, flat_irf)
#         res_flat = flat_data - fit_flat
        
#         # Reduced Chi-Squared
#         dof = T - (5 if model_type == 'bi-exponential' else 3)
#         chi2_flat = jnp.sum((res_flat**2) / jnp.clip(fit_flat, 1.0), axis=1) / dof
        
#         perr_all = self._compute_full_uncertainty(p_all, t_axis, flat_irf, flat_data, model_type, mode, hess_fn)

#         # 4. Reconstruction to Image Space
#         full_popt = np.zeros((H * W, p_all.shape[1]))
#         full_perr = np.zeros((H * W, p_all.shape[1]))
#         full_fit = np.zeros((H * W, T))
#         full_res = np.zeros((H * W, T))
#         full_chi2 = np.zeros(H * W)

#         full_popt[valid_idx] = np.array(p_all)
#         full_perr[valid_idx] = np.array(perr_all)
#         full_fit[valid_idx] = np.array(fit_flat)
#         full_res[valid_idx] = np.array(res_flat)
#         full_chi2[valid_idx] = np.array(chi2_flat)

#         print(f"Processing Complete in {time.time() - start_time:.2f}s")
#         return self._reconstruct_dataset(
#             full_popt.reshape(H, W, -1), 
#             full_perr.reshape(H, W, -1), 
#             full_fit.reshape(H, W, T),
#             full_res.reshape(H, W, T),
#             full_chi2.reshape(H, W),
#             model_type, mode
#         )

#     def _reconstruct_dataset(self, p_maps, e_maps, fit_cube, res_cube, chi2_map, model_type, mode):
#         S = p_maps[..., 0]
#         if model_type == 'bi-exponential':
#             a1 = p_maps[..., 1]
#             maps = {
#                 'Area_map': S, 'Area_err': e_maps[..., 0],
#                 'alpha1_map': a1, 'alpha1_err': e_maps[..., 1],
#                 'tau1_map': p_maps[..., 2], 'tau1_err': e_maps[..., 2],
#                 'tau2_map': p_maps[..., 3], 'tau2_err': e_maps[..., 3],
#                 'Bkg_map': p_maps[..., 4], 'Bkg_err': e_maps[..., 4],
#                 'alpha2_map': 1.0 - a1,
#                 'Int_A1_map': S * a1, 'Int_A2_map': S * (1.0 - a1),
#                 'Chi2_map': chi2_map
#             }
#         else:
#             maps = {
#                 'Area_map': S, 'Area_err': e_maps[..., 0],
#                 'tau_map': p_maps[..., 1], 'tau_err': e_maps[..., 1],
#                 'Bkg_map': p_maps[..., 2], 'Bkg_err': e_maps[..., 2],
#                 'Chi2_map': chi2_map
#             }
#         return {
#             'method': f"GPU_{mode}", 
#             'results': {
#                 'maps': maps,
#                 'fit_cube': fit_cube,
#                 'residual_cube': res_cube
#             }
#         }

#     def save_results(self, dataset, folder="results", data_name="GPU_Fit"):
#         if not os.path.exists(folder): os.makedirs(folder)
#         h5_path = os.path.join(folder, f"{data_name}_GPU_results.h5")
#         with h5py.File(h5_path, "w") as f:
#             f.attrs['method'] = dataset['method']
#             res_grp = f.create_group("results")
#             maps_grp = res_grp.create_group("maps")
#             for k, v in dataset['results']['maps'].items():
#                 maps_grp.create_dataset(k, data=np.array(v).astype(np.float32), compression="gzip")
#             res_grp.create_dataset("fit_cube", data=dataset['results']['fit_cube'].astype(np.float32), compression="gzip")
#             res_grp.create_dataset("residual_cube", data=dataset['results']['residual_cube'].astype(np.float32), compression="gzip")
#         print(f"Saved: {h5_path}")


        # def save_to_hdf5(self, dataset, file_name=None):
    #     """
    #     Saves JAX results to the standardized HDF5 format.
    #     Structure: 
    #     results/maps/[parameter_maps]
    #     results/TR_maps/fit_map, residual_map
    #     """
    #     if file_name is None:
    #         file_name = f"{dataset['name']}_GPU_results.h5"
        
    #     print(f"--- Saving Results to {file_name} ---")
        
    #     with h5py.File(file_name, "w") as f:
    #         # Metadata
    #         f.attrs['method'] = dataset['method']
    #         f.attrs['processor'] = "Fli_GPUProcessor_JAX"
            
    #         res_grp = f.create_group("results")
            
    #         # 1. Save Parameter Maps
    #         maps_grp = res_grp.create_group("maps")
    #         for k, v in dataset['results']['maps'].items():
    #             # Ensure data is float32 and compressed
    #             maps_grp.create_dataset(k, data=np.array(v).astype(np.float32), compression="gzip")
            
    #         # 2. Save Time-Resolved Maps (fit and residuals)
    #         tr_grp = res_grp.create_group("TR_maps")
    #         tr_grp.create_dataset("fit_map", 
    #                               data=dataset['results']['TR_maps']['fit_map'], 
    #                               compression="gzip")
    #         tr_grp.create_dataset("residual_map", 
    #                               data=dataset['results']['TR_maps']['residual_map'], 
    #                               compression="gzip")
            
    #     print(f"Successfully saved {file_name}")

    # def generate_full_results(self, p_maps, image_cube, irf_cube, mask, model_type, method, data_name):
    #     """
    #     Reconstructs the fit and residual cubes on the GPU before saving.
    #     """
    #     H, W, T = image_cube.shape
    #     t_axis = jnp.linspace(0, self.T_acq, T)
        
    #     # Flatten for vectorized reconstruction
    #     flat_p = jnp.array(p_maps.reshape(-1, p_maps.shape[-1]))
    #     flat_irf = jnp.array(irf_cube.reshape(-1, T))
    #     flat_data = jnp.array(image_cube.reshape(-1, T))
        
    #     # Vectorized model generation
    #     reconstruct_vmap = jit(vmap(self._model_kernel, in_axes=(0, None, 0, None), static_argnums=(3,)))
        
    #     print("Reconstructing TR-Maps on GPU...")
    #     # We process the whole image or in large chunks to avoid VRAM overflow
    #     fit_flat = np.array(reconstruct_vmap(flat_p, t_axis, flat_irf, model_type))
    #     res_flat = np.array(flat_data) - fit_flat
        
    #     # Mask out background for the TR maps
    #     fit_flat[~mask.flatten()] = 0
    #     res_flat[~mask.flatten()] = 0

    #     # Build the final dictionary structure
    #     S = p_maps[..., 0]
    #     if model_type == 'bi-exponential':
    #         a1 = p_maps[..., 1]
    #         maps = {
    #             'alpha1_map': a1, 'tau1_map': p_maps[..., 2],
    #             'alpha2_map': 1.0 - a1, 'tau2_map': p_maps[..., 3],
    #             'Int_A1_map': S * a1, 'Int_A2_map': S * (1.0 - a1),
    #             'B_map': p_maps[..., 4]
    #         }
    #     else:
    #         maps = {'A_map': S, 'tau_map': p_maps[..., 1], 'B_map': p_maps[..., 2]}

    #     dataset = {
    #         'name': data_name,
    #         'method': method,
    #         'results': {
    #             'maps': maps,
    #             'TR_maps': {
    #                 'fit_map': fit_flat.reshape(H, W, T).astype(np.float32),
    #                 'residual_map': res_flat.reshape(H, W, T).astype(np.float32)
    #             }
    #         }
    #     }
    #     return dataset
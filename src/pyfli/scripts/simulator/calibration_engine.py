# simulator/calibration_engine.py
import numpy as np
import json
import os
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from .sim_image_generator import FLIImageGenerator
from .sim_stat_test import FLIValidator

class FLICalibrator:
    def __init__(self, irf_data, method='analytical', threshold=10):
        """
        Calibrates hardware noise parameters and manages hardware profiles.
        """
        self.irf_data = irf_data
        self.method = method.lower()
        self.threshold = threshold
        self.validator = FLIValidator(method=self.method, threshold=self.threshold)
        self.iteration = 0
        self.opt_params = None

    def _get_valid_counts(self, data_cube):
        """Helper to get valid intensities using the validator's logic."""
        _, counts = self.validator._preprocess_cube(data_cube)
        return counts

    def objective_function(self, x, exp_decay_cube, base_cfg):
        self.iteration += 1
        current_cfg = base_cfg.copy()
        current_cfg['dcr'] = x[0]
        if len(x) > 1:
            current_cfg['read_sigma'] = x[1]

        gen = FLIImageGenerator(self.irf_data, image_shape=(32, 32), 
                                roi_params=[current_cfg], method=self.method)
        sim_dataset = gen.generate_image()

        try:
            results = self.validator.run_comprehensive_test(sim_dataset, exp_decay_cube)
            if results is None: return 2.0
            
            p_val = results['ks_p_value']
            # Hybrid loss: minimize distance from ideal intersection and maximize p-value
            loss = (1.0 - p_val) + (1.0 - results['hist_intersection'])
            return loss
        except Exception as e:
            return 2.0

    def run_calibration(self, exp_decay_cube, base_config, initial_guess=[0.05, 1.2]):
        print(f"--- Starting Calibration: {self.method.upper()} ---")
        bounds = [(0, 0.5), (0, 10.0)]
        
        res = minimize(self.objective_function, x0=initial_guess, 
                       args=(exp_decay_cube, base_config),
                       bounds=bounds, method='L-BFGS-B', tol=1e-2)

        self.opt_params = {'dcr': res.x[0], 'read_sigma': res.x[1] if len(res.x)>1 else 0}
        
        final_cfg = base_config.copy()
        final_cfg.update(self.opt_params)
        return final_cfg

    # --- Persistence Layer ---
    def save_hardware_profile(self, filename="hw_profile.json"):
        """Saves calibrated parameters to a JSON file."""
        if self.opt_params is None:
            print("No calibrated parameters to save.")
            return
        
        profile = {
            "method": self.method,
            "threshold": self.threshold,
            "calibrated_params": self.opt_params
        }
        
        with open(filename, 'w') as f:
            json.dump(profile, f, indent=4)
        print(f"Hardware profile saved to {filename}")

    @staticmethod
    def load_hardware_profile(filename):
        """Loads a hardware profile from JSON."""
        if not os.path.exists(filename):
            print(f"File {filename} not found.")
            return None
        with open(filename, 'r') as f:
            return json.load(f)

    # --- Analysis & Visualization ---
    def plot_noise_sensitivity(self, train_exp_cube, base_config, 
                               dcr_range=(0.01, 0.2, 10), 
                               sigma_range=(0.5, 4.0, 10)):
        print("Generating Sensitivity Surface...")
        dcr_vals = np.linspace(*dcr_range)
        sigma_vals = np.linspace(*sigma_range)
        p_matrix = np.zeros((len(dcr_vals), len(sigma_vals)))
        
        target_counts = self._get_valid_counts(train_exp_cube)

        for i, dcr in enumerate(dcr_vals):
            for j, sigma in enumerate(sigma_vals):
                cfg = base_config.copy()
                cfg['dcr'], cfg['read_sigma'] = dcr, sigma
                
                gen = FLIImageGenerator(self.irf_data, image_shape=(32, 32), 
                                        roi_params=[cfg], method=self.method)
                sim_data = gen.generate_image()
                sim_counts = self._get_valid_counts(sim_data['raw_data']['decay'])
                
                _, p_val = stats.ks_2samp(sim_counts, target_counts)
                p_matrix[i, j] = p_val

        plt.figure(figsize=(9, 7))
        im = plt.imshow(p_matrix, extent=[sigma_vals[0], sigma_vals[-1], 
                                         dcr_vals[0], dcr_vals[-1]],
                        origin='lower', aspect='auto', cmap='magma')
        plt.colorbar(im, label='KS P-Value')
        plt.xlabel('Read Noise Sigma')
        plt.ylabel('DCR')
        
        if self.opt_params:
            plt.scatter(self.opt_params['read_sigma'], self.opt_params['dcr'], 
                        color='cyan', marker='x', s=100, label='Optimized')
        plt.show()
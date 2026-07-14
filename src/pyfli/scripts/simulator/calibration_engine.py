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
    def __init__(self, irf_data, method='analytical', threshold=10, normalize_stats=False):
        self.irf_data = irf_data
        self.method = method.lower()
        self.threshold = threshold
        self.normalize_stats = normalize_stats
        self.validator = FLIValidator(method=self.method, threshold=self.threshold)
        self.iteration = 0
        self.opt_params = None

    def _get_valid_counts(self, data_cube):
        _, counts = self.validator._preprocess_cube(data_cube)
        return counts

    def objective_function(self, x, exp_decay_cube, base_cfg):
        self.iteration += 1
        current_cfg = base_cfg.copy()
        
        # x = [DCR, Read_Sigma, Intensity_Alpha]
        current_cfg['dcr'] = x[0]
        current_cfg['read_sigma'] = x[1]
        
        pc_key = 'pc' if 'pc' in base_cfg else 'photo_count'
        orig_pc = list(base_cfg.get(pc_key, (8, 2)))
        current_cfg[pc_key] = (x[2], orig_pc[1])

        gen = FLIImageGenerator(self.irf_data, image_shape=(32, 32), 
                                roi_params=[current_cfg], method=self.method,
                                verbose=False)
        sim_dataset = gen.generate_image()

        try:
            results = self.validator.run_comprehensive_test(
                sim_dataset, exp_decay_cube, normalize=self.normalize_stats
            )
            if results is None: return 2.0
            
            p_val = results['ks_p_value']
            loss = (1.0 - p_val) + (1.0 - results['hist_intersection'])
            return loss
        except Exception:
            return 2.0

    def display_report(self, results):
        if results is None:
            print("No results to display.")
            return

        print("\n" + "="*60)
        print(f"STATISTICAL VALIDATION REPORT (N={results['sample_size']} Pixels)")
        print("="*60)
        print(f"{'Metric':<25} | {'Value':<15} | {'Target'}")
        print("-" * 60)
        print(f"{'Cosine Similarity':<25} | {results['cosine_similarity']:<15.4f} | >0.99")
        print(f"{'KL Divergence':<25} | {results['kl_divergence']:<15.4f} | <0.01")
        print(f"{'KS P-Value':<25} | {results['ks_p_value']:<15.4e} | >0.05")
        print(f"{'Hist Intersection':<25} | {results['hist_intersection']:<15.4f} | -> 1.0")
        print("="*60 + "\n")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        axes[0].plot(results['sim_vec'], label='Simulated (Mean)', lw=2)
        axes[0].plot(results['exp_vec'], label='Experimental (Mean)', ls='--', lw=2)
        axes[0].set_yscale('log')
        axes[0].set_title("Temporal Profile Fidelity")
        axes[0].set_xlabel("Time Bin")
        axes[0].set_ylabel("Normalized Intensity")
        axes[0].grid(True, which='both', alpha=0.3)
        axes[0].legend()

        axes[1].hist(results['sim_counts'], bins=50, alpha=0.5, label='Simulated', density=True, color='tab:blue')
        axes[1].hist(results['exp_counts'], bins=50, alpha=0.5, label='Experimental', density=True, color='tab:orange')
        axes[1].set_title("Integrated Intensity PDF")
        axes[1].set_xlabel("Photon Counts (Integrated)")
        axes[1].set_ylabel("Probability Density")
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()

    def run_calibration(self, exp_decay_cube, base_config, initial_guess=None):
        print(f"--- Starting Calibration: {self.method.upper()} (Norm: {self.normalize_stats}) ---")
        self.iteration = 0
        
        pc_key = 'pc' if 'pc' in base_config else 'photo_count'
        _, exp_counts = self.validator._preprocess_cube(exp_decay_cube)
        
        if initial_guess is None:
            # Smart guess for intensity alpha based on mean counts
            rough_alpha = np.mean(exp_counts) / 10 if self.method == 'analytical' else 5.0
            initial_guess = [0.01, 1.2, rough_alpha]

        # REVISED BOUNDS: Lowered DCR max to 0.1 to prevent overfitting in low-signal regimes
        bounds = [(0, 0.1), (0, 10.0), (0.1, 1000.0)]
        
        res = minimize(self.objective_function, x0=initial_guess, 
                       args=(exp_decay_cube, base_config),
                       bounds=bounds, method='L-BFGS-B', tol=1e-2)

        self.opt_params = {
            'dcr': float(res.x[0]), 
            'read_sigma': float(res.x[1]),
            pc_key: (float(res.x[2]), float(base_config[pc_key][1]))
        }
        
        final_cfg = base_config.copy()
        final_cfg.update(self.opt_params)
        
        print("\nGenerating Final Optimized Calibration Report...")
        gen = FLIImageGenerator(self.irf_data, image_shape=(32, 32), 
                                roi_params=[final_cfg], method=self.method)
        final_sim = gen.generate_image()
        metrics = self.validator.run_comprehensive_test(final_sim, exp_decay_cube, normalize=self.normalize_stats)
        self.display_report(metrics)
        
        return final_cfg

    def cross_validate(self, calibrated_cfg, test_exp_cube):
        print(f"\n--- Cross-Validation (Norm: {self.normalize_stats}) ---")
        gen = FLIImageGenerator(self.irf_data, image_shape=test_exp_cube.shape[:2], 
                                roi_params=[calibrated_cfg], method=self.method)
        sim_dataset = gen.generate_image()
        metrics = self.validator.run_comprehensive_test(sim_dataset, test_exp_cube, normalize=self.normalize_stats)
        self.display_report(metrics)
        return metrics

    def save_hardware_profile(self, filename="hw_profile.json"):
        if self.opt_params is None: return
        profile = {"method": self.method, "threshold": self.threshold, 
                   "normalize_used": self.normalize_stats, "params": self.opt_params}
        with open(filename, 'w') as f:
            json.dump(profile, f, indent=4)
        print(f"Profile saved to {filename}")

    @staticmethod
    def load_hardware_profile(filename):
        if not os.path.exists(filename): return None
        with open(filename, 'r') as f: return json.load(f)

    def plot_noise_sensitivity(self, train_exp_cube, base_config, dcr_range=(0.001, 0.1, 10), sigma_range=(0.5, 4.0, 10)):
        print("Generating Sensitivity Surface (Processing...)")
        dcr_vals = np.linspace(*dcr_range)
        sigma_vals = np.linspace(*sigma_range)
        p_matrix = np.zeros((len(dcr_vals), len(sigma_vals)))
        target_counts = self._get_valid_counts(train_exp_cube)

        for i, dcr in enumerate(dcr_vals):
            for j, sigma in enumerate(sigma_vals):
                cfg = base_config.copy()
                cfg['dcr'], cfg['read_sigma'] = dcr, sigma
                gen = FLIImageGenerator(self.irf_data, image_shape=(32, 32),
                                         roi_params=[cfg], 
                                         method=self.method, verbose = False)
                sim_data = gen.generate_image()
                sim_counts = self._get_valid_counts(sim_data['raw_data']['decay'])
                _, p_val = stats.ks_2samp(sim_counts, target_counts)
                p_matrix[i, j] = p_val

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(p_matrix, extent=[sigma_vals[0], sigma_vals[-1], dcr_vals[0], dcr_vals[-1]],
                       origin='lower', aspect='auto', cmap='magma')
        fig.colorbar(im, ax=ax, label='KS P-Value')
        ax.set_xlabel('Read Noise Sigma')
        ax.set_ylabel('DCR')
        if self.opt_params:
            ax.scatter(self.opt_params['read_sigma'], self.opt_params['dcr'], color='cyan', marker='x', s=100, label='Optimal Point')
            ax.legend()
        plt.show()
        return fig, p_matrix
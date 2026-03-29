# simulator/sim_stat_test .py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.special import rel_entr

class FLIValidator:
    def __init__(self, method='analytical', threshold=10):
        """
        Comprehensive Statistical Validator for FLI datasets.
        
        Args:
            method: 'analytical' (Macro) or 'tcspc'.
            threshold: Minimum integrated photon counts to consider a pixel valid.
        """
        self.method = method.lower()
        self.threshold = threshold

    def _preprocess_cube(self, data_cube):
        """
        Reshapes (H, W, T) to (N, T) and filters pixels below the intensity threshold.
        """
        if data_cube.ndim == 3:
            H, W, T = data_cube.shape
            flat_data = data_cube.reshape(-1, T)
        else:
            # Handle cases where data might already be (N, T)
            flat_data = data_cube
            T = flat_data.shape[-1]

        # Calculate integrated intensity per pixel (sum over time axis)
        pixel_intensities = np.sum(flat_data, axis=1)
        
        # Apply mask based on threshold
        valid_mask = pixel_intensities >= self.threshold
        filtered_data = flat_data[valid_mask]
        filtered_intensities = pixel_intensities[valid_mask]
        
        return filtered_data, filtered_intensities

    def run_comprehensive_test(self, sim_dataset, exp_decay_cube):
        """
        Executes a fair statistical comparison between simulated and experimental data.
        
        Args:
            sim_dataset: Dictionary from FLIImageGenerator.generate_image().
            exp_decay_cube: 3D numpy array of experimental measurements.
        """
        # 1. Flatten and Threshold
        sim_raw = sim_dataset['raw_data']['decay']
        exp_flat, exp_counts = self._preprocess_cube(exp_decay_cube)
        sim_flat_all, sim_counts_all = self._preprocess_cube(sim_raw)

        n_exp = exp_flat.shape[0]
        n_sim_total = sim_flat_all.shape[0]

        if n_exp == 0:
            print("Error: No experimental pixels passed the intensity threshold.")
            return None

        # 2. FAIR SUBSAMPLING
        # To avoid sample-size bias in KS-tests, match simulated N to experimental N
        if n_sim_total > n_exp:
            indices = np.random.choice(n_sim_total, size=n_exp, replace=False)
            sim_flat = sim_flat_all[indices]
            sim_counts = sim_counts_all[indices]
        else:
            sim_flat = sim_flat_all
            sim_counts = sim_counts_all
            print(f"Warning: Simulated valid pixels ({n_sim_total}) fewer than experimental ({n_exp}).")

        # --- Method 1: Temporal Decay Profile Analysis ---
        # Average temporal vector across all valid pixels
        sim_vec = np.mean(sim_flat, axis=0)
        exp_vec = np.mean(exp_flat, axis=0)
        
        # Cosine Similarity (Vector alignment in T-space)
        cos_sim = 1 - cosine(sim_vec, exp_vec)
        
        # KL Divergence (Probabilistic similarity)
        p = sim_vec / (np.sum(sim_vec) + 1e-12)
        q = exp_vec / (np.sum(exp_vec) + 1e-12)
        kl_div = np.sum(rel_entr(p, q))

        # --- Method 2: Integrated Intensity Distribution (Photon Counts) ---
        # Kolmogorov-Smirnov Test for distribution identity
        ks_stat, p_value = stats.ks_2samp(sim_counts, exp_counts)
        
        # Histogram Intersection Area
        bins = np.linspace(min(sim_counts.min(), exp_counts.min()), 
                           max(sim_counts.max(), exp_counts.max()), 50)
        hist_sim, _ = np.histogram(sim_counts, bins=bins, density=True)
        hist_exp, _ = np.histogram(exp_counts, bins=bins, density=True)
        intersection = np.minimum(hist_sim, hist_exp).sum() * (bins[1] - bins[0])

        # --- Output and Visualization ---
        self._print_summary(cos_sim, kl_div, ks_stat, p_value, intersection, len(sim_counts))
        self._plot_results(sim_vec, exp_vec, sim_counts, exp_counts)

        return {
            "cosine_similarity": cos_sim,
            "kl_divergence": kl_div,
            "ks_p_value": p_value,
            "hist_intersection": intersection,
            "sample_size": len(sim_counts)
        }

    def _print_summary(self, cos_sim, kl_div, ks_stat, p_value, intersection, n):
        print("\n" + "="*60)
        print(f"STATISTICAL VALIDATION REPORT (N={n} Pixels)")
        print("="*60)
        print(f"{'Metric':<25} | {'Value':<15} | {'Target'}")
        print("-" * 60)
        print(f"{'Cosine Similarity':<25} | {cos_sim:<15.4f} | >0.99")
        print(f"{'KL Divergence':<25} | {kl_div:<15.6f} | <0.01")
        print(f"{'KS Statistic':<25} | {ks_stat:<15.4f} | -> 0.0")
        print(f"{'KS P-Value':<25} | {p_value:<15.4e} | >0.05")
        print(f"{'Hist Intersection':<25} | {intersection:<15.4f} | -> 1.0")
        print("="*60 + "\n")

    def _plot_results(self, sim_vec, exp_vec, sim_counts, exp_counts):
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Mean Temporal Decay (Log Scale)
        ax[0].semilogy(sim_vec, label='Simulated (Mean)', color='tab:blue', lw=2)
        ax[0].semilogy(exp_vec, '--', label='Experimental (Mean)', color='tab:orange', lw=2)
        ax[0].set_title("Temporal Profile Fidelity", fontweight='bold')
        ax[0].set_xlabel("Time Bin")
        ax[0].set_ylabel("Normalized Intensity")
        ax[0].legend()
        ax[0].grid(True, which="both", alpha=0.3)
        
        # Plot 2: Intensity Probability Density Function
        ax[1].hist(sim_counts, bins=50, alpha=0.5, label='Simulated', color='tab:blue', density=True)
        ax[1].hist(exp_counts, bins=50, alpha=0.5, label='Experimental', color='tab:orange', density=True)
        ax[1].set_title("Integrated Intensity PDF", fontweight='bold')
        ax[1].set_xlabel("Photon Counts (Integrated)")
        ax[1].set_ylabel("Probability Density")
        ax[1].legend()
        
        plt.tight_layout()
        plt.show()
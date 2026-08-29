"""
Validate simulated and experimental decay distributions with summary statistics and
plots.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLI/FLIM
data generation, hardware noise modeling, calibration, and validation tools. Public API
includes classes :class:`FLIValidator`.
"""

from typing import Any

import matplotlib.pyplot as plt

# simulator/sim_stat_test .py
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.special import rel_entr

from pyfli import logging


class FLIValidator:
    """
    Compare simulated and experimental decay cubes with distribution summaries and
    diagnostic plots. It preprocesses cubes, computes similarity metrics, and reports
    validation statistics.

    Parameters
    ----------
    method : str
        Algorithm or model-selection method to use.
    threshold : int
        Threshold applied to counts, masks, or statistics.
    """

    def __init__(self, method: str = "analytical", threshold: int = 10) -> None:
        self.method = method.lower()
        self.threshold = threshold

    def _preprocess_cube(self, data_cube: np.ndarray) -> tuple[Any, ...]:
        """
        Run the preprocess cube routine.

        Parameters
        ----------
        data_cube : np.ndarray
            Array cube processed by the routine.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing processed decay cube, mask, and normalization metadata.
        """
        if data_cube.ndim == 3:
            H, W, T = data_cube.shape
            flat_data = data_cube.reshape(-1, T)
        else:
            flat_data = data_cube
            T = flat_data.shape[-1]

        pixel_intensities = np.sum(flat_data, axis=1)
        valid_mask = pixel_intensities >= self.threshold
        filtered_data = flat_data[valid_mask]
        filtered_intensities = pixel_intensities[valid_mask]

        return filtered_data, filtered_intensities

    def run_comprehensive_test(
        self,
        sim_dataset: np.ndarray,
        exp_decay_cube: np.ndarray,
        normalize: bool = False,
    ) -> Any:
        """
        Args:
            normalize: If True, scales both intensity distributions to [0, 1]
                       to compare noise morphology rather than absolute scale.
        """
        sim_raw = sim_dataset["raw_data"]["decay"]
        exp_flat, exp_counts = self._preprocess_cube(exp_decay_cube)
        sim_flat_all, sim_counts_all = self._preprocess_cube(sim_raw)

        n_exp = exp_flat.shape[0]
        n_sim_total = sim_flat_all.shape[0]

        if n_exp == 0:
            return None

        # Fair subsampling
        if n_sim_total > n_exp:
            indices = np.random.choice(n_sim_total, size=n_exp, replace=False)
            sim_flat = sim_flat_all[indices]
            sim_counts = sim_counts_all[indices]
        else:
            sim_flat, sim_counts = sim_flat_all, sim_counts_all

        # --- PRE-NORMALIZATION TOGGLE ---
        if normalize:
            # Scale counts to [0, 1] based on their respective max values
            exp_counts = exp_counts / (np.max(exp_counts) + 1e-12)
            sim_counts = sim_counts / (np.max(sim_counts) + 1e-12)

        # Method 1: Temporal
        sim_vec = np.mean(sim_flat, axis=0)
        exp_vec = np.mean(exp_flat, axis=0)
        cos_sim = 1 - cosine(sim_vec, exp_vec)

        p = sim_vec / (np.sum(sim_vec) + 1e-12)
        q = exp_vec / (np.sum(exp_vec) + 1e-12)
        kl_div = np.sum(rel_entr(p, q))

        # Method 2: Intensity Distribution
        ks_stat, p_value = stats.ks_2samp(sim_counts, exp_counts)

        bins = np.linspace(
            min(sim_counts.min(), exp_counts.min()),
            max(sim_counts.max(), exp_counts.max()),
            50,
        )
        hist_sim, _ = np.histogram(sim_counts, bins=bins, density=True)
        hist_exp, _ = np.histogram(exp_counts, bins=bins, density=True)
        intersection = np.minimum(hist_sim, hist_exp).sum() * (bins[1] - bins[0])

        return {
            "cosine_similarity": cos_sim,
            "kl_divergence": kl_div,
            "ks_p_value": p_value,
            "hist_intersection": intersection,
            "sample_size": len(sim_counts),
            "sim_vec": sim_vec,
            "exp_vec": exp_vec,
            "sim_counts": sim_counts,
            "exp_counts": exp_counts,
        }

    def _print_summary(
        self,
        cos_sim: Any,
        kl_div: np.ndarray,
        ks_stat: np.ndarray,
        p_value: np.ndarray,
        intersection: np.ndarray,
        n: Any,
    ) -> None:
        """
        Print summary.

        Parameters
        ----------
        cos_sim : Any
            Cosine similarity statistic printed in the validation summary.
        kl_div : np.ndarray
            Kullback-Leibler divergence values to plot or summarize.
        ks_stat : np.ndarray
            Kolmogorov-Smirnov statistic printed in the validation summary.
        p_value : np.ndarray
            P-value printed in the validation summary.
        intersection : np.ndarray
            Histogram-intersection score printed in the validation summary.
        n : Any
            Number of samples, bins, gates, or plotted items.

        Returns
        -------
        None
            No object is returned; the function perform print summary.
        """
        logging.info("\n" + "=" * 60)
        logging.info(f"STATISTICAL VALIDATION REPORT (N={n} Pixels)")
        logging.info("=" * 60)
        logging.info(f"{'Metric':<25} | {'Value':<15} | {'Target'}")
        logging.info("-" * 60)
        logging.info(f"{'Cosine Similarity':<25} | {cos_sim:<15.4f} | >0.99")
        logging.info(f"{'KL Divergence':<25} | {kl_div:<15.6f} | <0.01")
        logging.info(f"{'KS Statistic':<25} | {ks_stat:<15.4f} | -> 0.0")
        logging.info(f"{'KS P-Value':<25} | {p_value:<15.4e} | >0.05")
        logging.info(f"{'Hist Intersection':<25} | {intersection:<15.4f} | -> 1.0")
        logging.info("=" * 60 + "\n")

    def _plot_results(
        self,
        sim_vec: np.ndarray,
        exp_vec: np.ndarray,
        sim_counts: np.ndarray,
        exp_counts: np.ndarray,
    ) -> np.ndarray:
        """
        Plot results.

        Parameters
        ----------
        sim_vec : np.ndarray
            Simulated vector used by the validation plot.
        exp_vec : np.ndarray
            Experimental vector used by the validation plot.
        sim_counts : np.ndarray
            Simulated histogram counts used by the validation plot.
        exp_counts : np.ndarray
            Experimental histogram counts used by the validation plot.

        Returns
        -------
        np.ndarray
            Matplotlib figure or axes containing the simulation results.
        """
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Mean Temporal Decay (Log Scale)
        ax[0].semilogy(sim_vec, label="Simulated (Mean)", color="tab:blue", lw=2)
        ax[0].semilogy(
            exp_vec, "--", label="Experimental (Mean)", color="tab:orange", lw=2
        )
        ax[0].set_title("Temporal Profile Fidelity", fontweight="bold")
        ax[0].set_xlabel("Time Bin")
        ax[0].set_ylabel("Normalized Intensity")
        ax[0].legend()
        ax[0].grid(True, which="both", alpha=0.3)

        # Plot 2: Intensity Probability Density Function
        ax[1].hist(
            sim_counts,
            bins=50,
            alpha=0.5,
            label="Simulated",
            color="tab:blue",
            density=True,
        )
        ax[1].hist(
            exp_counts,
            bins=50,
            alpha=0.5,
            label="Experimental",
            color="tab:orange",
            density=True,
        )
        ax[1].set_title("Integrated Intensity PDF", fontweight="bold")
        ax[1].set_xlabel("Photon Counts (Integrated)")
        ax[1].set_ylabel("Probability Density")
        ax[1].legend()

        plt.tight_layout()
        plt.show()
        return fig

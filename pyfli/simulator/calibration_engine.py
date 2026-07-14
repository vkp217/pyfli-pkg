"""
Calibrate simulator hardware parameters against experimental decay cubes.

This module belongs to :mod:`pyfli.simulator` and is part of PyFLI synthetic FLIM data
generation, hardware noise modeling, calibration, and validation tools. Public API
includes classes :class:`FLICalibrator`.
"""

from typing import Any

from pyfli import logging

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
    """
    Estimate simulator hardware parameters from experimental decay cubes. It optimizes
    noise and detector settings, reports calibration quality, cross-validates results,
    and can save reusable hardware profiles.

    Parameters
    ----------
    irf_data : np.ndarray
        Instrument response data used to convolve or simulate decays.
    method : str
        Algorithm or model-selection method to use.
    threshold : int
        Threshold applied to counts, masks, or statistics.
    normalize_stats : bool
        Whether calibration statistics are normalized before comparison.
    """

    def __init__(
        self,
        irf_data: np.ndarray,
        method: str = "analytical",
        threshold: int = 10,
        normalize_stats: bool = False,
    ) -> None:
        self.irf_data = irf_data
        self.method = method.lower()
        self.threshold = threshold
        self.normalize_stats = normalize_stats
        self.validator = FLIValidator(method=self.method, threshold=self.threshold)
        self.iteration = 0
        self.opt_params = None

    def _get_valid_counts(self, data_cube: np.ndarray) -> np.ndarray:
        """
        Return valid counts.

        Parameters
        ----------
        data_cube : np.ndarray
            Array cube processed by the routine.

        Returns
        -------
        np.ndarray
            Mask or count array identifying valid simulated pixels.
        """
        _, counts = self.validator._preprocess_cube(data_cube)
        return counts

    def objective_function(
        self, x: np.ndarray, exp_decay_cube: np.ndarray, base_cfg: np.ndarray
    ) -> Any:
        """
        Run the objective function routine.

        Parameters
        ----------
        x : np.ndarray
            Input array, coordinate, or signal being transformed.
        exp_decay_cube : np.ndarray
            Experimental decay cube used for calibration.
        base_cfg : np.ndarray
            Base simulator configuration copied during calibration.

        Returns
        -------
        Any
            Object produced by objective function.
        """
        self.iteration += 1
        current_cfg = base_cfg.copy()

        # x = [DCR, Read_Sigma, Intensity_Alpha]
        current_cfg["dcr"] = x[0]
        current_cfg["read_sigma"] = x[1]

        pc_key = "pc" if "pc" in base_cfg else "photo_count"
        orig_pc = list(base_cfg.get(pc_key, (8, 2)))
        current_cfg[pc_key] = (x[2], orig_pc[1])

        gen = FLIImageGenerator(
            self.irf_data,
            image_shape=(32, 32),
            roi_params=[current_cfg],
            method=self.method,
            verbose=False,
        )
        sim_dataset = gen.generate_image()

        try:
            results = self.validator.run_comprehensive_test(
                sim_dataset, exp_decay_cube, normalize=self.normalize_stats
            )
            if results is None:
                return 2.0

            p_val = results["ks_p_value"]
            loss = (1.0 - p_val) + (1.0 - results["hist_intersection"])
            return loss
        except Exception:
            return 2.0

    def display_report(self, results: Any) -> None:
        """
        Display report.

        Parameters
        ----------
        results : Any
            Calibration, fitting, or validation results.

        Returns
        -------
        None
            No object is returned; the function display report.
        """
        if results is None:
            logging.info("No results to display.")
            return

        logging.info("\n" + "=" * 60)
        logging.info(
            f"STATISTICAL VALIDATION REPORT (N={results['sample_size']} Pixels)"
        )
        logging.info("=" * 60)
        logging.info(f"{'Metric':<25} | {'Value':<15} | {'Target'}")
        logging.info("-" * 60)
        logging.info(
            f"{'Cosine Similarity':<25} | {results['cosine_similarity']:<15.4f} | >0.99"
        )
        logging.info(
            f"{'KL Divergence':<25} | {results['kl_divergence']:<15.4f} | <0.01"
        )
        logging.info(f"{'KS P-Value':<25} | {results['ks_p_value']:<15.4e} | >0.05")
        logging.info(
            f"{'Hist Intersection':<25} | {results['hist_intersection']:<15.4f} | -> 1.0"
        )
        logging.info("=" * 60 + "\n")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(results["sim_vec"], label="Simulated (Mean)", lw=2)
        axes[0].plot(results["exp_vec"], label="Experimental (Mean)", ls="--", lw=2)
        axes[0].set_yscale("log")
        axes[0].set_title("Temporal Profile Fidelity")
        axes[0].set_xlabel("Time Bin")
        axes[0].set_ylabel("Normalized Intensity")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()

        axes[1].hist(
            results["sim_counts"],
            bins=50,
            alpha=0.5,
            label="Simulated",
            density=True,
            color="tab:blue",
        )
        axes[1].hist(
            results["exp_counts"],
            bins=50,
            alpha=0.5,
            label="Experimental",
            density=True,
            color="tab:orange",
        )
        axes[1].set_title("Integrated Intensity PDF")
        axes[1].set_xlabel("Photon Counts (Integrated)")
        axes[1].set_ylabel("Probability Density")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    def run_calibration(
        self,
        exp_decay_cube: np.ndarray,
        base_config: np.ndarray,
        initial_guess: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Run calibration.

        Parameters
        ----------
        exp_decay_cube : np.ndarray
            Experimental decay cube used for calibration.
        base_config : np.ndarray
            Base simulator configuration used for calibration or sensitivity analysis.
        initial_guess : np.ndarray | None
            Initial optimizer parameter vector.

        Returns
        -------
        np.ndarray
            Calibration results for the configured simulator.
        """
        logging.info(
            f"--- Starting Calibration: {self.method.upper()} (Norm: {self.normalize_stats}) ---"
        )
        self.iteration = 0

        pc_key = "pc" if "pc" in base_config else "photo_count"
        _, exp_counts = self.validator._preprocess_cube(exp_decay_cube)

        if initial_guess is None:
            # Smart guess for intensity alpha based on mean counts
            rough_alpha = (
                np.mean(exp_counts) / 10 if self.method == "analytical" else 5.0
            )
            initial_guess = [0.01, 1.2, rough_alpha]

        # REVISED BOUNDS: Lowered DCR max to 0.1 to prevent overfitting in low-signal regimes
        bounds = [(0, 0.1), (0, 10.0), (0.1, 1000.0)]

        res = minimize(
            self.objective_function,
            x0=initial_guess,
            args=(exp_decay_cube, base_config),
            bounds=bounds,
            method="L-BFGS-B",
            tol=1e-2,
        )

        self.opt_params = {
            "dcr": float(res.x[0]),
            "read_sigma": float(res.x[1]),
            pc_key: (float(res.x[2]), float(base_config[pc_key][1])),
        }

        final_cfg = base_config.copy()
        final_cfg.update(self.opt_params)

        logging.info("\nGenerating Final Optimized Calibration Report...")
        gen = FLIImageGenerator(
            self.irf_data,
            image_shape=(32, 32),
            roi_params=[final_cfg],
            method=self.method,
        )
        final_sim = gen.generate_image()
        metrics = self.validator.run_comprehensive_test(
            final_sim, exp_decay_cube, normalize=self.normalize_stats
        )
        self.display_report(metrics)

        return final_cfg

    def cross_validate(
        self, calibrated_cfg: np.ndarray, test_exp_cube: np.ndarray
    ) -> np.ndarray:
        """
        Run the cross validate routine.

        Parameters
        ----------
        calibrated_cfg : np.ndarray
            Calibrated simulator configuration used for validation.
        test_exp_cube : np.ndarray
            Experimental decay cube used for cross-validation.

        Returns
        -------
        np.ndarray
            Cross-validation scores for the calibration model.
        """
        logging.info(f"\n--- Cross-Validation (Norm: {self.normalize_stats}) ---")
        gen = FLIImageGenerator(
            self.irf_data,
            image_shape=test_exp_cube.shape[:2],
            roi_params=[calibrated_cfg],
            method=self.method,
        )
        sim_dataset = gen.generate_image()
        metrics = self.validator.run_comprehensive_test(
            sim_dataset, test_exp_cube, normalize=self.normalize_stats
        )
        self.display_report(metrics)
        return metrics

    def save_hardware_profile(self, filename: str = "hw_profile.json") -> None:
        """
        Save hardware profile.

        Parameters
        ----------
        filename : str
            File name used for saving or loading results.

        Returns
        -------
        None
            No object is returned; the function save hardware profile.
        """
        if self.opt_params is None:
            return
        profile = {
            "method": self.method,
            "threshold": self.threshold,
            "normalize_used": self.normalize_stats,
            "params": self.opt_params,
        }
        with open(filename, "w") as f:
            json.dump(profile, f, indent=4)
        logging.info(f"Profile saved to {filename}")

    @staticmethod
    def load_hardware_profile(filename: str) -> Any:
        """
        Load hardware profile.

        Parameters
        ----------
        filename : str
            File name used for saving or loading results.

        Returns
        -------
        Any
            Object produced by load hardware profile.
        """
        if not os.path.exists(filename):
            return None
        with open(filename, "r") as f:
            return json.load(f)

    def plot_noise_sensitivity(
        self,
        train_exp_cube: np.ndarray,
        base_config: np.ndarray,
        dcr_range: tuple[float, float, int] = (0.001, 0.1, 10),
        sigma_range: tuple[float, float, int] = (0.5, 4.0, 10),
    ) -> tuple[Any, ...]:
        """
        Plot noise sensitivity.

        Parameters
        ----------
        train_exp_cube : np.ndarray
            Array cube processed by the routine.
        base_config : np.ndarray
            Base simulator configuration used for calibration or sensitivity analysis.
        dcr_range : tuple[float, float, int]
            Dark-count-rate values evaluated during sensitivity analysis.
        sigma_range : tuple[float, float, int]
            Read-noise sigma values evaluated by the sensitivity plot.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing noise-sensitivity figure data and summary metrics.
        """
        logging.info("Generating Sensitivity Surface (Processing...)")
        dcr_vals = np.linspace(*dcr_range)
        sigma_vals = np.linspace(*sigma_range)
        p_matrix = np.zeros((len(dcr_vals), len(sigma_vals)))
        target_counts = self._get_valid_counts(train_exp_cube)

        for i, dcr in enumerate(dcr_vals):
            for j, sigma in enumerate(sigma_vals):
                cfg = base_config.copy()
                cfg["dcr"], cfg["read_sigma"] = dcr, sigma
                gen = FLIImageGenerator(
                    self.irf_data,
                    image_shape=(32, 32),
                    roi_params=[cfg],
                    method=self.method,
                    verbose=False,
                )
                sim_data = gen.generate_image()
                sim_counts = self._get_valid_counts(sim_data["raw_data"]["decay"])
                _, p_val = stats.ks_2samp(sim_counts, target_counts)
                p_matrix[i, j] = p_val

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(
            p_matrix,
            extent=[sigma_vals[0], sigma_vals[-1], dcr_vals[0], dcr_vals[-1]],
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        fig.colorbar(im, ax=ax, label="KS P-Value")
        ax.set_xlabel("Read Noise Sigma")
        ax.set_ylabel("DCR")
        if self.opt_params:
            ax.scatter(
                self.opt_params["read_sigma"],
                self.opt_params["dcr"],
                color="cyan",
                marker="x",
                s=100,
                label="Optimal Point",
            )
            ax.legend()
        plt.show()
        return fig, p_matrix

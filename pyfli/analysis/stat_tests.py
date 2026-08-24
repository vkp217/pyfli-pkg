"""
Compare simulated and experimental FLI/FLIM distributions with classical and
multivariate tests.

This module belongs to :mod:`pyfli.analysis` and is part of PyFLI post-processing,
diagnostics, statistical comparison, and result-loading utilities for fitted FLI/FLIM
datasets. Public API includes classes :class:`TestStat` and
:class:`FLIDistributionTest`.
"""

from typing import Any

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel


class TestStat:
    """
    Run the test stat routine.
    batches. The class groups Anderson-Darling, Kolmogorov-Smirnov, likelihood-ratio,
    bootstrap confidence interval, and Bayesian evidence helpers behind one object.

    Parameters
    ----------
    sim_batch : np.ndarray
        Simulated result batch used as the reference distribution.
    exp_batch : np.ndarray
        Experimental result batch used for comparison.
    eps : float
        Small numerical tolerance used to avoid division-by-zero and boundary issues.
    """

    def __init__(
        self, sim_batch: np.ndarray, exp_batch: np.ndarray, eps: float = 1e-12
    ) -> None:
        self.sim = np.asarray(sim_batch, dtype=np.float64)
        self.exp = np.asarray(exp_batch, dtype=np.float64)
        self.eps = eps

        assert self.sim.shape == self.exp.shape
        self.B, self.n_bins = self.sim.shape

        # Normalize to PDFs
        self.sim_pdf = self.sim / (self.sim.sum(axis=1, keepdims=True) + eps)
        self.exp_pdf = self.exp / (self.exp.sum(axis=1, keepdims=True) + eps)

        # CDFs
        self.sim_cdf = np.cumsum(self.sim_pdf, axis=1)
        self.exp_cdf = np.cumsum(self.exp_pdf, axis=1)

    # Anderson–Darling Test (Shape Sensitive)

    def anderson_darling(self) -> np.ndarray:
        """
        Batch AD statistic (two-sample version approximation)
        """
        ad_stats = np.zeros(self.B)

        for i in range(self.B):
            F = self.sim_cdf[i]
            G = self.exp_cdf[i]

            H = (F + G) / 2.0
            H = np.clip(H, self.eps, 1 - self.eps)

            ad = np.sum((F - G) ** 2 / (H * (1 - H)))
            ad_stats[i] = ad

        return ad_stats

    # Kolmogorov–Smirnov Test (CDF-based)

    def kolmogorov_smirnov(self) -> np.ndarray:
        """
        Run the kolmogorov smirnov routine.

        Returns
        -------
        np.ndarray
            Kolmogorov-Smirnov statistic and p-value for the supplied samples.
        """
        ks_stats = np.max(np.abs(self.sim_cdf - self.exp_cdf), axis=1)
        return ks_stats

    # Likelihood Ratio Test (Mono vs Bi)

    def likelihood_ratio(self) -> Any:
        """
        Poisson likelihood ratio:
        Λ = 2 (LL_bi - LL_mono)

        Assumes sim_batch = biexp model
        exp_batch = data
        """
        sim = self.sim
        exp = self.exp

        # Poisson log-likelihood
        LL = np.sum(exp * np.log(sim + self.eps) - sim, axis=1)

        # Null model: mono approx (fit best scalar exponential via total count scaling)
        mono_model = np.mean(sim, axis=1, keepdims=True)
        LL_null = np.sum(exp * np.log(mono_model + self.eps) - mono_model, axis=1)

        LR = 2 * (LL - LL_null)
        return LR

    # Bootstrap Confidence Intervals

    def bootstrap_ci(
        self, metric_func: np.ndarray, n_boot: int = 200
    ) -> tuple[Any, ...]:
        """
        Generic bootstrap CI over batch
        """
        values = metric_func()
        boot_means = []

        for _ in range(n_boot):
            idx = np.random.choice(self.B, self.B, replace=True)
            boot_means.append(np.mean(values[idx]))

        lower = np.percentile(boot_means, 2.5)
        upper = np.percentile(boot_means, 97.5)

        return lower, upper

    # Bayesian Evidence (AIC/BIC Approximation)

    def bayesian_evidence(self, k_mono: int = 2, k_bi: int = 4) -> np.ndarray:
        """
        Approximate log evidence using BIC
        """
        sim = self.sim
        exp = self.exp

        N = self.n_bins

        LL = np.sum(exp * np.log(sim + self.eps) - sim, axis=1)

        BIC_mono = -2 * LL + k_mono * np.log(N)
        BIC_bi = -2 * LL + k_bi * np.log(N)

        delta_BIC = BIC_mono - BIC_bi

        return delta_BIC

    # MASTER FUNCTION
    def run_all_tests(self) -> np.ndarray:
        """
        Run all tests.

        Returns
        -------
        np.ndarray
            Summary table or array containing the configured statistical test results.
        """
        results = {}

        # Core statistics
        results["anderson_darling"] = self.anderson_darling()
        results["ks_stat"] = self.kolmogorov_smirnov()
        results["likelihood_ratio"] = self.likelihood_ratio()
        results["delta_BIC"] = self.bayesian_evidence()

        # Confidence intervals
        results["AD_CI"] = self.bootstrap_ci(self.anderson_darling)
        results["KS_CI"] = self.bootstrap_ci(self.kolmogorov_smirnov)
        results["LR_CI"] = self.bootstrap_ci(self.likelihood_ratio)

        return results


class FLIDistributionTest:
    """
    Run the flidistribution test routine.
    MMD, energy distance, sliced Wasserstein, Frechet-style, and PCA-overlap metrics for
    validating whether simulations match measured data.

    Parameters
    ----------
    sim_batch : np.ndarray
        Simulated result batch used as the reference distribution.
    exp_batch : np.ndarray
        Experimental result batch used for comparison.
    eps : float
        Small numerical tolerance used to avoid division-by-zero and boundary issues.
    """

    def __init__(
        self, sim_batch: np.ndarray, exp_batch: np.ndarray, eps: float = 1e-12
    ) -> None:
        self.sim = sim_batch.astype(np.float64)
        self.exp = exp_batch.astype(np.float64)
        self.eps = eps

        # Normalize decays to PDFs
        self.sim /= self.sim.sum(axis=1, keepdims=True) + eps
        self.exp /= self.exp.sum(axis=1, keepdims=True) + eps

        self.N, self.D = self.sim.shape

    # ==========================================================
    # 1️⃣ Maximum Mean Discrepancy (BEST CHOICE)
    # ==========================================================
    def mmd(self, gamma: float | None = None) -> np.ndarray:
        """
        Kernel two-sample test.
        """
        if gamma is None:
            gamma = 1.0 / self.D

        Kxx = rbf_kernel(self.sim, self.sim, gamma=gamma)
        Kyy = rbf_kernel(self.exp, self.exp, gamma=gamma)
        Kxy = rbf_kernel(self.sim, self.exp, gamma=gamma)

        mmd_value = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()

        return mmd_value

    # ==========================================================
    # 2️⃣ Energy Distance
    # ==========================================================
    def energy_distance(self) -> Any:
        """
        Run the energy distance routine.

        Returns
        -------
        Any
            Object produced by energy distance.
        """
        X = self.sim
        Y = self.exp

        d_xy = np.linalg.norm(X[:, None] - Y[None, :], axis=2).mean()
        d_xx = np.linalg.norm(X[:, None] - X[None, :], axis=2).mean()
        d_yy = np.linalg.norm(Y[:, None] - Y[None, :], axis=2).mean()

        return 2 * d_xy - d_xx - d_yy

    # ==========================================================
    # 3️⃣ Sliced Wasserstein Distance
    # ==========================================================
    def sliced_wasserstein(self, n_projections: int = 50) -> np.ndarray:
        """
        Project high-D distributions to random 1D lines.
        """
        distances = []

        for _ in range(n_projections):
            direction = np.random.randn(self.D)
            direction /= np.linalg.norm(direction)

            proj_sim = self.sim @ direction
            proj_exp = self.exp @ direction

            distances.append(wasserstein_distance(proj_sim, proj_exp))

        return np.mean(distances)

    # ==========================================================
    # 4️⃣ Fréchet Distance (FID-style)
    # ==========================================================
    def frechet_distance(self) -> np.ndarray:
        """
        Run the frechet distance routine.

        Returns
        -------
        np.ndarray
            Frechet distance between the supplied curves or point sequences.
        """
        mu1 = self.sim.mean(axis=0)
        mu2 = self.exp.mean(axis=0)

        sigma1 = np.cov(self.sim, rowvar=False)
        sigma2 = np.cov(self.exp, rowvar=False)

        diff = mu1 - mu2

        cov_prod = sigma1 @ sigma2
        covmean = sqrtm(cov_prod)

        # Numerical stability
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)

        return fid

    # ==========================================================
    # 5️⃣ PCA Manifold Overlap
    # ==========================================================
    def pca_overlap(self, n_components: int = 10) -> np.ndarray:
        """
        Run the PCA overlap routine.

        Parameters
        ----------
        n_components : int
            Number of PCA components retained for the metric.

        Returns
        -------
        np.ndarray
            Overlap score between PCA projections of the supplied groups.
        """
        pca = PCA(n_components=n_components)

        combined = np.vstack([self.sim, self.exp])
        pca.fit(combined)

        sim_proj = pca.transform(self.sim)
        exp_proj = pca.transform(self.exp)

        sim_var = np.var(sim_proj, axis=0)
        exp_var = np.var(exp_proj, axis=0)

        overlap = np.mean(
            np.minimum(sim_var, exp_var) / (np.maximum(sim_var, exp_var) + self.eps)
        )

        return overlap

    # ==========================================================
    # MASTER FUNCTION
    # ==========================================================
    def run_all(self) -> dict[Any, Any]:
        """
        Run all.

        Returns
        -------
        dict[Any, Any]
            Dictionary containing the data produced by run all.
        """
        return {
            "MMD": self.mmd(),
            "EnergyDistance": self.energy_distance(),
            "SlicedWasserstein": self.sliced_wasserstein(),
            "FrechetDistance": self.frechet_distance(),
            "PCA_Overlap": self.pca_overlap(),
        }

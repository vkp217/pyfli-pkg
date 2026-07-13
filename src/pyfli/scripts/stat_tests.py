"""Statistical comparison tests for simulated vs experimental decay batches.

Provides two batch-oriented test suites:

- `TestStat`: shape/likelihood-based goodness-of-fit statistics (Anderson-
  Darling, Kolmogorov-Smirnov, Poisson likelihood ratio, and a BIC-based
  Bayesian evidence approximation) with bootstrap confidence intervals.
- `FLIDistributionTest`: distribution-level discrepancy metrics between
  two batches of normalized decay curves (MMD, energy distance, sliced
  Wasserstein distance, Frechet distance, and PCA manifold overlap).
"""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance
from scipy.linalg import sqrtm

class TestStat:
    """Batch goodness-of-fit statistics comparing simulated vs experimental decays.

    Each row of `sim_batch`/`exp_batch` is normalized into a PDF and CDF
    on construction; the various test methods then operate across the
    batch dimension, returning one statistic per batch element (or, for
    `bootstrap_ci`/`bayesian_evidence`, aggregated/derived values).

    Attributes:
        sim: Simulated decay batch, cast to float64.
        exp: Experimental decay batch, cast to float64.
        eps: Small constant used to avoid division by zero.
        B: Batch size (number of rows).
        n_bins: Number of time bins (columns) per decay.
        sim_pdf: `sim` normalized to sum to 1 along each row.
        exp_pdf: `exp` normalized to sum to 1 along each row.
        sim_cdf: Cumulative sum of `sim_pdf` along each row.
        exp_cdf: Cumulative sum of `exp_pdf` along each row.
    """

    def __init__(self, sim_batch, exp_batch, eps=1e-12):
        """
        sim_batch : (B, n_bins)
        exp_batch : (B, n_bins)
        """
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

    def anderson_darling(self):
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

    def kolmogorov_smirnov(self):
        """Compute the per-batch Kolmogorov-Smirnov statistic.

        Returns:
            np.ndarray: Shape `(B,)` array of KS statistics, the maximum
            absolute difference between the simulated and experimental
            CDFs for each batch element.
        """
        ks_stats = np.max(np.abs(self.sim_cdf - self.exp_cdf), axis=1)
        return ks_stats


    # Likelihood Ratio Test (Mono vs Bi)

    def likelihood_ratio(self):
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

    def bootstrap_ci(self, metric_func, n_boot=200):
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

    def bayesian_evidence(self, k_mono=2, k_bi=4):
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
    def run_all_tests(self):
        """Run all batch statistics and bootstrap confidence intervals.

        Returns:
            dict: With keys `'anderson_darling'`, `'ks_stat'`,
            `'likelihood_ratio'`, `'delta_BIC'` (each a `(B,)` array from
            the corresponding method), and `'AD_CI'`, `'KS_CI'`,
            `'LR_CI'` (each a `(lower, upper)` tuple from
            `bootstrap_ci` applied to the Anderson-Darling, KS, and
            likelihood-ratio statistics respectively).
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
    """Distribution-level discrepancy metrics between two decay batches.

    Both batches are normalized (row-wise) to PDFs on construction. The
    methods compute complementary notions of "distance" between the
    resulting empirical distributions of simulated vs experimental decay
    curves: kernel-based (MMD), geometric (energy distance, sliced
    Wasserstein), Gaussian-approximation (Frechet distance), and
    variance-overlap in a shared PCA subspace.

    Attributes:
        sim: Simulated decay batch, row-normalized to PDFs (float64).
        exp: Experimental decay batch, row-normalized to PDFs (float64).
        eps: Small constant used to avoid division by zero.
        N: Number of samples (rows) in each batch.
        D: Number of bins (columns) per decay.
    """

    def __init__(self, sim_batch, exp_batch, eps=1e-12):
        """
        sim_batch: (N, n_bins)
        exp_batch: (N, n_bins)
        """
        self.sim = sim_batch.astype(np.float64)
        self.exp = exp_batch.astype(np.float64)
        self.eps = eps

        # Normalize decays to PDFs
        self.sim /= (self.sim.sum(axis=1, keepdims=True) + eps)
        self.exp /= (self.exp.sum(axis=1, keepdims=True) + eps)

        self.N, self.D = self.sim.shape

    # ==========================================================
    # 1️⃣ Maximum Mean Discrepancy (BEST CHOICE)
    # ==========================================================
    def mmd(self, gamma=None):
        """
        Kernel two-sample test.
        """
        if gamma is None:
            gamma = 1.0 / self.D

        Kxx = rbf_kernel(self.sim, self.sim, gamma=gamma)
        Kyy = rbf_kernel(self.exp, self.exp, gamma=gamma)
        Kxy = rbf_kernel(self.sim, self.exp, gamma=gamma)

        mmd_value = (
            Kxx.mean()
            + Kyy.mean()
            - 2 * Kxy.mean()
        )

        return mmd_value

    # ==========================================================
    # 2️⃣ Energy Distance
    # ==========================================================
    def energy_distance(self):
        """Compute the (squared Euclidean) statistical energy distance.

        Uses the standard estimator `2*E|X-Y| - E|X-X'| - E|Y-Y'|` over
        all pairwise distances between and within the two batches.

        Returns:
            float: Energy distance between the simulated and
            experimental distributions; 0 for identical distributions,
            larger for more dissimilar ones.
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
    def sliced_wasserstein(self, n_projections=50):
        """
        Project high-D distributions to random 1D lines.
        """
        distances = []

        for _ in range(n_projections):
            direction = np.random.randn(self.D)
            direction /= np.linalg.norm(direction)

            proj_sim = self.sim @ direction
            proj_exp = self.exp @ direction

            distances.append(
                wasserstein_distance(proj_sim, proj_exp)
            )

        return np.mean(distances)

    # ==========================================================
    # 4️⃣ Fréchet Distance (FID-style)
    # ==========================================================
    def frechet_distance(self):
        """Compute the Frechet distance (FID-style) between the two batches.

        Fits a multivariate Gaussian to each batch (mean and covariance
        across samples) and computes the closed-form Frechet distance
        between the two Gaussians, using `scipy.linalg.sqrtm` for the
        matrix square root of the covariance product (discarding any
        residual imaginary component for numerical stability).

        Returns:
            float: Frechet distance between the simulated and
            experimental distributions.
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

        fid = (
            diff @ diff
            + np.trace(sigma1 + sigma2 - 2 * covmean)
        )

        return fid

    # ==========================================================
    # 5️⃣ PCA Manifold Overlap
    # ==========================================================
    def pca_overlap(self, n_components=10):
        """Measure variance overlap between the batches in a shared PCA subspace.

        Fits PCA on the concatenation of both batches, projects each
        batch into that subspace, and compares per-component variances.

        Args:
            n_components: Number of PCA components to fit and compare.

        Returns:
            float: Mean, over components, of the ratio
            `min(var_sim, var_exp) / max(var_sim, var_exp)`. Values
            close to 1 indicate similar spread along the shared
            principal components; values close to 0 indicate divergent
            spread.
        """
        pca = PCA(n_components=n_components)

        combined = np.vstack([self.sim, self.exp])
        pca.fit(combined)

        sim_proj = pca.transform(self.sim)
        exp_proj = pca.transform(self.exp)

        sim_var = np.var(sim_proj, axis=0)
        exp_var = np.var(exp_proj, axis=0)

        overlap = np.mean(np.minimum(sim_var, exp_var) /
                          (np.maximum(sim_var, exp_var) + self.eps))

        return overlap

    # ==========================================================
    # MASTER FUNCTION
    # ==========================================================
    def run_all(self):
        """Run all distribution-level discrepancy metrics.

        Returns:
            dict: With keys `'MMD'`, `'EnergyDistance'`,
            `'SlicedWasserstein'`, `'FrechetDistance'`, and
            `'PCA_Overlap'`, each holding the corresponding method's
            scalar result (computed with default arguments).
        """
        return {
            "MMD": self.mmd(),
            "EnergyDistance": self.energy_distance(),
            "SlicedWasserstein": self.sliced_wasserstein(),
            "FrechetDistance": self.frechet_distance(),
            "PCA_Overlap": self.pca_overlap()
        }

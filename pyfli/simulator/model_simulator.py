# simulator/simulator_engine
import numpy as np

from .distributions import ParameterSampler
from .noise_models import NoiseEngine
from .sim_helper import irf_picker


class FLIModelSimulator:
    def __init__(
        self,
        irf_full,
        tau2=(1, 0.5),
        tau2_dist="normal",  # 'normal' -> truncated_normal | 'beta' -> sample_beta
        tau2_beta_range=(4.8, 0.2),  # (scale, offset), only used when tau2_dist='beta'
        efficiency=(5, 5),
        A1_fraction=(5, 5),
        photo_count=(1.2, 5),
        mono_fraction=0.2,
        bit=8,
        n_cycles=800_000,
        dcr=0.05,
        laser_feq=80,
        seed=None,
        **kwargs,
    ):

        irf = irf_picker(irf_full)
        # Timing and Normalization
        irf_sum = irf.sum()
        if not np.isfinite(irf_sum) or irf_sum <= 0:
            raise ValueError(
                f"Invalid IRF: sum={irf_sum}. IRF must be non-negative and non-zero."
            )
        self.irf = irf / irf_sum
        self.rng = np.random.default_rng(seed)
        self.laser_period = 1000 / laser_feq
        # N bins each of width dt covering [0, laser_period); endpoint-exclusive
        self.dt = self.laser_period / len(self.irf)
        self.t = np.arange(len(self.irf)) * self.dt

        #  Parameters Storage
        self.params_cfg = {
            "tau2": tau2,
            "tau2_dist": tau2_dist,
            "tau2_beta_range": tau2_beta_range,
            "eff": efficiency,
            "A1": A1_fraction,
            "pc": photo_count,
            "mono": mono_fraction,
            "bit": bit,
            "cycles": n_cycles,
            "dcr": dcr,
            **kwargs,
        }

    def _sample_tau2(self):
        """Draws tau2 from either a truncated normal or a beta prior, per tau2_dist."""
        if self.params_cfg["tau2_dist"] == "beta":
            scale, offset = self.params_cfg["tau2_beta_range"]
            t2 = ParameterSampler.sample_beta(
                *self.params_cfg["tau2"], scale=scale, offset=offset, rng=self.rng
            )
        else:
            t2 = ParameterSampler.truncated_normal(*self.params_cfg["tau2"])
        # Guard against a zero (or negative) lifetime, which would blow up any 1/tau2 term downstream
        return max(t2, 1e-3)

    def sample_mono_params(self):
        """Samples the lifetime parameter for a single pixel (pure single-exponential)."""
        t2 = self._sample_tau2()
        return {"mono": True, "tau": t2}

    def sample_bi_params(self):
        """Samples lifetime and fraction parameters for a single pixel."""
        t2 = self._sample_tau2()

        E = ParameterSampler.sample_beta(
            *self.params_cfg["eff"], scale=0.998, offset=0.001, rng=self.rng
        )
        A1 = ParameterSampler.sample_beta(
            *self.params_cfg["A1"], scale=0.998, offset=0.001, rng=self.rng
        )
        A2 = 1.0 - A1
        t1 = t2 * (1 - E)
        # Pulsed-repetition correction (same formula as mono mode)
        w1 = A1 * (1 - np.exp(-self.laser_period / t1))
        w2 = A2 * (1 - np.exp(-self.laser_period / t2))
        denom = w1 + w2
        f = w1 / denom if denom > 0 else A1
        return {
            "mono": False,
            "E": E,
            "f": f,
            "tau1": t1,
            "tau2": t2,
            "A1": A1,
            "A2": A2,
        }

    def sample_params(self):
        if self.rng.random() < self.params_cfg["mono"]:
            return self.sample_mono_params()
        return self.sample_bi_params()

    def get_model_analytical_decay(self, p):
        T = self.laser_period
        if p.get("mono", False):
            tau = p["tau"]
            scaling_factor = 1.0 / (1.0 - np.exp(-T / tau))
            return scaling_factor * np.exp(-self.t / tau)

        decay = np.zeros_like(self.t)
        i = 1
        while f"tau{i}" in p:
            tau_i = p[f"tau{i}"]
            A_i = p[f"A{i}"]
            scaling_factor_i = 1.0 / (1.0 - np.exp(-T / tau_i))
            decay = decay + A_i * scaling_factor_i * np.exp(-self.t / tau_i)
            i += 1
        return decay

    def simulate_model_tcspc(self, p, n_cycles, mu_per_cycle):
        total_photons = self.rng.poisson(mu_per_cycle * n_cycles)
        if total_photons == 0:
            return np.zeros_like(self.t)

        # Emission (Inverse Transform Sampling)
        times = np.empty(total_photons)
        if p.get("mono", False):
            times[:] = self.rng.exponential(p["tau"], size=total_photons)
        else:
            comp1 = self.rng.random(total_photons) < p["f"]
            times[comp1] = self.rng.exponential(p["tau1"], size=comp1.sum())
            times[~comp1] = self.rng.exponential(p["tau2"], size=(~comp1).sum())

        # IRF Convolution (Sampling from the IRF distribution)
        irf_cdf = np.cumsum(self.irf)
        irf_cdf[-1] = 1.0  # pin to exactly 1.0 so searchsorted never returns len(irf)
        irf_shifts = np.searchsorted(irf_cdf, self.rng.random(total_photons)) * self.dt

        # Filter for pile-up and repetitive excitation window
        arrival_times = NoiseEngine.tcspc_pileup_filter(
            times + irf_shifts, self.laser_period
        )

        # Binning
        bins = (arrival_times / self.dt).astype(np.int32)
        hist = np.bincount(bins[bins < len(self.t)], minlength=len(self.t))

        return hist.astype(np.float64)

# pyfli/simulator/main_common.py

"""
Shared engine-wrapping pipelines for the continuous (ICCD-style) and discrete
(TCSPC-style) simulator pairs.

``combined/main_factory.py`` (:class:`~pyfli.simulator.combined.main_factory.MacroSimulator`,
:class:`~pyfli.simulator.combined.main_factory.TCSPCSimulator`, wrapping
:class:`~pyfli.simulator.combined.simulator_engine.FLIEngine`) and
``separate/main_factory_gen.py`` (:class:`~pyfli.simulator.separate.main_factory_gen.ContinuousSimulator`,
:class:`~pyfli.simulator.separate.main_factory_gen.PhotonCountSimulator`, wrapping
:class:`~pyfli.simulator.separate.model_simulator.FLIModelSimulator`) implement the identical
noise/scaling pipeline; they differ only in which engine class they wrap, the names of
a few methods on that engine, and whether the "mono" branch trims the returned maps
dict. Subclasses declare those differences as class attributes; this module owns the
actual pipeline math so it is defined exactly once.
"""

from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from .distributions import ParameterSampler
from .noise_models import NoiseEngine


def _build_maps(
    p: Any, photon_count: Any, mono_only_maps: bool, h_shift: Any
) -> dict[Any, Any]:
    """Build the parameter-maps dict, optionally trimmed for mono pixels."""
    if mono_only_maps and p["mono"]:
        return {
            "tau_map": p["tau"],
            "photon_count_map": photon_count,
            "mono_map": p["mono"],
            "h_shift_jit_map": h_shift,
            "v_shift_map": 0,
        }
    return {
        "tau1_map": p["tau1"],
        "tau2_map": p["tau2"],
        "alpha1_map": p["f"],
        "A1_map": p["A1"],
        "A2_map": p["A2"],
        "fret_efficiency_map": p["E"],
        "tau_mean_map": p["tau1"] * p["f"] + p["tau2"] * (1 - p["f"]),
        "photon_count_map": photon_count,
        "mono_map": p["mono"],
        "h_shift_jit_map": h_shift,
        "v_shift_map": 0,
    }


class BaseContinuousSimulator:
    """
    Shared ``__call__`` pipeline for MacroSimulator/ContinuousSimulator: samples
    lifetime parameters, scales an IRF-convolved analytical decay to a Beta-sampled
    peak intensity, and applies the jitter/QE/DCR/read-noise/Poisson/quantization
    pipeline.

    Subclasses set the following class attributes:

    engine_cls : type
        The engine class to wrap (``FLIEngine`` / ``FLIModelSimulator``).
    sample_params_name : str
        Name of the engine method that samples lifetime parameters.
    analytical_decay_name : str
        Name of the engine method that returns the clean analytical decay.
    mono_only_maps : bool
        Whether ``p["mono"]`` trims the returned maps dict down to
        ``{tau_map, photon_count_map, mono_map}`` (``True``, as in
        ``ContinuousSimulator``) or the full map set is always returned
        regardless of mono status (``False``, as in ``MacroSimulator``).
    """

    engine_cls: type
    sample_params_name: str
    analytical_decay_name: str
    mono_only_maps: bool = False

    def __init__(
        self, irf_data: np.ndarray, sensor_type: str = "continuous", **cfg: Any
    ) -> None:
        # Toggles
        self.use_jitter = cfg.get("jitter", True)
        self.use_dcr = cfg.get("dcr_on", True)
        self.use_poisson = cfg.get("poisson", True)
        self.use_qe = cfg.get("qe_on", True)
        self.use_read_noise = cfg.get("read_noise_on", True)
        self.use_rounding = cfg.get("round_on", True)
        self.use_clipping = cfg.get("clip_on", True)

        self.sensor_type = sensor_type.upper()
        self.engine = self.engine_cls(irf_data, **cfg)

    def __call__(self) -> dict[Any, Any]:
        """
        Run the instance as a callable.

        Returns
        -------
        dict[Any, Any]
            Dictionary containing the data produced by call.
        """
        p = getattr(self.engine, self.sample_params_name)()

        # 1. Determine target intensity (A) based on bit-depth
        alpha_pc, beta_pc = self.engine.params_cfg["pc"]
        bit_depth = self.engine.params_cfg["bit"]
        max_adc_val = (2**bit_depth) - 1

        # Sample peak intensity (A) from Beta distribution
        A = ParameterSampler.beta_sample(alpha_pc, beta_pc, scale=max_adc_val)

        # Generating Clean Analytical Convolution
        clean_decay = getattr(self.engine, self.analytical_decay_name)(p)
        full_conv = fftconvolve(clean_decay, self.engine.irf, mode="full")[
            : len(clean_decay)
        ]

        # INITIAL SCALING: Scale the convolved signal so its PEAK matches A
        if np.max(full_conv) > 0:
            scale_factor = A / np.max(full_conv)
            obs = full_conv * scale_factor
        else:
            obs = full_conv.copy()

        # Applying Modular Noise Pipeline
        shift = 0
        if self.use_jitter:
            shift = np.random.randint(-2, 3)
            n = len(obs)
            if shift > 0:
                obs = np.concatenate([np.zeros(shift), obs[: n - shift]])
            elif shift < 0:
                obs = np.concatenate([obs[-shift:], np.zeros(-shift)])

        if self.use_qe:
            obs = obs * ParameterSampler.sample_qe(self.sensor_type)

        if self.use_dcr:
            bit_scaling = bit_depth / 8.0
            obs = NoiseEngine.apply_dcr(
                obs, self.engine.params_cfg["dcr"] * bit_scaling
            )

        if self.use_read_noise and self.sensor_type == "CONTINUOUS":
            hw = ParameterSampler.sample_noise_params(bit_depth, self.sensor_type)
            obs = NoiseEngine.apply_read_noise(obs, hw["read_sigma"])

        if self.use_poisson:
            # Poisson must be applied to the intensity-scaled signal
            obs = NoiseEngine.apply_poisson(obs)

        # Final Quantization
        if self.use_rounding:
            obs = np.round(obs)

        if self.use_clipping:
            obs = np.clip(obs, 0, max_adc_val)

        # FINAL FIT SCALING: Scale the "Clean Fit" to match the "Observed Peak"
        # This ensures the residuals (obs - fit) reflect ONLY the noise/stochastics
        obs_peak = np.max(obs)
        if obs_peak > 0:
            fit_map = full_conv * (
                obs_peak / np.max(full_conv) if np.max(full_conv) > 0 else 1.0
            )
            if self.use_jitter:
                n = len(fit_map)
                if shift > 0:
                    fit_map = np.concatenate([np.zeros(shift), fit_map[: n - shift]])
                elif shift < 0:
                    fit_map = np.concatenate([fit_map[-shift:], np.zeros(-shift)])
        else:
            fit_map = np.zeros_like(obs)

        maps = _build_maps(p, A, self.mono_only_maps, shift)

        return {
            "raw_data": {"decay": obs, "irf": self.engine.irf},
            "results": {
                "maps": maps,
                "TR_maps": {"fit_map": fit_map, "residual_map": obs - fit_map},
            },
        }


class BaseDiscreteSimulator:
    """
    Shared ``__call__`` pipeline for TCSPCSimulator/PhotonCountSimulator: samples
    lifetime parameters, runs a TCSPC photon-by-photon histogram, and applies the
    jitter/DCR/quantization pipeline.

    Subclasses set the following class attributes:

    engine_cls : type
        The engine class to wrap (``FLIEngine`` / ``FLIModelSimulator``).
    sample_params_name : str
        Name of the engine method that samples lifetime parameters.
    analytical_decay_name : str
        Name of the engine method that returns the clean analytical decay.
    simulate_tcspc_name : str
        Name of the engine method that runs the TCSPC photon histogram.
    mono_only_maps : bool
        As in :class:`BaseContinuousSimulator`.
    """

    engine_cls: type
    sample_params_name: str
    analytical_decay_name: str
    simulate_tcspc_name: str
    mono_only_maps: bool = False

    def __init__(
        self, irf_data: np.ndarray, sensor_type: str = "discrete", **cfg: Any
    ) -> None:
        self.use_jitter = cfg.get("jitter", True)
        self.use_dcr = cfg.get("dcr_on", True)
        self.use_qe = cfg.get("qe_on", False)

        # TCSPC is inherently integer-based, but clipping simulates counter overflow
        self.use_clipping = cfg.get("clip_on", True)

        self.sensor_type = sensor_type.upper()
        # TCSPC counters are 16-bit by default
        cfg.setdefault("bit", 16)
        self.engine = self.engine_cls(irf_data, **cfg)

    def __call__(self) -> dict[Any, Any]:
        """
        Run the instance as a callable.

        Returns
        -------
        dict[Any, Any]
            Dictionary containing the data produced by call.
        """
        p = getattr(self.engine, self.sample_params_name)()
        low_cycles, high_cycles = self.engine.params_cfg["cycles"]
        alpha_cyc, beta_cyc = self.engine.params_cfg["pc"]
        n_cycles = int(
            round(
                ParameterSampler.sample_beta(
                    alpha_cyc,
                    beta_cyc,
                    scale=high_cycles - low_cycles,
                    offset=low_cycles,
                )
            )
        )
        mu_per_cycle = 0.01
        bit_depth = self.engine.params_cfg["bit"]
        max_bin_count = (2**bit_depth) - 1

        effective_mu = (
            mu_per_cycle * ParameterSampler.sample_qe(self.sensor_type)
            if self.use_qe
            else mu_per_cycle
        )
        obs = getattr(self.engine, self.simulate_tcspc_name)(p, n_cycles, effective_mu)

        # Fit Scaling
        total_photons_expected = effective_mu * n_cycles
        clean = getattr(self.engine, self.analytical_decay_name)(p)
        fit_norm = fftconvolve(clean, self.engine.irf, mode="full")[: len(clean)]
        fit = (
            fit_norm * (total_photons_expected / np.sum(fit_norm))
            if np.sum(fit_norm) > 0
            else fit_norm
        )

        shift = 0
        if self.use_jitter:
            shift = np.random.randint(-2, 3)
            n = len(obs)
            if shift > 0:
                obs = np.concatenate([np.zeros(shift), obs[: n - shift]])
                fit = np.concatenate([np.zeros(shift), fit[: n - shift]])
            elif shift < 0:
                obs = np.concatenate([obs[-shift:], np.zeros(-shift)])
                fit = np.concatenate([fit[-shift:], np.zeros(-shift)])

        if self.use_dcr:
            # DCR is a detector property (dark counts/bin/cycle) — independent of counter bit depth
            obs = NoiseEngine.apply_dcr(obs, self.engine.params_cfg["dcr"])

        if self.use_clipping:
            obs = np.clip(obs, 0, max_bin_count)

        maps = _build_maps(p, total_photons_expected, self.mono_only_maps, shift)

        return {
            "raw_data": {"decay": obs, "irf": self.engine.irf},
            "results": {
                "maps": maps,
                "TR_maps": {"fit_map": fit, "residual_map": obs - fit},
            },
        }

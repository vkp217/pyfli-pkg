import numpy as np

from .main_factory_gen import ContinuousSimulator, PhotonCountSimulator

SIMULATOR_TYPES = {
    "photon_counter": PhotonCountSimulator,
    "continuous": ContinuousSimulator,
}


def concat_sim_data(*datasets):
    """
    Concatenate multiple sim.sample() output dicts along the batch axis (axis=0).
    Assumes all datasets share identical keys and per-sample shapes.
    """
    keys = datasets[0].keys()
    # sanity check: all datasets must have the same keys
    for i, d in enumerate(datasets[1:], start=1):
        if set(d.keys()) != set(keys):
            missing = set(keys) - set(d.keys())
            extra = set(d.keys()) - set(keys)
            raise ValueError(
                f"Key mismatch in dataset {i}: missing={missing}, extra={extra}"
            )

    combined = {}
    for k in keys:
        arrs = [d[k] for d in datasets]
        combined[k] = np.concatenate(arrs, axis=0)
    return combined


class SimOutput:
    def __init__(self, simulator):
        self.simulator = simulator

    def run(self):
        val = self.simulator()
        maps = val["results"]["maps"]
        if maps["mono_map"]:
            tau1 = tau2 = tau = maps["tau_map"]
            alpha1 = 1.0
            efficiency = 1.0
        else:
            tau1, tau2 = maps["tau1_map"], maps["tau2_map"]
            tau = maps["tau_mean_map"]
            alpha1 = maps["alpha1_map"]
            efficiency = maps["fret_efficiency_map"]
        return {
            "decay": val["raw_data"]["decay"],
            "irf_": val["raw_data"]["irf"],
            "tau1": tau1,
            "tau2": tau2,
            "tau": tau,
            "alpha1": alpha1,
            "photon_count": maps["photon_count_map"],
            "Efficiency": efficiency,
        }


class SimOutputWithIRFOffset(SimOutput):
    def __init__(self, simulator, irf_1d):
        super().__init__(simulator)
        self.irf_1d = irf_1d

    def _compute_irf_offset(self):
        i_off = self.irf_1d
        i_off_sum = i_off.sum()
        if not np.isfinite(i_off_sum) or i_off_sum <= 0:
            raise ValueError(
                f"Invalid IRF: sum={i_off_sum}. IRF must be non-negative and non-zero."
            )
        i_off = i_off / i_off_sum
        return i_off

    def run(self):
        out = super().run()
        out["irf"] = self._compute_irf_offset()
        return out


class SimGenerator:
    """
    Wraps IRF shifting + PhotonCountSim/ContinuousSim + SimOutputWithIRFOffset
    for a single config. Only accepts one config dict — raises if given a
    list/tuple of configs.

    Parameters
    ----------
    simulator_type : str
        Which simulator engine to sample from — "photon_counter" (default,
        :class:`PhotonCountSimulator`) or "continuous"
        (:class:`ContinuousSimulator`).
    """

    def __init__(
        self,
        irf_data,
        config,
        a_range=(-20, 100),
        b_range=(0, 10),
        pixel=(0, 0),
        simulator_type="photon_counter",
    ):
        if isinstance(config, (list, tuple)):
            raise TypeError(
                "SimGenerator accepts exactly one config dict, not a "
                "list/tuple of configs. Instantiate one generator per config "
                "and combine their sampled outputs afterward if you need "
                "multiple configs."
            )
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config)}")
        if simulator_type not in SIMULATOR_TYPES:
            raise ValueError(
                f"simulator_type must be one of {list(SIMULATOR_TYPES)}, "
                f"got {simulator_type!r}"
            )

        self.config = config
        self.a_range = a_range
        self.b_range = b_range
        self.simulator_cls = SIMULATOR_TYPES[simulator_type]
        self.I_base = irf_data[pixel[0], pixel[1], :].astype(float)  # shape (n_bins,)
        self.n_bins = self.I_base.shape[0]

    def _make_shifted_irf_1d(self, a, b):
        """
        Circular shift I(t) -> I(t-a), add offset b. Returns 1D (n_bins,).
        NOTE: np.roll requires an integer shift; `a` is rounded to the
        nearest int here since it's sampled from a continuous uniform range.
        """
        a_int = int(round(a))
        return np.roll(self.I_base, a_int) + b

    def simulate_once(self):
        a = np.random.uniform(*self.a_range)
        b = np.random.uniform(*self.b_range)
        irf_1d = self._make_shifted_irf_1d(a, b)

        fli_simulator = self.simulator_cls(irf_data=irf_1d, **self.config)
        out = SimOutputWithIRFOffset(fli_simulator, irf_1d).run()
        out["h_shift"] = a
        out["v_shift"] = b
        return out


def make_simulator(simulate_fn, num_samples):
    """Run simulate_fn num_samples times and stack each output key along a new leading axis."""
    samples = [simulate_fn() for _ in range(num_samples)]
    keys = samples[0].keys()
    return {
        key: np.stack([np.asarray(s[key]) for s in samples], axis=0) for key in keys
    }

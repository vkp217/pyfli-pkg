# Quickstart

This walks through the minimal steps to load an FLI dataset and inspect the decay/IRF traces.

## Load experimental data

[`DataOperations`](../api/data_io.md) is the entry point for reading raw FLI data alongside its instrument response function (IRF), background, and mask:

```python
from pyfli import DataOperations

loader = DataOperations(
    data_path="experimental_data.sdt",
    irf_path="instrument_data.txt",
    bg_path="background_data.tif",
    mask_path="background_data.png",
)

decay_data = loader.load_data()
irf_data = loader.load_irf()
```

## Fit lifetimes

Once you have decay and IRF data, pick a fitting strategy from the [solvers](../api/solvers.md) module — for example, a non-linear least-squares fit on CPU:

```python
from pyfli import Fli_CPUProcessor

fitter = Fli_CPUProcessor(decay_data, irf_data)
results = fitter.fit_with_estimator()
```

## Phasor analysis (model-free alternative)

For a model-free, per-pixel view of lifetime species, transform the same decay data into phasor space using the [phasor module](../api/phasor.md):

```python
from pyfli import phasor_continuous, plot_phasor

g, s = phasor_continuous(decay_data, irf_data)
plot_phasor(g, s)
```

## Simulate synthetic data

To generate synthetic FLI datasets for testing or benchmarking, see the [simulator](../api/simulator.md) module:

```python
from pyfli import TCSPC_sim

sim = TCSPC_sim(...)
synthetic_decay = sim.generate()
```

## Next steps

- Browse the full [API Reference](../api/index.md) for every available class and function.
- See [Supported Data Formats](data-formats.md) for the hardware/file types `pyfli` reads natively.

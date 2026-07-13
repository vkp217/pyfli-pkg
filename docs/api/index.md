# API Reference

The public API of `pyfli` is organized here by functional area rather than by internal file layout. Everything documented on these pages is importable directly from the top-level package unless a page states otherwise:

```python
from pyfli import DataOperations, LaguerreFLI, phasor_continuous
```

## Areas

| Area | Covers |
|---|---|
| [Data I/O](data_io.md) | Loading raw FLI data (ICCD, SPAD, TCSPC) and instrument-specific importers |
| [Preprocessing & Calibration](preprocessing.md) | IRF alignment, normalization, masking, ROI-based preprocessing |
| [Analytical Methods](analytical_methods.md) | Laguerre deconvolution, phasor-based helpers, mono/bi-exponential classification |
| [Solvers & Fitters](solvers.md) | NLSF, MLE, CPU/GPU, binned and global fitting engines |
| [Simulator](simulator.md) | Synthetic FLI dataset generation and noise modeling |
| [IRF Deconvolution](irf_deconvolution.md) | Low-level gate-matrix / cyclic-convolution deconvolution solver |
| [Phasor Analysis](phasor.md) | Continuous/discrete/gated phasor transforms, loci, lifetime extraction |
| [Session & Results Analysis](analysis.md) | Loading, aggregating, and plotting saved fitting/phasor results |
| [Visualization](visualization.md) | Plotting utilities, data viewers, colormap tooling |
| [ROI Tools](roi.md) | Interactive region-of-interest selection |
| [Single-Pixel (SPAD) Analysis](sp_analysis.md) | Basis-pattern generation and reconstruction for single-pixel/SPAD imaging |
| [Utilities](utilities.md) | Shared helpers: messaging, data saving, plotting/statistics helpers |

Each page is generated directly from the package's docstrings, so it always reflects the current source in [`src/pyfli`](https://github.com/vkp217/pyfli-pkg/tree/main/src/pyfli).

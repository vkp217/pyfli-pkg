# pyfli: A Unified Platform for FLI Data Processing

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://img.shields.io/pypi/v/pyfli-lib.svg)](https://pypi.org/project/pyfli-lib/)

`pyfli` is a comprehensive library designed for **Fluorescence Lifetime Imaging (FLI)** data processing. It streamlines the workflow for handling diverse file formats from various hardware manufacturers and provides a standardized pipeline for both traditional analytical and deep-learning-based inference.

---

## Key Features

* **Universal Processing Pipeline:** Simplifies the handling of multiple FLI file types (ICCD, SPAD, TCSPC).
* **Enhanced FLI Simulator:** A robust simulation engine adaptable to specific camera hardware parameters and noise models.
* **Standardized Inference:** Unified interface for time-resolved microscopy and macroscopic FLI data (MFLI).

## Supported Data Acquisition Methods

The platform provides native support for several high-end imaging systems:

1. **ICCD:** Intensified Charge-Coupled Device cameras for fast-gated, wide-field imaging.
2. **SwissSPAD2 & SwissSPAD3:** High-speed SPAD (Single-Photon Avalanche Diode) architectures for high-resolution photon counting.
3. **SPCImage/TCSPC:** Standardized processing for Time-Correlated Single Photon Counting microscopy data.

## Data Processing & Analysis

`pyfli` implements industry-standard analytical methods to extract lifetime information:

* **Non-linear Least Squares Fitting (NLSF):** Robust mathematical approach for exponential decay modeling.
* **Phasor Plot Analysis:** Graphical, model-free transformation of fluorescence decay into a 2D polar plot for easy species separation.
* **Maximum Likelihood Estimation (MLE):** Statistical estimator optimized for low-photon regimes.
* **Rapid Lifetime Determination (RLD):** Computationally efficient method for real-time applications and high-frame-rate data.
* **Laguerre Method (LET):** Laguerre Expansion Technique for model-free IRF deconvolution followed by multi-exponential lifetime extraction on a per-pixel basis.

---

## Installation

Install the stable version directly from PyPI:

```bash
pip install pyfli-lib
```

For users requiring GPU-based processing, install the optional tensor/AI dependencies:

```bash
pip install "pyfli-lib[gpu]"
```

## Quick Start

Even though the package is installed as `pyfli-lib`, you import it as `pyfli` in your scripts:

```python
from pyfli import DataOperations

loader = DataOperations(    
    data_path = "experimental_data.sdt",
    irf_path = "instrument_data.txt", 
    bg_path = "background_data.tif",   
    mask_path="background_data.png",
    )
decay_data = loader.load_data()
irf_data = loader.load_irf()

```

## Citation

If you use `pyfli` in your research, please cite this package:

> Pandey V. *pyfli: A Unified Platform for Fluorescence Lifetime Imaging Data Processing.*
> https://github.com/vkp217/pyfli-pkg/tree/joss-submission

```bibtex
@article{pandey2025pyfli,
  author  = {Pandey, Vikas},
  title   = {{pyfli}: A Unified Platform for Fluorescence Lifetime Imaging Data Processing},
  journal = {},
  year    = {2025},
  note    = {},
  url     = {https://github.com/vkp217/pyfli-pkg/tree/joss-submission}
}
```

If you use the **phasor SEPL analysis** functionality specifically, please also cite the following paper on which the phasor module is based:

> Michalet X. "Continuous and discrete phasor analysis of binned or time-gated periodic decays."
> *AIP Advances* **11**, 035331 (2021).
> https://doi.org/10.1063/5.0027834

---

## Repository & Issues

The source code is hosted on GitHub. Please report any bugs or feature requests via the issues tracker.
* **GitHub:** [https://github.com/vkp217/pyfli-pkg](https://github.com/vkp217/pyfli-pkg)


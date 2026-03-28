# pyfli: A Unified Platform for FLI Data Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
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

* **Non-linear Least Squares (NLLS) Fitting:** Robust mathematical approach for exponential decay modeling.
* **Phasor Plot Analysis:** Graphical, model-free transformation of fluorescence decay into a 2D polar plot for easy species separation.
* **Maximum Likelihood Estimation (MLE):** Statistical estimator optimized for low-photon regimes.
* **Rapid Lifetime Determination (RLD):** Computationally efficient method for real-time applications and high-frame-rate data.

---

## Installation

Install the stable version directly from PyPI:

```bash
pip install pyfli-lib
```

For users requiring deep-learning features (TensorFlow/PyTorch), install the optional AI dependencies:

```bash
pip install "pyfli-lib[ai]"
```

## Quick Start

Even though the package is installed as `pyfli-lib`, you import it as `pyfli` in your scripts:

```python
import pyfli

# Load an SDT (TCSPC) file
fli_data = pyfli.io.read_sdt("experimental_data.sdt")

# Perform a quick Phasor analysis
phasor_coords = pyfli.analysis.get_phasor(fli_data)

# Visualize results
pyfli.visualize.plot_phasor(phasor_coords)
```

## Repository & Issues

The source code is hosted on GitHub. Please report any bugs or feature requests via the issues tracker.
* **GitHub:** [https://github.com/vkp217/pyfli-pkg](https://github.com/vkp217/pyfli-pkg)


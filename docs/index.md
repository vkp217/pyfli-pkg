# pyfli

<p align="center">
  <img src="assets/logo.png" alt="PyFLI Logo" width="260"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/pyfli-lib/"><img src="https://img.shields.io/pypi/v/pyfli-lib.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
  <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/"><img src="https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg" alt="License"></a>
</p>

**pyfli** is a comprehensive library for **Fluorescence Lifetime Imaging (FLI)** data processing. It streamlines the workflow for handling diverse file formats from various hardware manufacturers and provides a standardized pipeline for both traditional analytical and deep-learning-based lifetime inference.

## Key Features

- **Universal processing pipeline** — simplifies handling of multiple FLI file types (ICCD, SPAD, TCSPC).
- **Enhanced FLI simulator** — a robust simulation engine adaptable to specific camera hardware parameters and noise models.
- **Standardized inference** — a unified interface for time-resolved microscopy and macroscopic FLI (MFLI) data.

## Supported Data Acquisition Methods

1. **ICCD** — Intensified Charge-Coupled Device cameras for fast-gated, wide-field imaging.
2. **SwissSPAD2 & SwissSPAD3** — high-speed SPAD (Single-Photon Avalanche Diode) architectures for high-resolution photon counting.
3. **SPCImage/TCSPC** — standardized processing for Time-Correlated Single Photon Counting microscopy data.

## Data Processing & Analysis

`pyfli` implements industry-standard analytical methods to extract lifetime information:

- **Non-linear Least Squares Fitting (NLSF)** — robust mathematical approach for exponential decay modeling.
- **Phasor Plot Analysis** — graphical, model-free transformation of fluorescence decay into a 2D polar plot for species separation.
- **Maximum Likelihood Estimation (MLE)** — statistical estimator optimized for low-photon regimes.
- **Rapid Lifetime Determination (RLD)** — computationally efficient method for real-time applications and high-frame-rate data.
- **Laguerre Method (LET)** — Laguerre Expansion Technique for model-free IRF deconvolution followed by multi-exponential lifetime extraction on a per-pixel basis.

## Where to go next

<div class="grid cards" markdown>

- **[Installation](getting-started/installation.md)** — get `pyfli` installed in your environment.
- **[Quickstart](getting-started/quickstart.md)** — load data and run your first fit in a few lines.
- **[API Reference](api/index.md)** — full reference for every public class and function.
- **[Citing pyfli](citing.md)** — how to cite this package and the phasor SEPL method.

</div>

## Repository & Issues

The source code is hosted on GitHub. Please report any bugs or feature requests via the issues tracker.

- **GitHub:** [github.com/vkp217/pyfli-pkg](https://github.com/vkp217/pyfli-pkg)
- **Contact:** [pyfli4lifetime@gmail.com](mailto:pyfli4lifetime@gmail.com)

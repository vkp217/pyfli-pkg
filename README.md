# pyfli: A Unified Platform for FLI Data Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

`pyfli` is a comprehensive library designed for **Fluorescence Lifetime Imaging (FLI)** data processing. It streamlines the workflow for handling diverse file formats from various hardware manufacturers and provides a standardized pipeline for both traditional analytical and deep-learning-based inference.

---

## Key Features

* **Universal Processing Pipeline:** Simplifies the handling of multiple FLI file types (ICCD, SPAD, TCSPC).
* **Enhanced FLI Simulator:** A robust simulation engine adaptable to specific camera hardware parameters and noise models.
* **Standardized Inference:** Unified interface for time-resolved microscopy and macroscopic FLI data.

## Supported Data Acquisition Methods

The platform provides native support for several high-end imaging systems:

1. **ICCD:** Intensified Charge-Coupled Device cameras for fast-gated, wide-field imaging.
2. **SwissSPAD2:** High-speed SPAD (Single-Photon Avalanche Diode) camera for high-resolution photon counting.
3. **SwissSPAD3:** Advanced SPAD architecture offering enhanced throughput and performance.
4. **SPCImage/TCSPC:** Standardized processing for Time-Correlated Single Photon Counting microscopy data.

## Data Processing & Analysis

`pyfli` implements a variety of industry-standard analytical methods:

### 1. Non-linear Least Squares (NLLS) Fitting
A robust mathematical approach for fitting exponential decay models by minimizing the sum of squared residuals between observed and theoretical decay curves.

### 2. Phasor Plot Analysis
A graphical, model-free transformation of fluorescence decay into a 2D polar plot. This simplifies the visualization of multi-exponential components and species separation.

### 3. Maximum Likelihood Estimation (MLE)
A statistical estimator that finds the lifetime parameters maximizing the likelihood of the observed photon counts, particularly effective for low-photon regimes.

### 4. Rapid Lifetime Determination (RLD)
A computationally efficient, single-shot method that estimates lifetimes using integrated intensity windows, ideal for real-time applications and high-frame-rate data.

---

## Installation

Since the project is currently in development, you can install it locally by cloning the repository and using `pip`:

```bash
git clone [https://github.com/your-username/pyfli.git](https://github.com/your-username/pyfli.git)
cd pyfli
pip install -e .
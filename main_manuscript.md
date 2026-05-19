---
title: 'pyfli: A Unified Python Platform for Fluorescence Lifetime Imaging Data Simulation, Processing and Visualization'
tags:
  - Python
  - fluorescence lifetime imaging
  - FLIM
  - FLI
  - TCSPC
  - SPAD
  - ICCD
  - phasor analysis
  - Laguerre deconvolution
  - GPU computing
  - biomedical optics
authors:
  - name: Vikas Pandey
    corresponding: true
    email: pandev2@rpi.edu
    affiliation: 1
affiliations:
  - name: Center for Modeling, Simulation and Imaging in Medicine, Rensselaer Polytechnic Institute, USA
    index: 1
bibliography: paper.bib
---

# Abstract

Fluorescence Lifetime Imaging (FLI) is a powerful quantitative imaging modality used across chemistry, biophysics, biomedical optics, and microscopy. It measures the nanosecond-scale decay of fluorescence emission and provides a quantitative, intensity-independent contrast for biological imaging useful from intracellular metabolic state to *in vivo* tumor characterisation. Despites its widespread use, the data acquisition and the analysis ecosystem remains fragmented: each detector family, Intensified Charge-Coupled Devices (ICCD), Single-Photon Avalanche Diode (SPAD) arrays, and Time-Correlated Single Photon Counting (TCSPC) microscopes, typically ships with its own proprietary file formats, vendor-specific software, and incompatible analytical conventions. This forces practitioners to maintain bespoke pipelines and limits reproducibility across hardware. I present **pyfli**, an open-source Python library that unifies FLI data ingestion, simulation, and lifetime estimation across detectors. The package implements five established analytical estimators, Non-linear Least Squares Fitting (NLSF), phasor analysis, Maximum Likelihood Estimation (MLE), Rapid Lifetime Determination (RLD), and Laguerre Expansion Technique (LET) for model-free Instrument Response Function (IRF) deconvolution, alongside CPU and GPU solvers, a configurable hardware-aware noise simulator. On top of that, this package extends to a single-pixel compressed-sensing reconstruction module for hyperspectral FLI as well. The library is distributed on PyPI as `pyfli-lib`, follows a modular registry-based architecture, and is intended to lower the activation energy for both routine FLI analysis and the development of new estimators.


# Statement of need

FLI is a mature quantitative imaging modality used widely in biophysics,
microscopy, and preclinical biomedical optics [@Becker2012; @Berezin2010].
Its hardware ecosystem, however, is heterogeneous. Intensified charge-coupled
device (ICCD) cameras dominate gated wide-field and macroscopic FLI;
SPAD arrays such as SwissSPAD2/3 provide megapixel-scale photon counting
[@Bruschini2019]; TCSPC microscopes record per-pixel arrival histograms in
scanning configurations. Each platform produces data with different geometry,
noise statistics, and file conventions.

The analytical landscape is equally fragmented. Non-linear least-squares
fitting dominates classical FLIM; phasor analysis [@Digman2008] is preferred
in lipid microscopy; maximum-likelihood estimation is used in the
photon-starved regime typical of *in vivo* imaging [@Kollner1992];
Laguerre-expansion techniques [@Jo2005] address model-free deconvolution; and
deep-learning approaches [@Smith2019] now offer real-time inference when
paired with calibrated simulators. These methods are typically scattered
across vendor software, academic prototypes, and disconnected community
packages, with inconsistent APIs and few common benchmarks.

`pyfli` was developed to (i) provide a uniform Python interface across ICCD,
SPAD, and TCSPC data; (ii) expose the major analytical estimators behind
consistent class APIs so that they can be compared on the same data;
(iii) provide matched CPU and GPU backends that allow the same estimator to
scale from a single decay to multi-megapixel cubes; and (iv) include a
hardware-aware simulator producing labelled synthetic data suitable for
training and validating deep-learning models. The library targets two
audiences: experimentalists who need a reliable, format-agnostic pipeline,
and method developers who need a common substrate for prototyping and
benchmarking new lifetime estimators.

# State of the field

Several open packages address parts of the FLI workflow. `FLUTE`
[@Gottlieb2023] and `PhasorPy` [@PhasorPy2024] provide phasor-based analysis
with strong visualisation; `FLIMfit` [@Warren2013] delivers a mature graphical
environment for least-squares FLIM fitting; `napari-flim-phasor-plotter`
integrates phasor analysis into the napari ecosystem. On the simulation side,
`flimlib` provides fast curve-fitting primitives but no end-to-end pipeline.
None of these packages simultaneously target (a) data ingestion across all
three major detector families, (b) multiple estimator families behind a
common API, (c) GPU-accelerated per-pixel fitting, and (d) a calibrated noise
simulator designed for generating deep-learning training data.

`pyfli` occupies this combined niche. To our knowledge it is the first
package that bundles least-squares, phasor, maximum-likelihood, rapid
lifetime determination, and Laguerre deconvolution behind a unified Python
interface with GPU support and an integrated simulator.

# Software design

`pyfli` is implemented in pure Python (≥3.11), comprises approximately 70
modules and 12,000 lines of code, and is organised into seven thematic
sub-packages exposed through a single top-level namespace:

- **`dataIO`** — universal data and IRF loaders for `.sdt`, `.mat`, `.tif`,
  `.npy`, `.txt`, and `.asc` files, with a detector-abstracted import path
  for ICCD, SwissSPAD2/3, and TCSPC outputs.
- **`dataCC`** — IRF alignment, normalisation, region-of-interest operations,
  background subtraction, hot-pixel rejection, and pile-up correction.
- **`analytical_methods`** — phasor analysis, non-linear least-squares
  fitting, Poisson maximum-likelihood estimation, and a Laguerre Expansion
  Technique (LET) fitter for model-free IRF deconvolution.
- **`solver`** — CPU and GPU lifetime processors built on NumPy/SciPy and
  PyTorch respectively, plus binned, global, and MLE variants and a fitting
  comparator.
- **`simulator`** — a forward-model engine, parameter distribution samplers,
  modular noise models (Poisson shot noise, dark count rate, Gaussian read
  noise, gain, quantisation), batch simulators, and a calibration/validation
  engine.
- **`spAnalysis`** — single-pixel compressed-sensing reconstruction for
  hyperspectral time-resolved imaging, with Hadamard and Fourier-DCT bases
  and linear, total-variation, and Poisson-likelihood reconstructors
  [@Pian2017].
- **`dataVnP`, `roiMaker`, `utils_common`** — visualisation, ROI editing,
  and shared utilities.

Three design decisions shape the package. First, **registry-based dispatch**
(loaders, noise models, reconstructors) is preferred over deep class
hierarchies, keeping each subsystem extensible without subclassing pressure.
Second, **CPU/GPU symmetry**: the `Fli_CPUProcessor` and `Fli_GPUProcessor`
classes accept the same fitter object and dataset and produce comparable
outputs, with the GPU path using differentiable reparameterisations
(`exp` for positive scalars, `sigmoid` for amplitude fractions, offset
parameterisation for ordered lifetimes) to enforce physical bounds without
boundary-handling pathologies common in box-constrained gradient methods.
Third, the **simulator closes the loop with experiment**: `FLICalibrator`
fits simulator hyperparameters against an experimentally acquired calibration
sample, so the synthetic data used to train downstream deep-learning models
matches the noise statistics of the target instrument.

Quality control is provided by a `pytest` suite covering parameter
validation, distribution sampling, noise model statistics, phasor coordinate
computation, and the Laguerre fitter. Tests use only synthetic NumPy arrays
and require no external data.

## Example Usage

A typical end-to-end workflow loads experimental data and the matched IRF, performs IRF alignment, and runs a multi-exponential fit:

```python
from pyfli import DataOperations, IRFAligner, FLIFitter, Fli_GPUProcessor

# 1. Unified data ingestion across vendor formats
loader = DataOperations(
    data_path="experiment.sdt",
    irf_path="irf.txt",
    bg_path="background.tif",
    mask_path="mask.png",
)
decay = loader.load_data(sub_bg=True, hot_pixel=True)
irf   = loader.load_irf()

# 2. Align IRF and define bi-exponential bounds
irf_aligned = IRFAligner(decay, irf).align()
bounds = [(0, np.inf),         # A1
          (0.1, 0.6),          # tau1 (ns)
          (0, np.inf),          # A2
          (0.6, 1.5),          # tau2 (ns)
          (0, np.inf)]         # offset

# 3. Fit on GPU
fitter   = FLIFitter(model="bi-exponential", p0=bounds, irf=irf_aligned)
processor = Fli_GPUProcessor(freq=(80, 40), fitter_class=fitter)
result    = processor.run(decay)
```

A complementary workflow uses phasor analysis for model-free quality control:

```python
from pyfli import PhasorAnalyzer
phasor = PhasorAnalyzer(decay, freq=80, irf=irf_aligned)
phasor.compute()
phasor.plot_universal_circle()
```

For supervised method development, the simulator generates calibrated synthetic data:

```python
from pyfli import FLIImageGenerator, Batch_sim
gen = FLIImageGenerator(irf=irf, detector="SPAD",
                        tau_range=(0.3, 3.0), photon_count=(1e2, 1e4))
synthetic_cube, ground_truth = gen.generate(n_images=1000)
```

---

## Quality Control

The package includes a pytest test suite covering parameter validation, distribution sampling, noise model statistics, phasor coordinate computation, and the Laguerre fitter. Tests are executed with `pytest -v --tb=short` and run on synthetic arrays so they are self-contained. Continuous testing across Python 3.11 and 3.12, the two officially supported interpreter versions, ensures backward compatibility within the supported window. Additional validation is performed empirically by comparing simulator outputs against experimentally calibrated noise statistics through the `FLIValidator` class, and by cross-checking lifetime estimates between analytical methods on the same dataset.

---

## Discussion and Future Work

Three areas are targeted for future development. First, formal deep-learning model integration: the package currently provides the data substrate and the `DLModelComparator` interface, but pre-trained models for ICCD, SPAD, and TCSPC will be packaged and benchmarked against the analytical baselines in a forthcoming release. Second, expansion of the supported vendor formats to include Becker & Hickl `.spc` raw photon streams and PicoQuant `.ptu` files, both of which are common in TCSPC microscopy. Third, an explicit FAIR-data export path (HDF5 with provenance metadata) so that downstream tools and reviewers can recover the exact processing pipeline applied to any cube.

The current release provides user a full felxibility in pyfli implementation. Mature python script writer can use it directly in their framework and non-programming background users can use this in an upcoming webapp.

**Limitations.** The license (CC BY-NC-ND 4.0) restricts commercial use and derivative redistribution, which is intentional for a community edition but should be considered by potential industrial adopters. Finally, the GPU path requires CUDA-capable hardware for full benefit, although the CPU path remains functionally equivalent.

---

# Research impact statement

`pyfli` consolidates analytical methods and hardware support that previously
required several disconnected codebases, lowering the activation energy for
laboratories adopting FLI as a quantitative readout. By exposing
least-squares, phasor, MLE, RLD, and Laguerre estimators behind a common API
on identical data, it enables direct method comparison — a step that is
needed for principled estimator selection but has historically been
discouraged by interface friction. The matched CPU/GPU backends additionally
allow the same pipeline to scale from a single calibration decay to
multi-megapixel preclinical FLI cubes, supporting both bench-scale
methodological work and high-throughput biomedical studies.

The integrated simulator and calibration engine are intended to support the
growing body of work on deep-learning FLI inference, where realistic, labelled
training data calibrated to specific detectors is a recurrent bottleneck
[@Smith2019]. By generating training corpora that match the noise statistics
of a target instrument, `pyfli` aims to make these methods more reproducible
across laboratories and detector platforms.

# AI usage disclosure

Portions of the documentation and this manuscript were drafted with
assistance from a large language model (Anthropic Claude, 2026), used for
prose drafting and structural editing. All scientific content, software
design decisions, code, and final wording were reviewed and approved by the
author. The AI was not used to generate experimental results, software
behaviour claims, or citations; references were verified by the author.

# Acknowledgements

I acknowledge Dr. Xavier Michalet for his extensive mathematical implementation of phasor, it was very helpful in drafting the extensive functionality 
of phasor.
I thank Prof. Xavier Intes and Stefan Radev for their constant feedback, help and during the development of this package.
I thank Prof. Margarida Barroso, Sherry Catherine, Naxue Yuan, Luis Chavez for providing the biological samples and FLI data from different imaging set up.
I thanks Ismail Erbas for the extensive implementation of various detectors, data acquisition and analysis feedback.

# References

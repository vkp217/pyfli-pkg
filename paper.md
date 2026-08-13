---
title: 'PyFLI: A Unified, Detector-Agnostic, and Open-Source Python Package for End-to-End Fluorescence Lifetime Imaging'
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
    orcid: 0000-0001-5477-1095
    corresponding: true
    email: pandev2@rpi.edu
    affiliation: 1
  - name: Ismail Erbas
    affiliation: 1
  - name: Margarida Barroso
    affiliation: 2
  - name: Xavier Intes
    affiliation: 1
  - name: Stefan Radev
    affiliation: 1
affiliations:
  - name: Center for Modeling, Simulation and Imaging in Medicine, Rensselaer Polytechnic Institute, USA
    index: 1
    ror: 01rtyzb94
  - name: Albany Medical College, USA
    index: 2
date: 12 August 2026
bibliography: paper.bib
---

# Summary

Fluorescence Lifetime Imaging (FLI) is a powerful, concentration-independent quantitative imaging modality used across chemistry, biophysics, biomedical optics, and microscopy. Because the fluorescence lifetime is set by the photophysics of a fluorophore and its local environment rather than by how many molecules are present, FLI separates genuine environmental contrast, such as pH, ion concentration, viscosity, oxygenation, and Förster resonance energy transfer (FRET), from intensity artifacts like photobleaching and uneven illumination, from intracellular metabolic state to *in vivo* tumor characterisation. Despite its widespread use, the data acquisition and analysis ecosystem remains fragmented: each detector family, Intensified Charge-Coupled Devices (ICCD), Single-Photon Avalanche Diode (SPAD) arrays, and Time-Correlated Single Photon Counting (TCSPC) microscopes, typically ships with its own proprietary file formats, vendor-specific software, and incompatible analytical conventions. This forces practitioners to maintain bespoke pipelines and limits reproducibility across hardware. We present **PyFli**, an open-source Python library that unifies FLI data ingestion, simulation, and lifetime estimation across detectors and all methods in one place. In this version, the package implements five established analytical estimators, Non-linear Least Squares Fitting (NLSF), phasor analysis, Maximum Likelihood Estimation (MLE), Rapid Lifetime Determination (RLD), and Laguerre Expansion Technique (LET) for model-free Instrument Response Function (IRF) deconvolution, behind consistent APIs, alongside matched CPU and GPU solvers, a detector-physics-aware deconvolution engine, and a configurable hardware-aware noise simulator with Cramér–Rao-bound analysis which will expand in future as per further development. 
Additionally, this package extends to a single-pixel compressed-sensing reconstruction module for hyperspectral FLI as well. The library is distributed on PyPI as `pyfli-lib`, follows a modular registry-based architecture, and is intended to lower the activation energy for both routine FLI analysis and the development of new estimators.

# Statement of need

FLI is a mature quantitative imaging modality used widely in biophysics, microscopy, and preclinical biomedical optics [@Becker2012; @Berezin2010]. A FLI measurement records, at every pixel, a histogram of photon arrival times relative to a pulsed excitation; the measured histogram is not the underlying decay but its convolution with the instrument response function (IRF), corrupted by detector-specific noise. Its hardware ecosystem is correspondingly heterogeneous. Intensified charge-coupled device (ICCD) cameras integrate wide-field signal within nanosecond gates and carry multiplicative gain noise from the intensifier stage; SPAD arrays such as SwissSPAD2/3 provide megapixel-scale gated photon counting governed by per-gate binomial detection and dead-time [@Bruschini2019]; TCSPC build per-pixel arrival-time histograms photon by photon with picosecond precision, and are subject to classical pile-up distortion at
high count rates. Each platform therefore demands a tailored forward model and produces data with different geometry, noise statistics, and file conventions.

The analytical landscape is equally fragmented. Iterative reconvolution fitting methods non-linear least squares and maximum-likelihood estimation fit explicit exponential models and yield interpretable lifetimes and amplitudes, but require an assumed number of components and good initialization [@Kollner1992]; phasor analysis [@Digman2008] maps each decay to a point in a two-dimensional plane without fitting, enabling fast, model-free, visually intuitive species separation, and is preferred in
lipid microscopy; Laguerre-expansion techniques [@Jo2005] deconvolve the IRF using an orthonormal basis without committing to a fixed multi-exponential form; rapid lifetime determination trades accuracy for the speed needed in real-time or high-throughput settings; and deep-learning approaches [@Smith2019] offer real-time inference when paired with calibrated simulators. 
These methods answer different scientific questions and have different failure modes, so being able to apply and compare them on identical data is valuable, yet in most workflows each method lives in a
separate package with its own input conventions, scattered across vendor software, academic prototypes, and disconnected community packages, with inconsistent APIs and few common benchmarks.

The consequence of this fragmentation is a proliferation of bespoke conversion and bridging scripts: researchers routinely export from one tool, reshape arrays by hand, re-implement a second method to cross-check the first, and maintain private code to read each instrument they use. This glue code is rarely shared, seldom tested, and a frequent source of silent error, which undermines both reproducibility and method comparison.

`PyFli` is developed to (i) provide a uniform Python interface across detectos (ICCD, SPAD, and TCSPC) data; (ii) expose the major analytical estimators behind consistent class APIs so that they can be compared on the same data; (iii) provide matched CPU and GPU backends that allow the same estimator to
scale from a single decay to multi-megapixel cubes; and (iv) include a hardware-aware simulator producing labelled synthetic data suitable for training and validating deep-learning models. 
The python-based `PyFli` library targets two audiences: (1) experimentalists who need a reliable, format-agnostic pipeline, and (2) method developers who need a common substrate for prototyping and
benchmarking new lifetime estimators.

# State of the field

Several open packages address parts of the FLI workflow. `FLUTE` [@Gottlieb2023] and `PhasorPy` [@PhasorPy2024] provide phasor-based analysis with strong visualisation; `FLIMfit` [@Warren2013] delivers a mature graphical environment for least-squares FLIM fitting; `napari-flim-phasor-plotter` integrates phasor analysis into the napari ecosystem. On the fitting based analysis side, `flimlib` provides fast curve-fitting primitives but no end-to-end pipeline. None of these packages simultaneously target (a) data ingestion across all three major detector families, (b) multiple estimator families behind a common API, (c) GPU-accelerated per-pixel fitting, and (d) a calibrated noise simulator designed for generating deep-learning training data. Where a code-free graphical interface excels at guiding a specific downstream
workflow, `PyFli` targets the upstream, method-rich, multi-detector
processing problem and is built to be scripted, batched, and extended; the
two are not mutually exclusive, since `PyFli` can serve as the processing
engine beneath a graphical front end.

`PyFli` occupies this combined niche. To our knowledge it is the first
package that bundles least-squares, phasor, maximum-likelihood, rapid
lifetime determination, and Laguerre deconvolution behind a unified Python
interface with GPU support, a detector-physics-aware deconvolution engine,
and an integrated simulator.

# Software design

`PyFli` is implemented in pure Python (≥3.11) and, following a major refactor after the 0.1.18 PyPI release, comprises 94 modules and approximately 32,000 lines of code, organised into more than a dozen thematic sub-packages exposed under a single top-level `pyfli` namespace (the package root re-exports the most commonly used entry points directly, while more specialized classes are imported from their owning sub-package):

- **`io`** — universal data and IRF loaders (`DataOperations`, `Detector`, `DataSaver`) for `.sdt`, `.mat`, `.tif`, `.npy`, `.txt`, `.asc`, and compressed Leica `.lif`, with a detector-abstracted import path for ICCD, SwissSPAD2/3, and TCSPC outputs.
- **`data_cc`** — IRF alignment (`IRFAligner`), normalisation (`Normalization`), preprocessing, and region-of-interest operations (`ROIOperations`).
- **`analytical_methods`** — the class-based `PhasorAnalyzer`, the `LaguerreFLI` Laguerre Expansion Technique (LET) fitter for model-free IRF deconvolution, and shared analytical helpers.
- **`phasor`** — a newer, acquisition-mode-aware phasor-geometry toolkit (functional API) built on the universal-circle formalism.
- **`irf_deconvolution`** — detector-specific (TCSPC/SPAD/ICCD) observation models and a joint IRF-deconvolution solver.
- **`solver`** — CPU (`FLICPUProcessor`) and GPU (`FLIGPUProcessor`, PyTorch-based) lifetime processors, the shared `BaseFLIFitter` (NLSF) and `MLEFLIFitter` (Poisson MLE) fitters, plus `GlobalFLIFitter`, `BinnedFLIFitter`/`FLIBinner`, and a `FittingComparator`.
- **`simulator`** — a photon-level forward-model engine, parameter distribution samplers, modular noise models (Poisson shot noise, dark count rate, Gaussian read noise, gain, quantisation, TCSPC pile-up), batch simulators, and `FLICalibrator`/`FLIValidator` calibration and validation engines.
- **`sp_analysis`** — single-pixel compressed-sensing reconstruction for hyperspectral time-resolved imaging, with Hadamard and DCT bases and linear, total-variation, and Poisson-likelihood reconstructors [@Pian2017].
- **`reconstruction`** — rebuilds modeled decay cubes and fit-quality maps from any fitted parameter map, downstream of `solver`.
- **`bayes_utils`** — a Bayesian/deep-learning direct-inference pathway that runs a trained posterior-sampling model over a decay image and reconciles per-pixel posterior samples against the measured decay.
- **`data_vnp`** — comparative visualisation, a deep-learning-vs-analytical comparator (`DLModelComparator`), and a mono/bi-exponential classifier (`MonoBiClassifier`).
- **`analysis`** — session-level post-processing: result loading, fit/phasor diagnostics plotting, hypothesis testing (`TestStat`), and factor analysis.
- **`roi_maker`** — an interactive, PySide6-based ROI editor (`ROIMaker`).
- **`data_text`, `logging`, `img`** — message display, structured logging, and package assets.

At the workflow level, this functionality is organised into three
interconnected modules (Figure 1): a physics-guided **simulator** that
couples parameter priors with an instrument response function under
detector-specific noise statistics to generate labelled synthetic data; a
**parameter-estimation** module that ingests experimental decays across
supported instrument formats and feeds them to the analytical estimators
described below; and a **data-visualization** suite for comparative
plotting, fitting diagnostics, and cross-method statistics on data processed
either internally or by external software.

![Overview of the tripartite `PyFli` package. **(a)** The Simulator implements a physics-guided forward model that combines parameter priors (lifetime, component fractions, photon counts) and an instrument response function with hardware-specific noise statistics (ICCD, TCSPC, SPAD) to generate simulated decay data. **(b)** The Parameter Estimation module ingests experimental decay, IRF, and optional mask data across supported vendor formats through a unified data-import layer, and passes them to a choice of analytical estimator (Phasor, NLSF, MLE, Laguerre deconvolution, RLD/CMM) and decay-model order, yielding estimated parameters and fitting statistics; the architecture is expandable to deep-learning models. **(c)** The Data Visualization suite renders phasor plots, lifetime maps, and fitting diagnostics from `PyFli`-processed data, and supports comparative plotting and statistics computation against data processed by other software.](fig1.png)

## Design principles

Five principles guide the design. First, **separation of detector from
method**: the noise physics of an instrument is encoded once, in the
ingestion and deconvolution layers, so that any estimator can run on any
supported detector without re-implementation; loaders, noise models, and
reconstructors are dispatched through a registry rather than a deep class
hierarchy, keeping each subsystem extensible without subclassing pressure.
Second, a **common decay contract**: all loaders emit arrays with a
consistent axis convention and accompanying metadata (time-bin spacing,
laser period, IRF, mask), so estimators never need to know the file's
origin. Third, **method interoperability**: the estimators share input
and result conventions, which makes cross-method comparison a first-class
operation rather than an afterthought. Fourth, **CPU/GPU symmetry**: the
`FLICPUProcessor` and `FLIGPUProcessor` classes accept the same fitter
class and dataset and produce comparable outputs, with the GPU path using
differentiable reparameterisations (`exp` for positive scalars, `sigmoid`
for amplitude fractions, offset parameterisation for ordered lifetimes) to
enforce physical bounds without the boundary-handling pathologies common in
box-constrained gradient methods, and it can additionally report per-pixel
Cramér–Rao uncertainty estimates during the fit itself. Fifth, **the
simulator closes the loop with experiment**: `FLICalibrator` fits simulator
hyperparameters against an experimentally acquired calibration sample, so
synthetic data used to train downstream deep-learning models matches the
noise statistics of the target instrument, and the API is scriptable and
deterministic so results carry the configuration that produced them.

## Detector-agnostic data ingestion

Ingestion is centered on a `Detector` class that dispatches to
detector-specific readers and a higher-level `DataOperations` interface that
orchestrates loading of decay data, IRF, background, and masks, including
parallel loading of large datasets. The supported readers are summarized in
Table 1.

**Table 1.** Detectors and formats natively supported by `PyFli`'s ingestion layer.

| Reader | Instrument / format | Acquisition physics |
|---|---|---|
| `SS2` | SwissSPAD2 SPAD array | Massively parallel gated single-photon counts; binomial per-gate detection |
| `SS3` | SwissSPAD3 SPAD array | Higher-resolution gated SPAD photon counting |
| `ICCD` | Intensified CCD camera | Wide-field nanosecond time-gating; multichannel-plate gain noise |
| `BH_TCSPC` | Becker \& Hickl TCSPC (`.sdt`) | Photon-by-photon arrival-time histogramming; pile-up at high rate |
| `generic` | TIFF / NPY / MAT / TXT / HDF5 | Format-only import for arbitrary pipelines |
| Leica LIF | Light-sheet FLIM (compressed) | Decodes reduced time-tagged photon streams to a decay cube |

A distinctive capability is native decoding of compressed Leica light-sheet
FLIM data: `PyFli` decompresses the "reduced Time Tagged" record stream
described in Leica's own patent filings (US20230344447A1 / US12278654B2) and
reconstructs a four-dimensional decay cube indexed by mosaic/frame, image
height, image width, and TCSPC histogram bin, while also exposing the
vendor-computed parameter maps for reference, with helper routines to
collapse the frame axis and inspect the result interactively. Becker \&
Hickl `.sdt` files are read through an integrated, established parsing
engine, whereas `.lif` decoding uses a custom internal layer written for
this package. ICCD and SPAD acquisitions are assembled from their gate or
frame stacks, and the generic reader covers TIFF, NumPy, MATLAB, text, and
HDF5 containers for pipelines that have already exported to a neutral
format. Background subtraction supports either a single file or the mean of
a folder of background acquisitions, and IRF and mask handling are
integrated so that downstream estimators receive a fully specified problem
context.

IRF preparation is handled by a dedicated alignment utility (`IRFAligner`)
that estimates the temporal offset between IRF and decay by
rising-edge detection and applies the shift either by sub-bin Fourier phase
shifting or by integer circular shifting, on a global (`align`) or per-pixel
(`align_pixel`) basis. Accurate IRF alignment is a precondition for unbiased
reconvolution and phasor calibration, and centralizing it ensures every
estimator starts from a consistently registered IRF.

## Analytical lifetime estimation

FLI decays are modeled as a sum of exponential components convolved with the
IRF. For a decay sampled at times $t_k$ with $m$ components, the noise-free
model is

$$\hat{y}(t) = b + \mathrm{IRF}_s(t) * \sum_{i=1}^{m} A_i \exp\left(-\frac{t}{\tau_i}\right),$$

where $A_i$ and $\tau_i$ are the amplitude and lifetime of component $i$,
$\mathrm{IRF}_s$ is the shifted instrument response, $*$ denotes
convolution, and $b$ is a constant background offset. The amplitude
fractions and the amplitude-weighted mean lifetime, the quantities most
often reported, follow as

$$\alpha_i = \frac{A_i}{\sum_j A_j}, \qquad \tau_m = \sum_i \alpha_i \tau_i.$$

`PyFli` exposes five complementary estimators for these quantities,
summarized in Table 2.

**Table 2.** Complementary lifetime estimators in `PyFli`.

| Estimator | Model assumption | Core strength | Typical use case |
|---|---|---|---|
| `NLSF` | Fixed $m$-exponential + IRF | Interpretable amplitudes and lifetimes | General-purpose, moderate–high photons |
| `MLE` | Fixed $m$-exponential + IRF | Unbiased under shot noise | Low-photon, fast acquisition |
| `Phasor` | Model-free | Fast, visual species separation | Screening, heterogeneity mapping |
| `LET` | Basis expansion, no fixed $m$ | Robust to unknown decay form | Complex / unknown decays |
| `RLD` | Window ratios | Lowest per-pixel cost | Real-time, high-throughput |

### Non-linear least-squares fitting (NLSF)

`BaseFLIFitter` (`pyfli.solver`) and the broader solver framework recover
parameters by minimizing the weighted sum of squared residuals between the
reconvolved model and the measured decay, optionally restricted to a
user-defined time gate:

$$\chi^2(\theta) = \sum_k w_k \left(y_k - \hat{y}_k(\theta)\right)^2.$$

Minimization uses `scipy.optimize`'s bounded and unconstrained least-squares
backends, with analytic parameter uncertainties derived from the covariance
of the converged fit and a moment-based (or user-supplied) initial-guess
plugin. The fitter operates per pixel and across whole images, with both CPU
(`FLICPUProcessor`) and GPU (`FLIGPUProcessor`) execution paths. Model-selection
helpers compare mono- versus bi-exponential hypotheses, and convenience
functions in `shared_metrics` return amplitude-weighted mean lifetimes and
FRET efficiencies directly.

### Maximum-likelihood (Poisson) estimation

Photon-counting noise is Poisson, not Gaussian, and least squares becomes
biased in the low-photon regime typical of fast or photon-starved
acquisitions. `MLEFLIFitter` (`pyfli.solver`), a subclass of `BaseFLIFitter`,
minimizes the negative Poisson log-likelihood, the statistically efficient
choice for shot-noise-limited data:

$$-\ln L(\theta) = \sum_k \left[\hat{y}_k(\theta) - y_k \ln \hat{y}_k(\theta)\right].$$

Both single-pixel and whole-image fitting are supported on CPU and GPU (the
`FLIGPUProcessor.fit_image` entry point selects NLSF or MLE via a `mode`
argument, and can report per-pixel Cramér–Rao uncertainty when `CRLB=True`),
and the MLE estimator shares the same model-comparison machinery as NLSF
(`FittingComparator`) so the two can be applied to identical data and
reconciled.

### Phasor analysis

The phasor transform provides a fit-free view of lifetimes by mapping each
decay to a point $(g, s)$ via the cosine and sine Fourier components at
harmonic $n$ of the laser angular frequency $\omega$:

$$g(n) = \frac{\sum_k y_k \cos(n\omega t_k)}{\sum_k y_k}, \qquad s(n) = \frac{\sum_k y_k \sin(n\omega t_k)}{\sum_k y_k}.$$

`PyFli` provides two complementary phasor implementations. The class-based
`PhasorAnalyzer` (`pyfli.analytical_methods`) computes phasors on CPU or GPU
(`create_phasor_cpu`/`create_phasor_gpu`), calibrates them against an
aligned IRF (`calibrate`), and derives phase and modulation lifetimes:

$$\tau_\phi = \frac{1}{n\omega}\left(\frac{s}{g}\right), \qquad \tau_M = \frac{1}{n\omega}\sqrt{\frac{1}{g^2+s^2}-1}.$$

A companion plotting mixin renders phasor diagrams, lifetime-colored phasor
maps, harmonic comparisons, and per-pixel fit overlays, and results are
serializable to HDF5 for archival.

Standard phasor formulas assume an ideal continuous decay sampled over a
full period, but real acquisitions are binned, gated, truncated, or offset,
each of which alters the phasor geometry. A second, newer top-level `phasor`
sub-package (re-exported directly from `pyfli`) implements the
acquisition-mode-aware universal-circle formalism of Michalet (2021), with
closed-form single-exponential-phasor-locus (SEPL) expressions for
continuous decays, decays sampled into discrete bins, single and multiple
square gates of finite width, decays recorded over only part of the period,
and decays offset by an IRF or pulse delay. An `AcquisitionConfig` object
captures frequency, gate width, recording window, and offset, and
`phasor_from_config` dispatches to the appropriate formula, letting users
place gated or truncated phasor data on a geometrically correct backbone —
important for ICCD and SPAD systems, whose gating departs significantly from
the ideal continuous case.

### Laguerre expansion technique (LET)

LET deconvolves the IRF without assuming a fixed number of exponentials by
expanding the impulse response on a discrete orthonormal Laguerre basis.
The `LaguerreFLI` class (`pyfli.analytical_methods`, also re-exported at the
package root) constructs the basis from the recurrence

$$b_j(n) = \sqrt{\alpha}\, b_j(n-1) + \sqrt{\alpha}\, b_{j-1}(n) - b_{j-1}(n-1),$$

where $\alpha \in (0,1)$ is the scale parameter controlling how quickly the
basis functions decay and $b_j(n)$ is the $j$-th basis function at sample
$n$. Each measured decay is modeled as a non-negative combination of
IRF-convolved basis functions, and the expansion coefficients are recovered
per pixel by non-negative or ordinary least squares, with an optional safe
fallback for ill-conditioned pixels. The scale parameter can be fixed or
optimized automatically, and optional smoothness regularization is
supported. Because the basis spans a continuum of decay shapes rather than
a discrete component count, LET is robust for complex or unknown decay
forms; lifetimes and fractions are then extracted from the reconstructed
impulse response, and correctness of the closed-form recurrence is enforced
by a dedicated regression test suite.

### Rapid lifetime determination (RLD)

For real-time and high-throughput settings, `PyFli`'s solver framework
includes an `rld_based_guess` estimator (`pyfli.solver.base_static`) that
derives an effective lifetime in closed form from ratios of integrated
signal over a small number of time windows. RLD sacrifices the resolution of
multi-component fitting for a per-pixel cost low enough to keep pace with
high-frame-rate acquisition, and is used as a fast initial-guess plugin that
seeds `BaseFLIFitter`/`MLEFLIFitter`, or can be called directly for a
first-pass lifetime map ahead of full fitting.

## Detector-specific deconvolution engine

A dedicated sub-package, `irf_deconvolution`, makes the detector's noise
physics explicit rather than treating every measurement as Gaussian. Its
`detector_weights` module maps observed counts to an underlying intensity
through a detector-specific observation model (`TCSPCParams`, `SPADParams`,
`ICCDParams`, `make_observation`) and assigns each time bin a statistically
motivated weight. For **TCSPC**, it applies a Coates-type correction that
recovers the true intensity $\Lambda$ from the recorded counts $N$ and
excitation rate $n_{ex}$,

$$\Lambda = -n_{ex} \ln\left(1 - \frac{N}{n_{ex}}\right),$$

with weight $w = \left[\Lambda \cdot n_{ex} / (n_{ex} - N)\right]^{-1}$
inflated to reflect the reduced information of piled-up bins. For **SPAD**
detection, which is binomial per gate, it inverts the analogous binomial
relation to recover intensity and weights bins by the corresponding binomial
variance, the correct treatment for SwissSPAD-class arrays. For **ICCD**,
the intensifier introduces multiplicative gain noise on top of photon
statistics, so it uses a compound Poisson–Gaussian model with a
multichannel-plate excess-noise factor ($F^2 \approx 2$) and a
`generalized_anscombe` variance-stabilizing transform so the stabilized data
can be treated with the same least-squares core.

On top of the observation model, the `fli_solver` module (`solve_flim`,
`SolverConfig`) solves for decay parameters under shared regularization:
Tikhonov damping for stability and a total-variation penalty that suppresses
noise while preserving edges between regions of differing lifetime, built on
a gate matrix and cyclic-convolution decay basis (`build_gate_matrix`,
`decay_basis`, `cyclic_conv`) so the IRF and decay model can be solved
jointly. The result is an estimator matched to the instrument rather than
forced into a one-size-fits-all Gaussian assumption.

## Advanced solver framework

Beyond the per-pixel estimators, `PyFli`'s `solver` sub-package provides a
structured framework for whole-image and spatially aware fitting.
`BaseFLIFitter` abstracts the choice of estimator, fit range, uncertainty
calculation, and model comparison; specialized subclasses add capabilities
that matter at scale. `GlobalFLIFitter` clusters spatially similar pixels
into super-pixels, fits representative cluster decays, and stitches the
results back to full resolution, sharing statistical strength across pixels
in photon-starved regions. `FLIBinner`/`BinnedFLIFitter` apply configurable
spatial binning before fitting to trade resolution for signal. Derived
quantities such as amplitude-weighted mean lifetime and FRET efficiency are
computed directly (`shared_metrics`), and `FLICPUProcessor.process_image`
and `FLIGPUProcessor.fit_image` apply any configured fitter class across an
entire image — the latter optionally returning per-pixel Cramér–Rao
uncertainty alongside the fit — letting users move from a quick per-pixel
map to a carefully regularized analysis without changing tools.

## Compressive single-pixel SPAD imaging

The `sp_analysis` sub-package supports single-pixel and compressive FLI, in
which a scene is sampled through a sequence of spatial patterns (for
example, on a digital micromirror device) and reconstructed computationally,
with time-resolved single-photon detection. It supplies orthogonal sensing
bases, `HadamardBasis` and `DCTBasis`, and a `MeasurementSimulator`, and
reconstructs a four-dimensional cube indexed by the two spatial coordinates,
the temporal decay axis, and an optional spectral axis [@Pian2017]. Three
reconstructors are provided: `LinearReconstructor`, a back-projection
(ghost-imaging) solver; `TVReconstructor`, an isotropic total-variation
solver for Gaussian-noise regimes; and `SPADPoissonReconstructor`, a
Poisson-likelihood total-variation solver matched to photon-counting SPAD
statistics.

## Decay reconstruction and fit diagnostics

A separate `reconstruction` sub-package closes the loop between a fit and
the data it was fit to. `ParameterToDecayReconstruction` rebuilds a modeled
decay cube directly from any fitted parameter map, independent of which
estimator produced it, so the reconvolved model can be compared pixel-by-
pixel against the original measurement as an external goodness-of-fit check
downstream of the solver.

## Physics-based simulation and Cramér–Rao analysis

Validation of any lifetime estimator requires data whose true parameters are
known, which only simulation can guarantee. At the most detailed level, a
photon-by-photon Monte-Carlo TCSPC simulator (`TCSPCSimulator`) draws
individual photon arrival times from the convolved decay, reproducing
genuine counting statistics rather than adding Gaussian noise to an analytic
curve; a matched `MacroSimulator` targets wide-field ICCD-style acquisition,
and `ContinuousSimulator`/`PhotonCountSimulator` cover additional sampling
regimes. A `NoiseEngine` applies Poisson photon noise, dark-count rate, read
noise, timing jitter, and TCSPC pile-up so synthetic data can be made to
resemble a specific instrument. Higher-level generators (`FLIImageGenerator`,
`FLIModelImageGenerator`) produce spatially heterogeneous lifetime fields —
one simulator configuration per labeled ROI region — with intensity masking
and ROI structure, and `BatchSimulator` builds batches across parameter
sweeps for training-data generation.

Precision is characterized two ways. The simulation engine computes the
Fisher information and associated Cramér–Rao lower bound (CRLB) for the
lifetime parameters, the theoretical best achievable precision given the
photon budget and acquisition settings,
$\mathrm{CRLB}(\theta_i) \ge [I(\theta)]^{-1}_{ii}$; and `FLIGPUProcessor.fit_image`
can report the same per-pixel CRLB directly during a real fit
(`CRLB=True`), not only in simulation. This lets users ask not only whether
an estimator recovers the truth on average but whether it approaches the
information-theoretic limit, and to determine in advance how many photons an
experiment requires. `FLICalibrator` and `FLIValidator` close the loop
between simulation and the analytical methods by fitting simulator
hyperparameters to, and validating simulator output against, an
experimentally acquired calibration sample.

## Bayesian and deep-learning direct inference

The `bayes_utils` sub-package provides a direct-inference pathway that
complements the analytical estimators. `BiPipeline` runs a trained
posterior-sampling model (built on BayesFlow/Keras, installed via the
optional `pyfli-lib[tf]` extra) over a decay image to produce per-pixel
posterior samples of the lifetime parameters; `BestParamFitSelector` then
selects and aggregates the posterior-sample combination that best matches
the measured decay for each pixel, and `plot_pixel_posterior_fit`
visualizes a single pixel's posterior-predictive fit against the
measurement. This gives `PyFli` a route to fast, uncertainty-aware
inference once a posterior model has been trained (for example, on data
from the physics-based simulator described above), alongside the
model-agnostic analytical estimators.

## Comparison, classification, and visualization

Because `PyFli`'s estimators share result conventions, the `data_vnp` and
`analysis` sub-packages treat cross-method comparison as a primary task. A
multi-source plotting framework (`Plotter`, `PlotKit`, `SubplotVisualizer`,
`plot_2d_subplots`) loads results from several methods or datasets through a
unified ingestion layer and renders them side by side as spatial parameter
maps, histograms, and other comparative views through a single declarative
interface. `DLModelComparator` adds distribution-distance metrics, originally
built to compare deep-learning inference (including output from `bayes_utils`)
against analytical references, so an inferred lifetime map can be
quantitatively scored against a fitted one.

A mono/bi-exponential classifier (`MonoBiClassifier`, with
`ParamCorrelationMatrix` for cross-method correlation) labels each pixel as
better described by a one- or two-component model and measures agreement
between methods, reporting pairwise overlap and cross-method parameter
correlation. The `analysis` sub-package wraps this together with session-level
result loading, fit and phasor diagnostics (`plot_fitting_maps`,
`plot_diagnostics`, `plot_pixel_evidence`, `plot_statistical_comparison`,
`run_mono_bi_classifier`), and a `FactorAnalysis` tool, turning the question
"do my methods agree, and where do they disagree?" into a few function
calls, on data processed both internally and by external software.

## Interactive region-of-interest tooling

Single-cell and region-level analyses depend on accurate masks. The
`roi_maker` sub-package ships an interactive ROI editor built on PySide6
(`ROIMaker(intensity_2d, save_path=...)`) that lets users draw ROIs over an
intensity image via `.draw()`, then export them with `.save_masks()`,
`.get_multi_cluster_mask()`, or `.get_binary_mask()`; it also creates masks
automatically by intensity thresholding or clustering. These masks integrate
directly with the ingestion layer and the solver framework so that
region-summed decays, which substantially improve fitting precision
relative to single pixels, can be analyzed without leaving the library.

## Statistical testing for lifetime comparison

Quantitative comparison of lifetime distributions, between treatments,
between methods, or against simulated truth, requires appropriate statistics
rather than visual inspection alone. The `analysis` sub-package's `TestStat`
class complements the visualization layer with hypothesis tests and
effect-size measures suited to the often non-normal, heavy-tailed
distributions of pixel- and cell-level lifetimes. These tests are wired into
the comparison framework so that a difference highlighted in a plot can be
accompanied by a quantitative significance and effect-size estimate, and
into the simulator so an estimator's recovery can be tested against ground
truth across many synthetic realizations.

## Implementation and availability

`PyFli` builds on the scientific Python ecosystem, including NumPy and SciPy
for numerics, scikit-image and OpenCV for image operations, scikit-learn for
clustering and classification, pandas for tabular results, Matplotlib and
seaborn for plotting, PyTorch for tensor and accelerated computation,
`sdtfile` and `tifffile` for instrument I/O, and PySide6 for the interactive
ROI editor; these are core, required dependencies. Two optional extras
separate accelerated execution from deep-learning inference: `pyfli-lib[gpu]`
installs the CUDA runtime packages used by the PyTorch-based GPU solver, and
`pyfli-lib[tf]` installs TensorFlow/Keras for the `bayes_utils` Bayesian
direct-inference pipeline. Linting and formatting are enforced with `ruff`
via `pre-commit` hooks, checked in CI on every push and pull request against
`main`/`dev` alongside the test suite. The package is released on PyPI as
`pyfli-lib`, developed openly on GitHub with a `dev`-branch contribution
workflow, and documented on a versioned Sphinx documentation site, available
at [pyfli.org](https://pyfli.org).

## Example Usage

A typical end-to-end workflow loads experimental data and the matched IRF, performs IRF alignment, and runs a multi-exponential fit:

```python
from pyfli import DataOperations
from pyfli.data_cc import IRFAligner
from pyfli.solver import BaseFLIFitter, FLIGPUProcessor

# 1. Unified data ingestion across vendor formats
loader = DataOperations(
    data_path="experiment.sdt",
    irf_path="irf.txt",
    bg_path="background.tif",
    mask_path="mask.png",
)
decay = loader.load_data(sub_bg=True, hot_pixel=True)
irf   = loader.load_irf()

# 2. Align IRF and decay
irf_aligned = IRFAligner(decay, irf).align()

# 3. Fit a bi-exponential model on GPU, with per-pixel CRLB uncertainty
processor = FLIGPUProcessor(freq=(80, 40), fitter_class=BaseFLIFitter)
result = processor.fit_image(
    decay, irf_aligned, mode="NLSF", model_type="bi-exponential", CRLB=True,
)
```

A complementary workflow uses phasor analysis for model-free quality control:

```python
from pyfli.analytical_methods import PhasorAnalyzer

phasor = PhasorAnalyzer(frequency_hz=80e6, time_axis_ns=time_axis, n_harmonics=1)
g, s = phasor.create_phasor_cpu(decay)
g_cal, s_cal = phasor.calibrate(g, s, irf_aligned)
```

For supervised method development, the simulator generates calibrated synthetic data:

```python
from pyfli import FLIModelImageGenerator

gen = FLIModelImageGenerator(irf_data=irf, image_shape=(64, 64), method="ICCD")
synthetic = gen.generate_image()  # decay/IRF/fit cubes + ground-truth parameter maps
```

A model-free cross-check pairs Laguerre deconvolution with the phasor transform on the same data:

```python
from pyfli import LaguerreFLI
from pyfli.analytical_methods import PhasorAnalyzer

lag = LaguerreFLI(n_components=2, dt=0.048, auto_alpha=True)
lag.fit(decay, irf)
tau_map = lag.tau_mean_

ph = PhasorAnalyzer(frequency_hz=80e6, time_axis_ns=time_axis, n_harmonics=1)
g, s = ph.create_phasor_cpu(decay)
```

# Quality Control

The package includes a pytest test suite, spanning 19 test modules, covering parameter validation, distribution sampling, noise model statistics, the CPU and GPU solvers, the binned fitter and fitting comparator, phasor coordinate computation (both the class-based and top-level phasor APIs), the Laguerre fitter, decay reconstruction, factor analysis, and the Bayesian inference pathway (posterior-pixel plotting and parameter-combination selection). Tests are executed with `pytest -v --tb=short` and run on synthetic arrays so they are self-contained. Continuous integration runs the suite on Ubuntu, Windows, and macOS for Python 3.11 on every push and pull request against `main`/`dev`; Python 3.12 is declared as a supported interpreter and is planned for addition to the CI matrix. A separate CI job enforces `ruff` linting and formatting via `pre-commit`. Additional validation is performed empirically by comparing simulator outputs against experimentally calibrated noise statistics through the `FLIValidator` class, and by cross-checking lifetime estimates between analytical methods on the same dataset.

# Discussion and Future Work

Four capabilities are unusual among existing FLI tools. First, the breadth
of native detector support, particularly the inclusion of SPAD arrays and
compressed light-sheet data alongside conventional TCSPC and ICCD, within
one ingestion layer. Second, the detector-physics-aware deconvolution
engine, which applies pile-up, dead-time, and multichannel-plate
excess-noise models rather than a universal Gaussian assumption, under
shared edge-preserving regularization. Third, the integration of a
compressive single-pixel SPAD reconstruction module and a
Cramér–Rao-aware Monte-Carlo simulator into the same package as the
conventional estimators, which makes information-theoretic benchmarking and
emerging photon-efficient architectures first-class rather than external
concerns. Fourth, a Bayesian direct-inference pathway (`bayes_utils`) that
runs trained posterior-sampling models over decay images and reconciles
per-pixel posterior samples against the measured decay, paired with a
dedicated reconstruction module that rebuilds modeled decay cubes from any
fitted parameter map for independent goodness-of-fit auditing — bringing
deep-learning inference and fit-quality diagnostics into the same package as
the analytical estimators, rather than treating them as external,
disconnected concerns.

Several limitations remain. The detector-specific noise models, though
grounded in the physics of pile-up, dead-time, and intensifier gain, rely on
accurate instrument parameters that the user must supply or calibrate;
mis-specified parameters will bias the correction. Full per-pixel fitting of
large images remains computationally demanding despite the GPU paths and the
super-pixel and binning strategies. The library is programmatic and assumes
familiarity with Python, which raises the barrier for non-programming users
relative to a graphical tool; the current release aims to give mature
Python users full flexibility to use `PyFli` directly within their own
frameworks, while non-programming users are intended to be served by an
upcoming web application. Time-lapse FLI with object tracking, and joint
multi-dataset modeling beyond the current comparison layer, are not yet
supported. The license (CC BY-NC-ND 4.0) restricts commercial use and
derivative redistribution, which is intentional for a community edition but
should be considered by potential industrial adopters, and the GPU path
requires CUDA-capable hardware for full benefit, although the CPU path
remains functionally equivalent. Finally, while the validation framework
against simulated ground truth is built in, broad benchmarking against a
wide range of commercial packages across many real samples is an ongoing
community effort that `PyFli`'s open, scriptable design is intended to
facilitate.

Three areas are targeted for further development. First, packaged,
pre-trained, and cross-detector-benchmarked posterior models for ICCD, SPAD,
and TCSPC: the `bayes_utils` direct-inference pathway and the
`DLModelComparator` interface already provide the infrastructure, but
shipping ready-to-use trained models, benchmarked against the analytical
baselines through `DLModelComparator`, remains ongoing work. Second,
expansion of the supported vendor formats to include Becker \& Hickl `.spc`
raw photon streams and PicoQuant `.ptu` files, both common in TCSPC
microscopy, alongside frequency-domain acquisition support. Third,
provenance metadata in the existing HDF5 result outputs (already used
throughout the solver, Laguerre, and phasor modules) so downstream tools and
reviewers can recover the exact processing pipeline applied to any cube,
alongside time-lapse handling with object tracking and further expansion of
the phasor-geometry toolkit. Because each subsystem is exposed through a
small, well-defined interface, these additions can be made without
disturbing existing workflows.

# Research impact statement

`PyFli` consolidates analytical methods and hardware support that previously
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
of a target instrument, `PyFli` aims to make these methods more reproducible
across laboratories and detector platforms.

# Author contributions

Vikas Pandey conceptualized the project, developed the entire codebase, and serves as the lead maintainer. Ismail Erbas contributed
supporting methods and tested the code. Xavier Intes and Margarida Barroso
provided funding and development support and facilitated biological sample
acquisition in their respective laboratories. Stefan Radev assisted with
repository organization, formatting, and standardization.

# AI usage disclosure

Portions of the documentation were drafted with
assistance from a large language model, and also used for prose drafting and structural editing. It was also used to add comments and documentation within the codebase itself. All scientific content, software design decisions, code, and final wording were reviewed and
approved by the author. The AI was not used to generate experimental results, software behaviour claims, or citations; references were verified
by the author.

# Acknowledgements

I acknowledge Dr. Xavier Michalet for his extensive mathematical implementation of phasor, it was very helpful in drafting the extensive functionality 
of phasor.
I thank Sherry Catherine, Naxue Yuan, Luis Chavez for providing the biological samples and FLI data from different imaging set up.

# References

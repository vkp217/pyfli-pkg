"""
Provide Bayesian-inference tooling for PyFLI.

This module belongs to :mod:`pyfli.bayes_utils` and covers the direct-inference
(BayesFlow/Keras) side of FLIM decay fitting, downstream of :mod:`pyfli.solver`
and :mod:`pyfli.reconstruction`: running a trained posterior-sampling model
over a decay image (:class:`BiPipeline`), selecting/aggregating per-pixel
posterior-sample parameter combinations against measured decay
(:class:`BestParamFitSelector`), and visualizing a single pixel's posterior
predictive fit (:func:`plot_pixel_posterior_fit`).
"""

from .inference import BiPipeline
from .param_combinations import BestParamFitSelector
from .posterior_pixel_plot import plot_pixel_posterior_fit

__all__ = ["BiPipeline", "BestParamFitSelector", "plot_pixel_posterior_fit"]

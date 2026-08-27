"""
Provide FLI decay-reconstruction tools for PyFLI.

This module belongs to :mod:`pyfli.reconstruction` and rebuilds modeled decay
cubes and fit-quality maps from fitted parameter maps, downstream of
:mod:`pyfli.solver`.
"""

__all__ = [
    "DetailedRecon",
    "ParamToDecay",
]


def __getattr__(name):
    if name == "ParamToDecay":
        from .decay_reconstruction import ParamToDecay

        return ParamToDecay
    if name == "DetailedRecon":
        from .detailed_results import DetailedRecon

        return DetailedRecon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""
Smoke tests — verify that all public symbols from pyfli are importable
and have the expected type.  These tests catch accidental breakage of
the public API without requiring any data files.
"""

import pytest
import pyfli


# ─────────────────────────────────────────────────────────────────────────────
# Package metadata
# ─────────────────────────────────────────────────────────────────────────────

def test_version_string_exists():
    assert hasattr(pyfli, "__version__")
    assert isinstance(pyfli.__version__, str)
    assert len(pyfli.__version__) > 0


def test_all_is_defined():
    assert hasattr(pyfli, "__all__")
    assert len(pyfli.__all__) > 0


# ─────────────────────────────────────────────────────────────────────────────
# All public symbols are importable
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("symbol", pyfli.__all__)
def test_symbol_accessible(symbol):
    assert hasattr(pyfli, symbol), f"pyfli.{symbol} not found"


# ─────────────────────────────────────────────────────────────────────────────
# Key classes are callable (not None, not a plain module)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("class_name", [
    "LaguerreFLI",
    "FLIFitter",
    "PoissonLikelihoodFitter",
    "PhasorAnalyzer",
    "Normalization",
    "DataSaver",
])
def test_class_is_callable(class_name):
    obj = getattr(pyfli, class_name)
    assert callable(obj), f"pyfli.{class_name} is not callable"


# ─────────────────────────────────────────────────────────────────────────────
# Simulator factory functions are callable
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fn_name", ["Macro_sim", "TCSPC_sim"])
def test_simulator_factories_callable(fn_name):
    obj = getattr(pyfli, fn_name)
    assert callable(obj)


# ─────────────────────────────────────────────────────────────────────────────
# SPAnalysis exports
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("symbol", ["BasisPatterns", "MeasurementSimulator", "Reconstructor"])
def test_spanalysis_symbols(symbol):
    assert hasattr(pyfli, symbol)
    assert callable(getattr(pyfli, symbol))


# ─────────────────────────────────────────────────────────────────────────────
# LaguerreFLI — basic instantiation without fitting
# ─────────────────────────────────────────────────────────────────────────────

def test_laguerre_instantiation():
    m = pyfli.LaguerreFLI(n_components=2, alpha=0.85, dt=0.05)
    assert m is not None
    assert m.n_components == 2


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions are callable
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fn_name", [
    "recovery_plot",
    "random_true_pixel",
    "data_masking",
    "save_plot",
    "load_flim_data",
    "collapse_to_xyt",
    "plot_xyt",
])
def test_utility_functions_callable(fn_name):
    fn = getattr(pyfli, fn_name)
    assert callable(fn)

"""Smoke tests for the curated top-level pyfli API."""

import pyfli


def test_no_manual_version_or_all_exports():
    assert not hasattr(pyfli, "__version__")
    assert not hasattr(pyfli, "__all__")


def test_top_level_core_symbols_are_public():
    expected = [
        "DataOperations",
        "Detector",
        "LaguerreFLI",
        "Normalization",
        "DataSaver",
        "Macro_sim",
        "TCSPC_sim",
        "BasisPatterns",
        "MeasurementSimulator",
        "Reconstructor",
    ]
    for name in expected:
        assert callable(getattr(pyfli, name))


def test_top_level_phasor_helpers_are_public():
    expected = [
        "AcquisitionConfig",
        "AcquisitionMode",
        "phasor_continuous",
        "phasor_discrete",
        "phasor_gated_single",
        "phasor_gated_N",
        "phasor_truncated",
        "phasor_offset",
        "phasor_from_config",
        "build_locus",
        "build_loci",
        "tau_grid",
        "universal_semicircle",
        "sepl_center_radius_discrete",
        "phase_lifetime",
        "modulus_lifetime",
        "lifetime_from_phasor",
        "phase_lifetime_gated",
        "fractional_components",
        "plot_phasor",
        "plot_locus_comparison",
        "plot_discrete_N_sweep",
    ]
    for name in expected:
        assert hasattr(pyfli, name)


def test_subpackages_available():
    assert pyfli.phasor is not None
    assert pyfli.spAnalysis is not None


def test_laguerre_instantiation():
    model = pyfli.LaguerreFLI(n_components=2, alpha=0.85, dt=0.05)
    assert model.n_components == 2


def test_flim_decay_cube_helpers_are_public():
    for name in ["load_flim_data", "collapse_to_xyt", "plot_xyt"]:
        assert callable(getattr(pyfli, name))

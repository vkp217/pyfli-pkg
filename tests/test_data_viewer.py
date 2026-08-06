"""Tests for pyfli.data_vnp.DataViewer."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pyfli.data_vnp import DataViewer


@pytest.fixture
def single_pixel_sim_result():
    """Mimic the dict shape returned by MacroSimulator/TCSPCSimulator.__call__."""
    t = np.linspace(0, 1, 64)
    decay = np.exp(-t / 0.3) + 0.01
    fit = np.exp(-t / 0.3)
    return {
        "raw_data": {"decay": decay, "irf": np.exp(-((t - 0.05) ** 2) / 0.0005)},
        "results": {
            "maps": {"tau1_map": 0.3, "tau2_map": 1.1, "alpha1_map": 0.7},
            "TR_maps": {"fit_map": fit, "residual_map": decay - fit},
        },
    }


def test_plot_pyfli_fit_summary_reads_nested_tr_maps(single_pixel_sim_result):
    """Regression test: TR_maps lives under results, key is residual_map (singular),
    matching every simulator/solver producer in the package."""
    fig, axes = DataViewer().plot_pyfli_fit_summary(
        single_pixel_sim_result, pixel=None, title="Test Summary"
    )
    assert fig is not None
    assert len(axes) == 4
    plt.close(fig)


def test_plot_pyfli_fit_summary_missing_nested_tr_maps_raises(single_pixel_sim_result):
    """The old top-level/plural key shape should no longer be accepted."""
    bad_data = {
        "raw_data": single_pixel_sim_result["raw_data"],
        "results": single_pixel_sim_result["results"],
        "TR_maps": {"fit_map": None, "residuals_map": None},
    }
    del bad_data["results"]["TR_maps"]
    with pytest.raises(KeyError):
        DataViewer().plot_pyfli_fit_summary(bad_data, pixel=None)

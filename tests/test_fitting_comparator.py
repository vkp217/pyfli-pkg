"""Tests for pyfli.solver.FittingComparator table printing and save_results."""

import logging

import pytest

from pyfli.solver import FittingComparator


class FakeSaver:
    def __init__(self):
        self.log_lines = []
        self.file_only_lines = []

    def log(self, message):
        self.log_lines.append(message)

    def log_to_file(self, message):
        self.file_only_lines.append(message)

    def save_plot(self, name, fig=None, dpi=300, close=False):
        raise AssertionError("save_plot should not be called when fig is None")


@pytest.fixture
def comparator():
    # freq/fitter classes are unused by _print_summary_table/save_results.
    return FittingComparator(freq=None, base_fitter_class=None, mle_fitter_class=None)


@pytest.fixture
def mono_results_table():
    return [
        [
            "LEAST_SQUARES",
            "NLSF",
            "YES",
            "1.23 ms",
            0.95,
            12.3,
            1.1,
            [2.1, 0.8, 0.0, -0.1],
        ],
        ["POISSON", "MLE", "FAIL", "N/A", 0.0, 0.0, 0.0, None],
    ]


def test_print_summary_table_uses_print_not_logging(
    capsys, caplog, comparator, mono_results_table
):
    """Regression test: the table must go through print(), not logging.info(),
    so it's plain stdout output (no 'INFO:pyfli:' prefix on every line) and so
    save_results' contextlib.redirect_stdout capture actually sees it."""
    with caplog.at_level(logging.INFO, logger="pyfli"):
        comparator._print_summary_table(mono_results_table, "mono-exponential")

    captured = capsys.readouterr()
    assert "LEAST_SQUARES" in captured.out
    assert "┌" in captured.out and "┐" in captured.out
    assert "INFO:pyfli" not in captured.out
    assert caplog.records == []


def test_save_results_writes_table_to_file_only(comparator, mono_results_table):
    """Regression test: the table must go to saver.log_to_file() (file-only),
    not saver.log() (which would also echo every row through logging.info),
    and save_results must no longer serialize the table to JSON."""
    saver = FakeSaver()
    comparator.save_results(
        saver, mono_results_table, fig=None, model_type="mono-exponential", name="cmp"
    )

    joined = "\n".join(saver.file_only_lines)
    assert "LEAST_SQUARES" in joined
    assert "POISSON" in joined
    assert saver.file_only_lines, (
        "save_results must forward the printed table to saver.log_to_file"
    )
    assert saver.log_lines == [], (
        "save_results must not echo the table through saver.log (noisy INFO: output)"
    )
    # FakeSaver has no save_json method -- if save_results still called it,
    # this test would fail with AttributeError instead of silently passing.

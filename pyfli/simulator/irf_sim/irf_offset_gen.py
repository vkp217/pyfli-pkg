# pyfli/simulator/irf_sim/irf_offset_gen.py

"""
Sample per-call IRF time-of-flight shifts and baseline offsets for the simulator
workflow.

This module belongs to :mod:`pyfli.simulator.irf_sim` and is part of PyFLI synthetic
FLI/FLIM data generation, hardware noise modeling, calibration, and validation tools.
Public API includes classes :class:`OffsetGen`.
"""

import numpy as np

from ..sim_helper import irf_picker


class OffsetGen:
    """
    Samples a horizontal time-of-flight shift ``a`` and a vertical baseline offset
    ``b`` for a single pixel's base IRF, and applies them to produce a shifted IRF
    trace: ``I(t) -> np.roll(I, round(a)) + b``.

    Parameters
    ----------
    irf_data : np.ndarray
        Full IRF cube, shape ``(H, W, n_bins)``.
    a_range : tuple[float, float]
        Uniform sampling range for the horizontal shift ``a``, in bins.
    b_range : tuple[float, float]
        Uniform sampling range for the vertical baseline offset ``b``.
    pixel : tuple[int, int]
        ``(row, col)`` pixel used to select the base IRF trace from ``irf_data``.
    """

    def __init__(self, irf_data, a_range=(-20, 100), b_range=(0, 10), pixel=(0, 0)):
        self.a_range = a_range
        self.b_range = b_range
        self.irf_data = irf_data
        self.I_base = irf_data[pixel[0], pixel[1], :].astype(float)  # shape (n_bins,)
        self.n_bins = self.I_base.shape[0]

    def _make_shifted_irf_1d(self, irf_1d_base, a, b):
        """
        Circular shift I(t) -> I(t-a), add offset b. Returns 1D (n_bins,).
        NOTE: np.roll requires an integer shift; `a` is rounded to the
        nearest int here since it's sampled from a continuous uniform range.
        """
        a_int = int(round(a))
        return np.roll(irf_1d_base, a_int) + b

    def sample(self):
        """Draws ``a`` and ``b`` and returns the shifted IRF as ``(irf_1d, a, b)``."""
        a = np.random.uniform(*self.a_range)
        b = np.random.uniform(*self.b_range)
        irf_1d = self._make_shifted_irf_1d(self.I_base, a, b)
        return irf_1d, a, b

    def sample_cube(self):
        """
        Applies an independently-sampled ``(a, b)`` to every pixel of the full
        3-D ``irf_data`` cube, returning the fully shifted IRF cube along with
        ``(H, W)`` maps of the ``a``/``b`` values drawn per pixel.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(irf_cube, a_map, b_map)`` — the shifted ``(H, W, n_bins)`` IRF
            cube, and the per-pixel ``a``/``b`` values used to build it.
        """
        if self.irf_data.ndim != 3:
            raise ValueError(
                "sample_cube requires a 3-D irf_data cube, got shape "
                f"{self.irf_data.shape}"
            )
        H, W, T = self.irf_data.shape
        a_map = np.random.uniform(*self.a_range, size=(H, W))
        b_map = np.random.uniform(*self.b_range, size=(H, W))

        irf_cube = np.empty((H, W, T), dtype=float)
        for i in range(H):
            for j in range(W):
                irf_cube[i, j, :] = self._make_shifted_irf_1d(
                    self.irf_data[i, j, :].astype(float), a_map[i, j], b_map[i, j]
                )
        return irf_cube, a_map, b_map

    def sample_picked(self, px: tuple[int, int] | None = None):
        """
        Picks one IRF trace out of the full 3-D ``irf_data`` cube via
        :func:`~pyfli.simulator.sim_helper.irf_picker` — an SNR-validated
        random pixel, or the given ``px`` — then draws ``a``/``b`` and
        applies them to it.

        Parameters
        ----------
        px : tuple[int, int] | None
            Explicit ``(x, y)`` pixel to pick, forwarded to ``irf_picker``.
            A random SNR-validated pixel is chosen when omitted.

        Returns
        -------
        tuple[np.ndarray, float, float]
            ``(irf_1d, a, b)`` — the shifted IRF trace and the shift/offset
            used to build it.
        """
        irf_1d_base, _ = irf_picker(self.irf_data, px=px)
        a = np.random.uniform(*self.a_range)
        b = np.random.uniform(*self.b_range)
        irf_1d = self._make_shifted_irf_1d(irf_1d_base.astype(float), a, b)
        return irf_1d, a, b

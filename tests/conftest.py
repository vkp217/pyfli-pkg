"""
Shared pytest fixtures for the pyfli test suite.

All fixtures produce synthetic data with no file I/O, so the test
suite runs in any environment with the package installed.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def n_bins():
    return 256


@pytest.fixture(scope="session")
def dt():
    """Time step in ns."""
    return 0.05


@pytest.fixture(scope="session")
def gaussian_irf(n_bins, dt):
    """Gaussian IRF, FWHM ~0.2 ns, centred at bin 10."""
    t = np.arange(n_bins) * dt
    irf = np.exp(-0.5 * ((t - 0.5) / 0.08) ** 2)
    irf /= irf.sum()
    return irf


@pytest.fixture(scope="session")
def mono_decay_1d(n_bins, dt, gaussian_irf):
    """Single-pixel mono-exponential decay convolved with IRF, tau=2 ns."""
    t = np.arange(n_bins) * dt
    h = np.exp(-t / 2.0)
    y = np.convolve(gaussian_irf, h, mode="full")[:n_bins]
    return y / y.max()


@pytest.fixture(scope="session")
def bi_decay_1d(n_bins, dt, gaussian_irf):
    """Single-pixel bi-exponential decay, tau1=0.5 ns, tau2=2.5 ns, a1=0.4."""
    t = np.arange(n_bins) * dt
    h = 0.4 * np.exp(-t / 0.5) + 0.6 * np.exp(-t / 2.5)
    y = np.convolve(gaussian_irf, h, mode="full")[:n_bins]
    return y / y.max()


@pytest.fixture(scope="session")
def decay_cube(n_bins, dt, gaussian_irf):
    """4×4 FLI image cube; each pixel has a random bi-exponential decay."""
    rng = np.random.default_rng(7)
    X, Y = 4, 4
    t = np.arange(n_bins) * dt
    cube = np.zeros((X, Y, n_bins))
    for i in range(X):
        for j in range(Y):
            a1 = rng.uniform(0.2, 0.8)
            h = a1 * np.exp(-t / 0.5) + (1 - a1) * np.exp(-t / 2.5)
            cube[i, j] = np.convolve(gaussian_irf, h, mode="full")[:n_bins]
    return cube / cube.max()

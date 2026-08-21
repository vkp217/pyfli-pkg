"""
Provide static readers and corrections for SPAD, TIFF, MAT, SDT, NumPy, and text data.

This module belongs to :mod:`pyfli.io` and is part of PyFLI detector importers, file
readers, saving helpers, and processed-data loaders. Public API includes classes
:class:`StaticDataOps`.
"""

from typing import Any

import numpy as np
import h5py
import tifffile
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sdtfile import SdtFile

from pyfli import logging


class StaticDataOps:
    """
    Group static low-level readers and correction routines for detector files. The
    methods cover pile-up correction, hot-pixel interpolation, and MAT, SDT, TIFF,
    NumPy, text, ASC, and SPAD HDF5 loading.
    """

    @staticmethod
    def pileup_correction(data: np.ndarray, bit_size: int = 10) -> Any:
        """
        Applies pileup correction to the photon counting data.
        Formula: corrected = -ln(1 - (measured / max_counts)) * max_counts
        """
        dynamic_range = 2**bit_size - 1
        # Ensure float32 to prevent precision loss or integer division issues
        safe_data = np.clip(data.astype(np.float32) / dynamic_range, 0, 0.9999)
        return -np.log(1 - safe_data) * dynamic_range

    @staticmethod
    def spad_hdf5_read(
        fname: str,
        gate_prefix: str | None = None,
        pile_up: bool = True,
        bit_size: int = 10,
    ) -> np.ndarray:
        """
        Read SPAD HDF5 data and normalize it to (H, W, T).

        The reader discovers split gate datasets or stacked 3D cubes from HDF5
        structure and metadata instead of requiring a fixed "Gate Images" group.
        gate_prefix remains available as a backwards-compatible discovery hint for
        existing SwissSPAD2 and SwissSPAD3 callers.

        Parameters
        ----------
        fname : str
            HDF5 file containing SPAD image data.
        gate_prefix : str | None
            Optional split-gate dataset prefix used as a discovery hint.
        pile_up : bool
            Whether pile-up correction should be applied after loading.
        bit_size : int
            Detector digitization bit depth used for pile-up correction.

        Returns
        -------
        np.ndarray
            SPAD image cube with shape (H, W, T) and float32 dtype.
        """
        from .spad_hdf5 import read_spad_hdf5

        result = read_spad_hdf5(
            fname,
            gate_prefix=gate_prefix,
        )

        tpsfs = result.data.astype(
            np.float32,
            copy=False,
        )

        if pile_up:
            tpsfs = StaticDataOps.pileup_correction(
                tpsfs,
                bit_size=bit_size,
            )

        return tpsfs

    @staticmethod
    def hotpixel_correct(data_3d: np.ndarray, hp_map: np.ndarray) -> Any:
        """
        Replace each pixel flagged in hp_map with the nanmedian of its 3×3
        spatial neighbourhood per time gate.
        hp_map  : 2D bool array  (H, W)
        data_3d : float array    (H, W, T)
        """
        cleaned = np.copy(data_3d)
        H, W = data_3d.shape[:2]
        for y, x in zip(*np.where(hp_map)):
            y_min, y_max = max(0, y - 1), min(H, y + 2)
            x_min, x_max = max(0, x - 1), min(W, x + 2)
            nb = data_3d[y_min:y_max, x_min:x_max, :].copy()
            nb[y - y_min, x - x_min, :] = np.nan
            cleaned[y, x, :] = np.nanmedian(nb, axis=(0, 1))
        return cleaned

    @staticmethod
    def load_hp_image(hp_path: str, ref_shape: np.ndarray) -> Any:
        """
        Load a hot pixel mask image (PNG / JPEG / TIFF) → bool (H, W).
        Auto-rotated if image is (W, H) instead of (H, W).
        ref_shape : (H, W) tuple from the corresponding data array.
        """
        mask = plt.imread(hp_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
        if mask.shape != ref_shape:
            if mask.shape == ref_shape[::-1]:
                logging.info(
                    f"[INFO] Hot pixel mask transposed from {mask.shape} → {ref_shape}."
                )
                mask = mask.T
            else:
                raise ValueError(
                    f"HP mask shape {mask.shape} cannot be matched to "
                    f"data spatial shape {ref_shape}."
                )
        return mask > 0

    @staticmethod
    def apply_interpolation_mask(
        data_3d: np.ndarray, hp_path: str | None = None
    ) -> Any:
        """
        Identifies hot pixels from a mask file and replaces them with the
        nanmedian of their 3×3 neighbourhood (excluding the hot pixel itself).
        Signature unchanged — safe to call from data_operations.py.
        """
        if not hp_path:
            raise ValueError("Hotpixel removal mask path (hp_path) is not provided.")

        hotpixel_mask = plt.imread(hp_path)
        if hotpixel_mask.ndim == 3:
            hotpixel_mask = hotpixel_mask[..., 0]

        if data_3d.shape[:2] != hotpixel_mask.shape:
            if (
                data_3d.shape[0] == hotpixel_mask.shape[1]
                and data_3d.shape[1] == hotpixel_mask.shape[0]
            ):
                hotpixel_mask = hotpixel_mask.T
            else:
                raise ValueError(
                    f"Shape mismatch: data {data_3d.shape[:2]} vs mask {hotpixel_mask.shape}"
                )

        return StaticDataOps.hotpixel_correct(data_3d, hotpixel_mask > 0)

    @staticmethod
    def load_mat_file(path: str) -> np.ndarray:
        """
        Load mat file.

        Parameters
        ----------
        path : str
            Filesystem path loaded or saved by the routine.

        Returns
        -------
        np.ndarray
            Data array loaded from a MATLAB file.
        """
        try:
            data = loadmat(path, squeeze_me=True)
            keys = [k for k in data.keys() if not k.startswith("__")]
            return np.asarray(data[keys[0]])

        except NotImplementedError:
            with h5py.File(path, "r") as mat_data:
                keys = [
                    k for k in mat_data.keys() if k not in ["#refs#", "#subsystem#"]
                ]
                return np.asarray(mat_data[keys[0]])

    @staticmethod
    def load_sdt_file(path: str) -> np.ndarray:
        """
        Load sdt file.

        Parameters
        ----------
        path : str
            Filesystem path loaded or saved by the routine.

        Returns
        -------
        np.ndarray
            Data array loaded from a Becker-Hickl SDT file.
        """
        return np.asarray(SdtFile(path).data[0])

    @staticmethod
    def load_tiff_file(path: str) -> np.ndarray:
        """
        Load tiff file.

        Parameters
        ----------
        path : str
            Filesystem path loaded or saved by the routine.

        Returns
        -------
        np.ndarray
            Data array loaded from a TIFF file.
        """
        return np.asarray(tifffile.imread(path))

    @staticmethod
    def load_npy_file(path: str) -> Any:
        """
        Load npy file.

        Parameters
        ----------
        path : str
            Filesystem path loaded or saved by the routine.

        Returns
        -------
        Any
            Object produced by load npy file.
        """
        return np.load(path)

    @staticmethod
    def load_txt_file(
        path: str, target_spatial: tuple[int, ...] = (512, 512)
    ) -> np.ndarray:
        """
        Load txt file.

        Parameters
        ----------
        path : str
            Filesystem path loaded or saved by the routine.
        target_spatial : tuple[int, ...]
            Target spatial shape used when loading text data.

        Returns
        -------
        np.ndarray
            Data array loaded from a text file.
        """
        data = np.loadtxt(path)
        if data.ndim == 1:
            # Reshape 1D IRF/Trace to 3D and tile across spatial dimensions
            data = np.tile(data.reshape(1, 1, -1), (*target_spatial, 1))
        return data

    @staticmethod
    def load_asc_file(path: str, target_spatial: tuple[int, ...] = (512, 512)) -> Any:
        """
        Load asc file.

        Parameters
        ----------
        path : str
            Filesystem path loaded or saved by the routine.
        target_spatial : tuple[int, ...]
            Target spatial shape used when loading text data.

        Returns
        -------
        Any
            Object produced by load asc file.
        """
        data_read = np.genfromtxt(path)
        data_1d = data_read[:, 1] if data_read.ndim == 2 else data_read.flatten()
        return np.tile(data_1d.reshape(1, 1, -1), (*target_spatial, 1))

    @staticmethod
    def SS3HDF5read(
        fname: str,
        pileCorr: bool = True,
        hot_pixels: bool = True,
        hp_path: str | None = None,
    ) -> Any:
        """Read SwissSPAD3 HDF5 data through the shared SPAD loader."""
        if hot_pixels and hp_path is None:
            raise ValueError("hp_path must be provided when hot_pixels=True.")

        try:
            from .spad_io import SpadIO

            result = SpadIO.load_ss3(
                fname,
                config={
                    "input_format": "hdf5",
                    "bit_depth": 10,
                    "pile_up": pileCorr,
                    "fold": False,
                },
                default_bit_depth=10,
            )

            tpsfs = result.data

            if hot_pixels:
                tpsfs = StaticDataOps.apply_interpolation_mask(
                    tpsfs,
                    hp_path=hp_path,
                )

            return tpsfs

        except Exception as exc:
            if isinstance(exc, ValueError):
                raise

            logging.error(f"HDF5 Load Error: {exc}")
            return None

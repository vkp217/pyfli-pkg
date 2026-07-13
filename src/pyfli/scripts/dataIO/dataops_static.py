# dataIO/dataops_static.py
"""Static (stateless) file-format readers and correction routines used by DataOperations and Detector."""
import os
import numpy as np
import h5py
import tifffile
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sdtfile import SdtFile
from pathlib import Path

class Staticdataops:
    """Namespace of static helper methods for reading FLIM file formats and applying corrections.

    All methods are ``@staticmethod`` and operate purely on their arguments,
    making them safe to call from both :class:`~pyfli.scripts.dataIO.dataoperations.DataOperations`
    and :class:`~pyfli.scripts.dataIO.detectorImport.Detector`.
    """
    @staticmethod
    def pileup_correction(data, bit_size=10):
        """
        Applies pileup correction to the photon counting data.
        Formula: corrected = -ln(1 - (measured / max_counts)) * max_counts
        """
        dynamic_range = 2**bit_size - 1    
        # Ensure float32 to prevent precision loss or integer division issues
        safe_data = np.clip(data.astype(np.float32) / dynamic_range, 0, 0.9999)
        return -np.log(1 - safe_data) * dynamic_range

    @staticmethod
    def spad_hdf5_read(fname, gate_prefix, pile_up=True, bit_size=10):
        """
        Generic SPAD HDF5 reader shared by SS2 and SS3.
        gate_prefix : key prefix used to identify gate datasets inside 'Gate Images'
            SS3 → 'Bottom G2 Gate'
            SS2 → 'Gate '
        Reads datasets matching gate_prefix, sorts numerically, stacks → (H, W, T) float32.
        """
        with h5py.File(fname, 'r') as f:
            gate_grp = f.get('Gate Images')
            if gate_grp is None:
                raise KeyError(f"'Gate Images' group not found in {fname}")
            gate_keys = sorted(
                (k for k in gate_grp.keys() if k.startswith(gate_prefix)),
                key=lambda k: int(k.split('Gate ')[-1])
            )
            if not gate_keys:
                raise KeyError(
                    f"No '{gate_prefix}N' datasets found in 'Gate Images' in {fname}")
            tpsfs = np.zeros(
                (*gate_grp[gate_keys[0]].shape, len(gate_keys)), dtype=np.float32)
            for i, key in enumerate(gate_keys):
                tpsfs[:, :, i] = gate_grp[key][:]
        if pile_up:
            tpsfs = Staticdataops.pileup_correction(tpsfs, bit_size=bit_size)
        return tpsfs

    @staticmethod
    def hotpixel_correct(data_3d, hp_map):
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
    def load_hp_image(hp_path, ref_shape):
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
                print(f"[INFO] Hot pixel mask transposed from {mask.shape} → {ref_shape}.")
                mask = mask.T
            else:
                raise ValueError(
                    f"HP mask shape {mask.shape} cannot be matched to "
                    f"data spatial shape {ref_shape}.")
        return mask > 0

    @staticmethod
    def apply_interpolation_mask(data_3d, hp_path=None):
        """
        Identifies hot pixels from a mask file and replaces them with the
        nanmedian of their 3×3 neighbourhood (excluding the hot pixel itself).
        Signature unchanged — safe to call from dataoperations.py.
        """
        if not hp_path:
            raise ValueError("Hotpixel removal mask path (hp_path) is not provided.")

        hotpixel_mask = plt.imread(hp_path)
        if hotpixel_mask.ndim == 3:
            hotpixel_mask = hotpixel_mask[..., 0]

        if data_3d.shape[:2] != hotpixel_mask.shape:
            if (data_3d.shape[0] == hotpixel_mask.shape[1] and
                    data_3d.shape[1] == hotpixel_mask.shape[0]):
                hotpixel_mask = hotpixel_mask.T
            else:
                raise ValueError(
                    f"Shape mismatch: data {data_3d.shape[:2]} vs mask {hotpixel_mask.shape}")

        return Staticdataops.hotpixel_correct(data_3d, hotpixel_mask > 0)

    @staticmethod
    def load_mat_file(path):
        """Load an array from a MATLAB ``.mat`` file.

        Tries ``scipy.io.loadmat`` first (for MATLAB v5-v7 files); falls
        back to ``h5py`` for v7.3 (HDF5-based) files, which raise
        ``NotImplementedError`` in ``loadmat``.

        Args:
            path: Path to the ``.mat`` file.

        Returns:
            numpy.ndarray: The first non-dunder variable found in the file.
        """
        try:
            data = loadmat(path, squeeze_me=True)
            keys = [k for k in data.keys() if not k.startswith('__')]
            return np.asarray(data[keys[0]])
        
        except NotImplementedError:
            with h5py.File(path, 'r') as mat_data:
                keys = [k for k in mat_data.keys() if k not in ['#refs#', '#subsystem#']]
                return np.asarray(mat_data[keys[0]])
        
    @staticmethod
    def load_sdt_file(path):
        """Load the first measurement block from a Becker & Hickl / PicoQuant ``.sdt`` file.

        Args:
            path: Path to the ``.sdt`` file.

        Returns:
            numpy.ndarray: Data of the first measurement block (index 0).
        """
        return np.asarray(SdtFile(path).data[0])

    @staticmethod
    def load_tiff_file(path):
        """Load a TIFF image file.

        Args:
            path: Path to the ``.tif``/``.tiff`` file.

        Returns:
            numpy.ndarray: The image data.
        """
        return np.asarray(tifffile.imread(path))

    @staticmethod
    def load_npy_file(path):
        """Load a NumPy ``.npy`` file.

        Args:
            path: Path to the ``.npy`` file.

        Returns:
            numpy.ndarray: The loaded array.
        """
        return np.load(path)

    @staticmethod
    def load_txt_file(path, target_spatial=(512, 512)):
        """Load a whitespace-delimited text file as an array.

        If the loaded data is 1D (e.g. a single IRF/trace curve), it is
        reshaped to ``(1, 1, N)`` and tiled across ``target_spatial`` to
        produce a pixel-variant cube.

        Args:
            path: Path to the ``.txt`` file.
            target_spatial: ``(H, W)`` spatial shape to tile a 1D trace
                across.

        Returns:
            numpy.ndarray: The loaded (and possibly tiled) array.
        """
        data = np.loadtxt(path)
        if data.ndim == 1:
            # Reshape 1D IRF/Trace to 3D and tile across spatial dimensions
            data = np.tile(data.reshape(1, 1, -1), (*target_spatial, 1))
        return data

    @staticmethod
    def load_asc_file(path, target_spatial=(512, 512)):
        """Load a PicoQuant-style ``.asc`` file (time column 0, counts column 1).

        The counts column is extracted, reshaped to ``(1, 1, N)``, and tiled
        across ``target_spatial`` to produce a pixel-variant cube.

        Args:
            path: Path to the ``.asc`` file.
            target_spatial: ``(H, W)`` spatial shape to tile the trace
                across.

        Returns:
            numpy.ndarray: Array of shape ``(*target_spatial, N)``.
        """
        data_read = np.genfromtxt(path)
        data_1d = data_read[:, 1] if data_read.ndim == 2 else data_read.flatten()
        return np.tile(data_1d.reshape(1, 1, -1), (*target_spatial, 1))

    @staticmethod
    def SS3HDF5read(fname, pileCorr=True, hot_pixels=True, hp_path=None):
        """
        Reads SS3 gated HDF5 data, optionally applying pile-up and hot-pixel corrections.
        Signature unchanged — safe to call from dataoperations.py.
        """
        if hot_pixels and hp_path is None:
            raise ValueError("hp_path must be provided when hot_pixels=True.")
        try:
            tpsfs = Staticdataops.spad_hdf5_read(
                fname, 'Bottom G2 Gate', pile_up=pileCorr)
            if hot_pixels:
                tpsfs = Staticdataops.apply_interpolation_mask(tpsfs, hp_path=hp_path)
            return tpsfs
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            print(f"HDF5 Load Error: {e}")
            return None
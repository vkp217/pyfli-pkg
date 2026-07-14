"""
Load raw decay, IRF, background, mask, and hot-pixel data from common FLIM sources.

This module belongs to :mod:`pyfli.io` and is part of PyFLI detector importers, file
readers, saving helpers, and processed-data loaders. Public API includes classes
:class:`DataOperations`.
"""

from typing import Any

from pyfli import logging

import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

# Import the static logic from your utility file
from .data_ops_static import StaticDataOps as ds


class DataOperations:
    """
    Load primary data, IRF, background, masks, and hot-pixel maps from common FLIM file
    formats. It provides a path-centric interface for raw loading, correction, and
    packaging into PyFLI-ready structures.

    Parameters
    ----------
    data_path : str | None
        Path to the primary decay data source.
    irf_path : str | None
        Path to the instrument response data source.
    bg_path : str | None
        Path to the background measurement used for subtraction or correction.
    mask_path : str | None
        Path to a binary or labeled mask used to select valid pixels.
    hp_path : str | None
        Path to a hot-pixel mask or image used for interpolation.
    """

    def __init__(
        self,
        data_path: str | None = None,
        irf_path: str | None = None,
        bg_path: str | None = None,
        mask_path: str | None = None,
        hp_path: str | None = None,
    ) -> None:
        self.data_path = data_path
        self.irf_path = irf_path
        self.bg_path = bg_path
        self.mask_path = mask_path
        self.hp_path = hp_path

        # --- MODULAR REGISTRY ---
        self.loader_registry = {
            ".mat": ds.load_mat_file,
            ".sdt": ds.load_sdt_file,
            ".tif": ds.load_tiff_file,
            ".tiff": ds.load_tiff_file,
            ".npy": ds.load_npy_file,
            ".txt": ds.load_txt_file,
            ".asc": ds.load_asc_file,
        }

    def load_data(
        self, sub_bg: bool = True, pile_up: bool = False, hot_pixel: bool = False
    ) -> Any:
        """
        Load data.

        Parameters
        ----------
        sub_bg : bool
            Whether background subtraction is applied.
        pile_up : bool
            Whether pile-up correction should be applied.
        hot_pixel : bool
            Whether hot-pixel correction should be applied.

        Returns
        -------
        Any
            Object produced by load data.
        """
        logging.info(f"Initiating DATA load from: {self.data_path}")
        return self._general_loader(
            self.data_path,
            sub_bg=sub_bg,
            pile_up=pile_up,
            hot_pixel=hot_pixel,
            label="DATA",
        )

    def load_background(self, pile_up: bool = False, hot_pixel: bool = False) -> Any:
        """Loads background. If folder, returns the mean average of all files."""
        if self.bg_path and os.path.isdir(self.bg_path):
            logging.info(f"Background FOLDER detected: {self.bg_path}")
            return self._load_from_folder(
                self.bg_path,
                sub_bg=False,
                pile_up=pile_up,
                hot_pixel=True,
                mode="mean",
                is_background=True,
                label="BG",
            )
        if self.bg_path:
            logging.info(f"Background FILE detected: {self.bg_path}")
            return self._general_loader(
                self.bg_path,
                sub_bg=False,
                pile_up=pile_up,
                hot_pixel=hot_pixel,
                label="BG",
            )

        logging.info("No background path provided.")
        return None

    def load_irf(
        self, sub_bg: bool = False, pile_up: bool = False, hot_pixel: bool = False
    ) -> Any:
        """
        Load irf.

        Parameters
        ----------
        sub_bg : bool
            Whether background subtraction is applied.
        pile_up : bool
            Whether pile-up correction should be applied.
        hot_pixel : bool
            Whether hot-pixel correction should be applied.

        Returns
        -------
        Any
            Object produced by load IRF.
        """
        logging.info(f"Initiating IRF load from: {self.irf_path}")
        return self._general_loader(
            self.irf_path,
            sub_bg=sub_bg,
            pile_up=pile_up,
            hot_pixel=hot_pixel,
            label="IRF",
        )

    def load_all_parallel(
        self, sub_bg: bool = True, pile_up: bool = False, hot_pixel: bool = False
    ) -> tuple[Any, ...]:
        """
        Load all parallel.

        Parameters
        ----------
        sub_bg : bool
            Whether background subtraction is applied.
        pile_up : bool
            Whether pile-up correction should be applied.
        hot_pixel : bool
            Whether hot-pixel correction should be applied.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing loaded data arrays, labels, and file metadata.
        """
        logging.info("Starting synchronized parallel loading for DATA, IRF, and BG...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            data_future = executor.submit(
                self.load_data, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel
            )
            irf_future = executor.submit(
                self.load_irf, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel
            )
            bg_future = executor.submit(
                self.load_background, pile_up=pile_up, hot_pixel=hot_pixel
            )

            return data_future.result(), irf_future.result(), bg_future.result()

    def make_dataset(
        self,
        name: str = "Experiment_1",
        source: str = "ICCD",
        sub_bg: bool = True,
        pile_up: bool = False,
        hot_pixel: bool = False,
    ) -> dict[Any, Any]:
        # Fix 2: Check for dimension consistency
        """
        Create dataset.

        Parameters
        ----------
        name : str
            Dataset, experiment, figure, or output name.
        source : str
            Source label recorded with the loaded dataset.
        sub_bg : bool
            Whether background subtraction is applied.
        pile_up : bool
            Whether pile-up correction should be applied.
        hot_pixel : bool
            Whether hot-pixel correction should be applied.

        Returns
        -------
        dict[Any, Any]
            Dictionary containing the data produced by make dataset.
        """
        if all([self.data_path, self.irf_path, self.bg_path]):
            data, irf, background = self.load_all_parallel(
                sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel
            )
        else:
            background = (
                self.load_background(pile_up=pile_up, hot_pixel=hot_pixel)
                if self.bg_path
                else None
            )
            data = (
                self.load_data(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
                if self.data_path
                else None
            )
            irf = (
                self.load_irf(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
                if self.irf_path
                else None
            )

        mask = self.load_mask()

        if data is not None and irf is not None:
            if data.shape[-1] != irf.shape[-1]:
                logging.warning(
                    f"[WARN] Temporal dimension mismatch! DATA: {data.shape[-1]}, IRF: {irf.shape[-1]}"
                )

        return {
            "name": name,
            "source": source,
            "raw_data": {
                "decay": data,
                "irf": irf,
                "background": background,
                "mask": mask,
            },
            "metadata": {
                "shape": data.shape if data is not None else None,
                "processing": {
                    "bg_sub": sub_bg,
                    "pile_up": pile_up,
                    "hot_pixel": hot_pixel,
                },
            },
            "result": {
                "maps": {"tau1_map": None, "tau2_map": None},
                "TR_maps": {"fit_map": None, "residuals_maps": None},
            },
        }

    def load_mask(self) -> Any:
        """
        Load mask.

        Returns
        -------
        Any
            Object produced by load mask.
        """
        if not self.mask_path:
            return None
        logging.info(f"Loading mask from: {self.mask_path}")
        mask = self._load_single_file(self.mask_path)
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = np.mean(mask, axis=-1)
        return (mask > np.min(mask)).astype(bool)

    def _general_loader(
        self,
        path: str,
        sub_bg: bool = True,
        pile_up: bool = False,
        hot_pixel: bool = False,
        label: str = "Data",
    ) -> Any:
        """
        Run the general loader routine.

        Parameters
        ----------
        path : str
            Filesystem path loaded or saved by the routine.
        sub_bg : bool
            Whether background subtraction is applied.
        pile_up : bool
            Whether pile-up correction should be applied.
        hot_pixel : bool
            Whether hot-pixel correction should be applied.
        label : str
            Display label assigned to the data or plot element.

        Returns
        -------
        Any
            Object produced by general loader.
        """
        if not path or not os.path.exists(path):
            abs_path = os.path.abspath(path) if path else "(None)"
            logging.error(f"[ERROR] {label} path not found: {abs_path}")
            return None

        if os.path.isfile(path):
            return self._load_single_file(path, pile_up, hot_pixel)
        else:
            return self._load_from_folder(path, sub_bg, pile_up, hot_pixel, label=label)

    def _load_single_file(
        self,
        file_path: str,
        pile_up: bool = False,
        hot_pixel: bool = False,
        active_hp: np.ndarray | None = None,
    ) -> Any:
        """
        Load single file.

        Parameters
        ----------
        file_path : str
            Path to the file being loaded.
        pile_up : bool
            Whether pile-up correction should be applied.
        hot_pixel : bool
            Whether hot-pixel correction should be applied.
        active_hp : np.ndarray | None
            Hot-pixel mask currently applied to the loaded data.

        Returns
        -------
        Any
            Object produced by load single file.
        """
        ext = os.path.splitext(file_path)[-1].lower()
        active_hp = active_hp or self.hp_path

        # Fix 3: Path validation for hot pixel mask
        if hot_pixel and (active_hp is None or not os.path.exists(active_hp)):
            logging.warning(
                f"[WARN] Hot-pixel correction skipped for {os.path.basename(file_path)}: hp_path invalid."
            )
            hot_pixel = False

        if ext in (".hdf5", ".h5"):
            return ds.SS3HDF5read(
                file_path, pileCorr=pile_up, hot_pixels=hot_pixel, hp_path=active_hp
            )

        loader_func = self.loader_registry.get(ext)
        if not loader_func:
            return None
        try:
            data_content = loader_func(file_path)
            if data_content is not None:
                # Fix 4: Pre-cast to float32 for processing safety
                data_content = data_content.astype(np.float32)
                if pile_up:
                    data_content = ds.pileup_correction(data_content)
                if hot_pixel:
                    data_content = ds.apply_interpolation_mask(
                        data_content, hp_path=active_hp
                    )
            return data_content
        except Exception as e:
            logging.error(f"[ERROR] Failed to load {file_path}: {e}")
            return None

    def _load_from_folder(
        self,
        folder_path: str,
        sub_bg: bool = True,
        pile_up: bool = False,
        hot_pixel: bool = False,
        active_hp: np.ndarray | None = None,
        mode: str = "sum",
        is_background: bool = False,
        label: str = "Data",
    ) -> Any:
        """
        Load from folder.

        Parameters
        ----------
        folder_path : str
            Directory containing detector files to load.
        sub_bg : bool
            Whether background subtraction is applied.
        pile_up : bool
            Whether pile-up correction should be applied.
        hot_pixel : bool
            Whether hot-pixel correction should be applied.
        active_hp : np.ndarray | None
            Hot-pixel mask currently applied to the loaded data.
        mode : str
            Mode selector used by the fitting, loading, or plotting routine.
        is_background : bool
            Whether the file should be loaded as a background measurement.
        label : str
            Display label assigned to the data or plot element.

        Returns
        -------
        Any
            Object produced by load from folder.
        """
        valid_exts = (".tif", ".tiff", ".hdf5", ".h5")
        files = sorted(
            [f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        )

        if not files:
            raise FileNotFoundError(f"No valid files found in {folder_path}")

        # Strict Requirement: HDF5 folders force correction True
        if any(f.lower().endswith((".hdf5", ".h5")) for f in files):
            pile_up, hot_pixel = True, True
            logging.info("[INFO] HDF5 folder detected. Corrections forced to True.")

        full_paths = [os.path.join(folder_path, f) for f in files]
        bg_avg = (
            self.load_background(pile_up=pile_up, hot_pixel=hot_pixel)
            if (sub_bg and not is_background)
            else None
        )

        first = self._load_single_file(full_paths[0], pile_up, hot_pixel, active_hp)
        if first is None:
            return None

        # Pre-allocate as float32
        stack = np.zeros((*first.shape, len(files)), dtype=np.float32)
        stack[..., 0] = first

        if len(files) > 1:
            task_args = [
                (i, p, pile_up, hot_pixel, active_hp or self.hp_path)
                for i, p in enumerate(full_paths[1:], start=1)
            ]
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(
                    tqdm(
                        executor.map(self._load_single_file_parallel, task_args),
                        total=len(task_args),
                        desc=f"Loading {label}",
                        leave=False,
                    )
                )
                for idx, res_data in results:
                    if res_data is not None and res_data.shape == first.shape:
                        # FIXED: Use ellipsis to target the last axis for 'idx'
                        stack[..., idx] = res_data

        # Fix 1: Subtraction with zero-floor
        if bg_avg is not None:
            for i in range(stack.shape[-1]):
                if bg_avg.shape == stack[..., i].shape:
                    stack[..., i] -= bg_avg
            stack = np.maximum(stack, 0)

        if is_background:
            return np.mean(stack, axis=-1)

        if stack.ndim == 4:
            return np.sum(stack, axis=-1) if mode == "sum" else np.mean(stack, axis=-1)

        return stack

    def _load_single_file_parallel(self, args: Any) -> tuple[Any, ...]:
        """
        Load single file parallel.

        Parameters
        ----------
        args : Any
            Worker argument tuple passed to the parallel file-processing helper.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing loaded file data and file metadata.
        """
        idx, path, pile_up, hot_pixel, active_hp = args
        res_data = self._load_single_file(path, pile_up, hot_pixel, active_hp)
        return idx, res_data

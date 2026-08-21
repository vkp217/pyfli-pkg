"""
Coordinate detector-specific loading workflows for SS2, SS3, ICCD, TCSPC, and generic
data.

This module belongs to :mod:`pyfli.io` and is part of PyFLI detector importers, file
readers, saving helpers, and processed-data loaders. Public API includes classes
:class:`Detector`.
"""

from dataclasses import replace
from typing import Any

import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from .spad_folding import apply_fold_layout
from .spad_io import SpadConfig, SpadIO
from pyfli import logging
from .data_ops_static import StaticDataOps as ds


class Detector:
    """
    Provide detector-specific loading workflows for PyFLI experiments. Use this high-
    level loader for SS3, SS2, ICCD, BH TCSPC, and generic data sources with optional
    background subtraction, pile-up correction, masks, and hot-pixel handling.

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
    bit_size : int
        Detector digitization bit depth used for pile-up correction.
    """

    def __init__(
        self,
        data_path: str | None = None,
        irf_path: str | None = None,
        bg_path: str | None = None,
        mask_path: str | None = None,
        hp_path: str | None = None,
        bit_size: int = 10,
    ) -> None:
        self.data_path = data_path
        self.irf_path = irf_path
        self.bg_path = bg_path
        self.mask_path = mask_path
        self.hp_path = hp_path
        self.bit_size = bit_size

    # ================================================================= #
    #  DETECTOR METHODS
    # ================================================================= #
    def SS3(
        self,
        name: str = "Experiment_1",
        sub_bg: bool = True,
        pile_up: bool = True,
        hot_pixel: bool = True,
        make_hp_map: bool = True,
        threshold_sigma: float = 5.0,
        config: SpadConfig | dict[str, Any] | None = None,
    ) -> Any:
        """
        Load SwissSPAD3 HDF5 data.

        Single-file input applies optional pile-up and folding. Folder input can also
        apply hot-pixel correction and background subtraction before file combination.

        Parameters
        ----------
        name : str
            Dataset or experiment name.
        sub_bg : bool
            Whether to subtract background in folder mode.
        pile_up : bool
            Whether to apply pile-up correction.
        hot_pixel : bool
            Whether to apply hot-pixel correction in folder mode.
        make_hp_map : bool
            Whether to derive the hot-pixel map from background data.
        threshold_sigma : float
            MAD threshold used for automatic hot-pixel detection.
        config : SpadConfig | dict[str, Any] | None
            SPAD loading and folding options.

        Returns
        -------
        Any
            Standard PyFLI dataset package.
        """
        return self._load_ss(
            "ss3",
            name,
            sub_bg,
            pile_up,
            hot_pixel,
            make_hp_map,
            threshold_sigma,
            config,
        )

    def SS2(
        self,
        name: str = "Experiment_1",
        sub_bg: bool = True,
        pile_up: bool = True,
        hot_pixel: bool = True,
        make_hp_map: bool = True,
        threshold_sigma: float = 5.0,
        config: SpadConfig | dict[str, Any] | None = None,
    ) -> Any:
        """
        Load SwissSPAD2 HDF5 or native BIN data.

        HDF5 input uses ``Gate Images/Gate N``. Native BIN input uses matched
        ``topN.bin`` and ``btmN.bin`` files. Single-acquisition input applies optional
        pile-up and folding. HDF5 folder input can also apply hot-pixel correction and
        background subtraction before file combination.

        Parameters
        ----------
        name : str
            Dataset or experiment name.
        sub_bg : bool
            Whether to subtract background in HDF5 folder mode.
        pile_up : bool
            Whether to apply pile-up correction.
        hot_pixel : bool
            Whether to apply hot-pixel correction in HDF5 folder mode.
        make_hp_map : bool
            Whether to derive the hot-pixel map from background data.
        threshold_sigma : float
            MAD threshold used for automatic hot-pixel detection.
        config : SpadConfig | dict[str, Any] | None
            SPAD loading, BIN, and folding options.

        Returns
        -------
        Any
            Standard PyFLI dataset package.
        """
        return self._load_ss(
            "ss2",
            name,
            sub_bg,
            pile_up,
            hot_pixel,
            make_hp_map,
            threshold_sigma,
            config,
        )

    def SPAD(
        self,
        name: str = "Experiment_1",
        config: SpadConfig | dict[str, Any] | None = None,
    ) -> Any:
        """
        Generic SPAD detector loader for HDF5 and native SwissSPAD2 binary data.

        HDF5 files are discovered from their structure, dimensions, attributes, and
        numeric ordering rather than fixed detector-specific group names. SwissSPAD2
        binary acquisitions are decoded from matched topN.bin / btmN.bin chunks and
        stitched into a 512 x 512 detector cube.

        Optional pile-up correction is applied before optional periodic temporal
        folding. Folding detects or uses an explicit circular phase shift, aligns the
        complete acquisition on the time axis, and sums repeated excitation periods.
        Background subtraction is not performed by this loader.

        Parameters
        ----------
        name : str
            Dataset or experiment name stored in the returned PyFLI package.
        config : SpadConfig | dict[str, Any] | None
            SPAD input, pile-up, HDF5 discovery, SwissSPAD2 binary, and folding options.

        Returns
        -------
        Any
            Standard PyFLI dataset package containing SPAD decay, optional
            IRF/background, mask, and complete import metadata.
        """
        if not self.data_path or not os.path.exists(self.data_path):
            raise ValueError("SPAD: data_path must be provided and must exist.")

        resolved_config = SpadConfig.from_value(
            config,
            default_bit_depth=self.bit_size,
        )

        decay_result = SpadIO.load(
            self.data_path,
            config=resolved_config,
            default_bit_depth=self.bit_size,
        )

        shared_fold_layout = decay_result.fold_layout if resolved_config.fold else None

        irf_result = None

        if self.irf_path:
            if not os.path.exists(self.irf_path):
                raise ValueError(f"SPAD: irf_path does not exist: {self.irf_path}")

            irf_result = SpadIO.load(
                self.irf_path,
                config=resolved_config,
                default_bit_depth=self.bit_size,
                fold_layout=shared_fold_layout,
            )

            if irf_result.data.shape != decay_result.data.shape:
                raise ValueError(
                    f"SPAD: IRF shape {irf_result.data.shape} "
                    f"does not match data shape "
                    f"{decay_result.data.shape}."
                )

        background_result = None

        if self.bg_path:
            if not os.path.exists(self.bg_path):
                raise ValueError(f"SPAD: bg_path does not exist: {self.bg_path}")

            background_result = SpadIO.load(
                self.bg_path,
                config=resolved_config,
                default_bit_depth=self.bit_size,
                fold_layout=shared_fold_layout,
            )

            if background_result.data.shape != decay_result.data.shape:
                raise ValueError(
                    f"SPAD: background shape "
                    f"{background_result.data.shape} does not "
                    f"match data shape {decay_result.data.shape}."
                )

        mask = self._load_mask()

        spad_metadata = {
            "decay": decay_result.metadata,
            "irf": (irf_result.metadata if irf_result is not None else None),
            "background": (
                background_result.metadata if background_result is not None else None
            ),
        }

        return self._package(
            decay_result.data,
            (irf_result.data if irf_result is not None else None),
            (background_result.data if background_result is not None else None),
            mask,
            name,
            source="SPAD",
            sub_bg=False,
            pile_up=resolved_config.pile_up,
            fold=resolved_config.fold,
            bit_size=resolved_config.bit_depth,
            spad_metadata=spad_metadata,
        )

    def ICCD(self, name: str = "Experiment_1") -> Any:
        """
        Intensified CCD detector.

        Both data_path and irf_path must be folders of TIFF files.
        Each TIFF represents one gate position; files are sorted alphabetically
        and stacked along the time axis → (H, W, N_gates).
        IRF is pixel-variant: its shape must exactly match the data shape.
        No pile-up, hot-pixel, or background correction is applied.
        """
        if not self.data_path or not os.path.isdir(self.data_path):
            raise ValueError("ICCD: data_path must be a folder of TIFF files.")

        decay = self._load_iccd_folder(self.data_path)

        irf = None
        if self.irf_path:
            if not os.path.isdir(self.irf_path):
                raise ValueError("ICCD: irf_path must be a folder of TIFF files.")
            irf = self._load_iccd_folder(self.irf_path)
            if irf.ndim != 3:
                raise ValueError(
                    f"ICCD: IRF must be 3D (H, W, T), got shape {irf.shape}."
                )
            if irf.shape != decay.shape:
                raise ValueError(
                    f"ICCD: IRF shape {irf.shape} does not match data shape {decay.shape}."
                )

        mask = self._load_mask()

        return self._package(decay, irf, None, mask, name, source="ICCD")

    def BH_TCSPC(
        self, name: str = "Experiment_1", sub_bg: bool = True, channel: int = 0
    ) -> Any:
        """
        Time-Correlated Single Photon Counting detectors.

        Supported formats: .sdt (Becker & Hickl / PicoQuant), .asc, .mat, .npy, .tif

        Pile-up correction is NOT applied.  TCSPC pile-up follows a dead-time
        model  C_true = C_meas / (1 − C_meas · τ_dead · f_rep)  that is
        detector-dependent and must be applied externally when needed.

        channel : SDT measurement block index for multi-block files (default 0).
        """
        decay = self._dispatch(
            self.data_path,
            sub_bg=sub_bg,
            pile_up=False,
            hot_pixel=False,
            channel=channel,
        )
        irf = self._dispatch(
            self.irf_path, sub_bg=False, pile_up=False, hot_pixel=False, channel=channel
        )
        bg = self._load_background(pile_up=False, hot_pixel=False)
        mask = self._load_mask()

        return self._package(
            decay,
            irf,
            bg,
            mask,
            name,
            source="BH-TCSPC",
            sub_bg=sub_bg,
            pile_up=False,
            hot_pixel=False,
            channel=channel,
        )

    def generic(
        self,
        name: str = "Experiment_1",
        sub_bg: bool = True,
        pile_up: bool = False,
        hot_pixel: bool = False,
    ) -> Any:
        """
        Generic loader: TIFF / NPY / MAT / TXT / HDF5.
        All corrections opt-in.
        """
        decay = self._dispatch(
            self.data_path, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel
        )
        irf = self._dispatch(
            self.irf_path, sub_bg=False, pile_up=pile_up, hot_pixel=hot_pixel
        )
        bg = self._load_background(pile_up=pile_up, hot_pixel=hot_pixel)
        mask = self._load_mask()

        return self._package(
            decay,
            irf,
            bg,
            mask,
            name,
            source="Generic",
            sub_bg=sub_bg,
            pile_up=pile_up,
            hot_pixel=hot_pixel,
        )

    # ================================================================= #
    #  ICCD-SPECIFIC LOADERS
    # ================================================================= #

    def _load_iccd_folder(self, folder_path: str) -> Any:
        """
        Loads sorted TIFF files as gate images → (H, W, N_gates) float32.
        No corrections applied.
        """
        files = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith((".tif", ".tiff"))
        )
        if not files:
            raise FileNotFoundError(f"No TIFF files in: {folder_path}")

        full_paths = [os.path.join(folder_path, f) for f in files]

        first_raw = ds.load_tiff_file(full_paths[0]).astype(np.float32)
        first = first_raw.mean(axis=-1) if first_raw.ndim == 3 else first_raw
        H, W = first.shape

        stack = np.zeros((H, W, len(files)), dtype=np.float32)
        stack[:, :, 0] = first

        def _read_gate(args: Any) -> tuple[Any, ...]:
            """
            Read gate.

            Parameters
            ----------
            args : Any
                Worker argument tuple passed to the parallel file-processing helper.

            Returns
            -------
            tuple[Any, ...]
                Tuple containing gate data and gate metadata read from disk.
            """
            idx, path = args
            raw = ds.load_tiff_file(path).astype(np.float32)
            return idx, raw.mean(axis=-1) if raw.ndim == 3 else raw

        if len(files) > 1:
            tasks = list(enumerate(full_paths[1:], start=1))
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                for idx, gate in tqdm(
                    ex.map(_read_gate, tasks),
                    total=len(tasks),
                    desc="Loading ICCD gates",
                    leave=False,
                ):
                    if gate is not None and gate.shape == (H, W):
                        stack[:, :, idx] = gate

        return stack

    # ================================================================= #
    #  SPAD PIPELINE HELPERS
    # ================================================================= #

    def _ss_loader(self, detector: str) -> Any:
        """Return the shared loader for a SwissSPAD detector."""
        if detector == "ss2":
            return SpadIO.load_ss2

        if detector == "ss3":
            return SpadIO.load_ss3

        raise ValueError(f"Unsupported SwissSPAD detector: '{detector}'.")

    def _first_hdf5(self, folder_path: str) -> str:
        """Return the first sorted HDF5 file in a directory."""
        if not os.path.isdir(folder_path):
            raise ValueError(f"Expected an HDF5 directory, got: {folder_path}")

        filenames = sorted(
            filename
            for filename in os.listdir(folder_path)
            if filename.lower().endswith((".h5", ".hdf5"))
        )

        if not filenames:
            raise FileNotFoundError(f"No HDF5 files found in: {folder_path}")

        return os.path.join(
            folder_path,
            filenames[0],
        )

    def _load_ss(
        self,
        detector: str,
        name: str,
        sub_bg: bool,
        pile_up: bool,
        hot_pixel: bool,
        make_hp_map: bool,
        threshold_sigma: float,
        config: SpadConfig | dict[str, Any] | None,
    ) -> Any:
        """Load SwissSPAD2 or SwissSPAD3 through the shared SPAD pipeline."""
        if detector not in ("ss2", "ss3"):
            raise ValueError(f"Unsupported SwissSPAD detector: '{detector}'.")

        label = detector.upper()
        source = "SwissSPAD2" if detector == "ss2" else "SwissSPAD3"

        if not self.data_path or not os.path.exists(self.data_path):
            raise ValueError(f"{label}: data_path must be provided and must exist.")

        if (
            hot_pixel
            and make_hp_map
            and (not np.isfinite(threshold_sigma) or threshold_sigma <= 0)
        ):
            raise ValueError(
                f"{label}: threshold_sigma must be positive, got {threshold_sigma}."
            )

        resolved_config = SpadConfig.from_value(
            config,
            default_bit_depth=self.bit_size,
        )

        resolved_config = replace(
            resolved_config,
            pile_up=bool(pile_up),
        )

        loader = self._ss_loader(detector)

        input_format = SpadIO.get_format(
            self.data_path,
            config=resolved_config,
            default_bit_depth=self.bit_size,
            detector=detector,
        )

        folder_mode = input_format == "hdf5" and os.path.isdir(self.data_path)

        if not folder_mode:
            decay_result = loader(
                self.data_path,
                config=resolved_config,
                default_bit_depth=self.bit_size,
            )

            irf_result = None

            if self.irf_path:
                if not os.path.exists(self.irf_path):
                    raise ValueError(
                        f"{label}: irf_path does not exist: {self.irf_path}"
                    )

                irf_result = loader(
                    self.irf_path,
                    config=resolved_config,
                    default_bit_depth=self.bit_size,
                    fold_layout=(
                        decay_result.fold_layout if resolved_config.fold else None
                    ),
                )

                if irf_result.data.shape != decay_result.data.shape:
                    raise ValueError(
                        f"{label}: IRF shape {irf_result.data.shape} "
                        f"does not match decay shape "
                        f"{decay_result.data.shape}."
                    )

            mask = self._load_mask()

            spad_metadata = {
                "decay": decay_result.metadata,
                "irf": (irf_result.metadata if irf_result is not None else None),
                "background": None,
            }

            return self._package(
                decay_result.data,
                (irf_result.data if irf_result is not None else None),
                None,
                mask,
                name,
                source=source,
                sub_bg=False,
                pile_up=resolved_config.pile_up,
                hot_pixel=False,
                make_hp_map=False,
                fold=resolved_config.fold,
                bit_size=resolved_config.bit_depth,
                input_format=input_format,
                spad_metadata=spad_metadata,
            )

        background_result = None
        bg_cube = None

        needs_background = sub_bg or (hot_pixel and make_hp_map)

        if needs_background:
            if not self.bg_path:
                raise ValueError(
                    f"{label}: bg_path is required when sub_bg=True "
                    "or automatic hot-pixel mapping is enabled."
                )

            if not os.path.isdir(self.bg_path):
                raise ValueError(
                    f"{label}: bg_path must be an HDF5 directory in folder mode."
                )

            bg_config = replace(
                resolved_config,
                input_format="hdf5",
                pile_up=False,
                fold=False,
                hdf5_folder_mode="mean",
            )

            background_result = loader(
                self.bg_path,
                config=bg_config,
                default_bit_depth=self.bit_size,
            )

            bg_cube = background_result.data.astype(
                np.float32,
                copy=False,
            )

        hp_map = None

        if hot_pixel:
            if make_hp_map:
                if bg_cube is None:
                    raise RuntimeError(
                        f"{label}: background data was not loaded "
                        "for hot-pixel detection."
                    )

                from .utils import DataIOUtils

                (
                    hp_map,
                    _,
                    _,
                    _,
                ) = DataIOUtils.hot_pixels(
                    bg_cube,
                    threshold_sigma=threshold_sigma,
                )

            else:
                if not self.hp_path:
                    raise ValueError(
                        f"{label}: hp_path is required when "
                        "hot_pixel=True and make_hp_map=False."
                    )

                if not os.path.isfile(self.hp_path):
                    raise ValueError(
                        f"{label}: hp_path must be an existing image file, "
                        f"got: {self.hp_path}"
                    )

                if bg_cube is not None:
                    reference_shape = bg_cube.shape[:2]

                else:
                    ref_config = replace(
                        resolved_config,
                        input_format="hdf5",
                        pile_up=False,
                        fold=False,
                    )

                    ref_result = loader(
                        self._first_hdf5(self.data_path),
                        config=ref_config,
                        default_bit_depth=self.bit_size,
                    )

                    reference_shape = ref_result.data.shape[:2]

                hp_map = ds.load_hp_image(
                    self.hp_path,
                    reference_shape,
                )

        decay_result = loader(
            self.data_path,
            config=resolved_config,
            default_bit_depth=self.bit_size,
            hot_pixel_map=hp_map,
            background=bg_cube,
            sub_bg=sub_bg,
        )

        irf_result = None

        if self.irf_path:
            if not os.path.isdir(self.irf_path):
                raise ValueError(
                    f"{label}: irf_path must be an HDF5 directory "
                    "when data_path is an HDF5 directory."
                )

            irf_format = SpadIO.get_format(
                self.irf_path,
                config=resolved_config,
                default_bit_depth=self.bit_size,
                detector=detector,
            )

            if irf_format != "hdf5":
                raise ValueError(
                    f"{label}: irf_path must resolve to HDF5 "
                    f"in folder mode, got '{irf_format}'."
                )

            irf_result = loader(
                self.irf_path,
                config=resolved_config,
                default_bit_depth=self.bit_size,
                fold_layout=(
                    decay_result.fold_layout if resolved_config.fold else None
                ),
                hot_pixel_map=hp_map,
                background=bg_cube,
                sub_bg=sub_bg,
            )

            if irf_result.data.shape != decay_result.data.shape:
                raise ValueError(
                    f"{label}: IRF shape {irf_result.data.shape} "
                    f"does not match decay shape "
                    f"{decay_result.data.shape}."
                )

        packaged_background = bg_cube if sub_bg else None

        if (
            packaged_background is not None
            and resolved_config.fold
            and decay_result.fold_layout is not None
        ):
            packaged_background = apply_fold_layout(
                packaged_background,
                decay_result.fold_layout,
            )

        mask = self._load_mask()

        spad_metadata = {
            "decay": decay_result.metadata,
            "irf": (irf_result.metadata if irf_result is not None else None),
            "background": (
                background_result.metadata if background_result is not None else None
            ),
        }

        return self._package(
            decay_result.data,
            (irf_result.data if irf_result is not None else None),
            packaged_background,
            mask,
            name,
            source=source,
            sub_bg=sub_bg,
            pile_up=resolved_config.pile_up,
            hot_pixel=hp_map is not None,
            make_hp_map=(make_hp_map if hot_pixel else False),
            fold=resolved_config.fold,
            bit_size=resolved_config.bit_depth,
            threshold_sigma=threshold_sigma,
            input_format=input_format,
            spad_metadata=spad_metadata,
        )

    # ================================================================= #
    #  GENERIC ROUTING  (SS3, TCSPC, generic)
    # ================================================================= #

    def _dispatch(
        self,
        path: str,
        sub_bg: bool,
        pile_up: bool,
        hot_pixel: bool,
        valid_exts: np.ndarray | None = None,
        **kw: Any,
    ) -> Any:
        """
        Run the dispatch routine.

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
        valid_exts : np.ndarray | None
            Allowed file extensions for detector loading.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by dispatch.
        """
        if not path:
            return None
        if not os.path.exists(path):
            logging.error(f"[ERROR] Path not found: {os.path.abspath(path)}")
            return None
        _exts = valid_exts or (
            ".tif",
            ".tiff",
            ".hdf5",
            ".h5",
            ".sdt",
            ".mat",
            ".npy",
            ".txt",
            ".asc",
        )
        if os.path.isdir(path):
            return self._load_folder(
                path,
                sub_bg=sub_bg,
                pile_up=pile_up,
                hot_pixel=hot_pixel,
                valid_exts=_exts,
                **kw,
            )
        return self._load_file(path, pile_up=pile_up, hot_pixel=hot_pixel, **kw)

    # ================================================================= #
    #  GENERIC FOLDER LOADING  (SS3, TCSPC, generic)
    # ================================================================= #

    def _load_folder(
        self,
        folder_path: str,
        sub_bg: bool = True,
        pile_up: bool = False,
        hot_pixel: bool = False,
        mode: str = "sum",
        valid_exts: tuple[str, ...] = (
            ".tif",
            ".tiff",
            ".hdf5",
            ".h5",
            ".sdt",
            ".mat",
            ".npy",
            ".txt",
            ".asc",
        ),
        **kw: Any,
    ) -> Any:
        """
        Load folder.

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
        mode : str
            Mode selector used by the fitting, loading, or plotting routine.
        valid_exts : tuple[str, ...]
            Allowed file extensions for detector loading.
        **kw : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by load folder.
        """
        files = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)
        )
        if not files:
            raise FileNotFoundError(f"No files matching {valid_exts} in: {folder_path}")

        full_paths = [os.path.join(folder_path, f) for f in files]
        bg_avg = (
            self._load_background(
                pile_up=pile_up, hot_pixel=hot_pixel, valid_exts=valid_exts
            )
            if sub_bg
            else None
        )

        first = self._load_file(
            full_paths[0], pile_up=pile_up, hot_pixel=hot_pixel, **kw
        )
        if first is None:
            return None

        stack = np.zeros((*first.shape, len(files)), dtype=np.float32)
        stack[..., 0] = first

        if len(files) > 1:
            tasks = [
                (i, p, pile_up, hot_pixel, kw)
                for i, p in enumerate(full_paths[1:], start=1)
            ]
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                for idx, data in tqdm(
                    ex.map(self._file_task, tasks),
                    total=len(tasks),
                    desc="Loading folder",
                    leave=False,
                ):
                    if data is not None and data.shape == first.shape:
                        stack[..., idx] = data

        if bg_avg is not None:
            for i in range(stack.shape[-1]):
                if bg_avg.shape == stack[..., i].shape:
                    stack[..., i] -= bg_avg
                else:
                    logging.warning(
                        f"[WARN] BG shape {bg_avg.shape} ≠ frame '{files[i]}' shape {stack[..., i].shape} — subtraction skipped."
                    )
            stack = np.maximum(stack, 0)

        # Collapse file axis: 4D (H,W,T,N_files) → 3D (H,W,T)
        # 3D stacks (from 2D files) are returned as-is
        if stack.ndim == 4:
            return np.sum(stack, axis=-1) if mode == "sum" else np.mean(stack, axis=-1)
        return stack

    def _file_task(self, args: Any) -> tuple[Any, ...]:
        """
        Run the file task routine.

        Parameters
        ----------
        args : Any
            Worker argument tuple passed to the parallel file-processing helper.

        Returns
        -------
        tuple[Any, ...]
            Tuple containing loaded file data and associated metadata.
        """
        idx, path, pile_up, hot_pixel, kw = args
        return idx, self._load_file(path, pile_up=pile_up, hot_pixel=hot_pixel, **kw)

    # ================================================================= #
    #  SINGLE FILE LOADING
    # ================================================================= #

    def _load_file(
        self,
        file_path: str,
        pile_up: bool = False,
        hot_pixel: bool = False,
        channel: int = 0,
        **_: Any,
    ) -> Any:
        """
        Load file.

        Parameters
        ----------
        file_path : str
            Path to the file being loaded.
        pile_up : bool
            Whether pile-up correction should be applied.
        hot_pixel : bool
            Whether hot-pixel correction should be applied.
        channel : int
            Detector channel index to read or decode.
        **_ : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        Any
            Object produced by load file.
        """
        if not file_path or not os.path.exists(file_path):
            return None

        ext = os.path.splitext(file_path)[-1].lower()

        try:
            if ext in (".hdf5", ".h5"):
                return ds.spad_hdf5_read(
                    file_path,
                    gate_prefix=None,
                    pile_up=pile_up,
                    bit_size=self.bit_size,
                )

            if ext == ".sdt":
                from sdtfile import SdtFile

                return np.asarray(
                    SdtFile(file_path).data[channel],
                    dtype=np.float32,
                )

            if ext == ".asc":
                return ds.load_asc_file(file_path).astype(np.float32)

            loaders = {
                ".mat": ds.load_mat_file,
                ".npy": ds.load_npy_file,
                ".tif": ds.load_tiff_file,
                ".tiff": ds.load_tiff_file,
                ".txt": ds.load_txt_file,
            }

            loader = loaders.get(ext)

            if loader is None:
                logging.warning(
                    f"[WARN] Unsupported format '{ext}': {os.path.basename(file_path)}"
                )
                return None

            data = loader(file_path).astype(np.float32)

            if pile_up:
                data = ds.pileup_correction(
                    data,
                    bit_size=self.bit_size,
                )

            if hot_pixel:
                data = ds.apply_interpolation_mask(
                    data,
                    hp_path=self.hp_path,
                )

            return data

        except Exception as e:
            logging.error(f"[ERROR] {os.path.basename(file_path)}: {e}")
            return None

    # ================================================================= #
    #  SwissSPAD3 HDF5 READER
    # ================================================================= #

    def _read_ss3_hdf5(
        self,
        fname: str,
        pile_up: bool = True,
        hot_pixel: bool = False,
    ) -> Any:
        """Read one SwissSPAD3 HDF5 cube through the shared SPAD loader."""
        _ = hot_pixel

        result = SpadIO.load_ss3(
            fname,
            config={
                "input_format": "hdf5",
                "bit_depth": self.bit_size,
                "pile_up": pile_up,
                "fold": False,
            },
            default_bit_depth=self.bit_size,
        )

        return result.data

    def _read_ss2_hdf5(
        self,
        fname: str,
        pile_up: bool = True,
    ) -> Any:
        """Read one SwissSPAD2 HDF5 cube through the shared SPAD loader."""
        result = SpadIO.load_ss2(
            fname,
            config={
                "input_format": "hdf5",
                "bit_depth": self.bit_size,
                "pile_up": pile_up,
                "fold": False,
            },
            default_bit_depth=self.bit_size,
        )

        return result.data

    def _correct_hotpixels(self, data_3d: np.ndarray, hot_pixel_map: np.ndarray) -> Any:
        """
        Run the correct hotpixels routine.

        Parameters
        ----------
        data_3d : np.ndarray
            Three-dimensional data cube processed by the routine.
        hot_pixel_map : np.ndarray
            Boolean map marking hot pixels to correct.

        Returns
        -------
        Any
            Object produced by correct hotpixels.
        """
        return ds.hotpixel_correct(data_3d, hot_pixel_map)

    # ================================================================= #
    #  BACKGROUND AND MASK
    # ================================================================= #

    def _load_background(
        self,
        pile_up: bool = False,
        hot_pixel: bool = False,
        valid_exts: np.ndarray | None = None,
    ) -> Any:
        """
        Load background.

        Parameters
        ----------
        pile_up : bool
            Whether pile-up correction should be applied.
        hot_pixel : bool
            Whether hot-pixel correction should be applied.
        valid_exts : np.ndarray | None
            Allowed file extensions for detector loading.

        Returns
        -------
        Any
            Object produced by load background.
        """
        if not self.bg_path:
            return None
        _exts = valid_exts or (
            ".tif",
            ".tiff",
            ".hdf5",
            ".h5",
            ".sdt",
            ".mat",
            ".npy",
            ".txt",
            ".asc",
        )
        if os.path.isdir(self.bg_path):
            return self._load_folder(
                self.bg_path,
                sub_bg=False,
                pile_up=pile_up,
                hot_pixel=hot_pixel,
                mode="mean",
                valid_exts=_exts,
            )
        return self._load_file(self.bg_path, pile_up=pile_up, hot_pixel=hot_pixel)

    def _load_mask(self) -> Any:
        """
        Load mask.

        Returns
        -------
        Any
            Object produced by load mask.
        """
        if not self.mask_path:
            return None
        raw = self._load_file(self.mask_path)
        if raw is None:
            return None
        if raw.ndim == 3:
            raw = np.mean(raw, axis=-1)
        return (raw > 0).astype(bool)

    # ================================================================= #
    #  DATASET PACKAGING
    # ================================================================= #

    def _package(
        self,
        decay: np.ndarray,
        irf: np.ndarray,
        background: np.ndarray,
        mask: np.ndarray,
        name: str,
        source: str,
        **processing: Any,
    ) -> dict[Any, Any]:
        """
        Run the package routine.

        Parameters
        ----------
        decay : np.ndarray
            Time-resolved decay signal or decay cube.
        irf : np.ndarray
            Instrument response function aligned with the decay signal.
        background : np.ndarray
            Background array packaged with the loaded dataset.
        mask : np.ndarray
            Boolean or labeled mask selecting pixels for the operation.
        name : str
            Dataset, experiment, figure, or output name.
        source : str
            Source label recorded with the loaded dataset.
        **processing : Any
            Additional keyword options forwarded to the underlying implementation.

        Returns
        -------
        dict[Any, Any]
            Dictionary containing the data produced by package.
        """
        if decay is not None and irf is not None:
            if decay.shape[-1] != irf.shape[-1]:
                logging.warning(
                    f"[WARN] Temporal mismatch: decay {decay.shape[-1]} bins, IRF {irf.shape[-1]} bins."
                )
        return {
            "name": name,
            "source": source,
            "raw_data": {
                "decay": decay,
                "irf": irf,
                "background": background,
                "mask": mask,
            },
            "metadata": {
                "shape": decay.shape if decay is not None else None,
                "processing": processing,
            },
            "result": {
                "maps": {"tau1_map": None, "tau2_map": None},
                "TR_maps": {"fit_map": None, "residuals_map": None},
            },
        }

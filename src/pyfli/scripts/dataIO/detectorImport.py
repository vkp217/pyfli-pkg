import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .dataops_static import Staticdataops as ds


class Detector:
    """
    Detector-specific FLIM data loader.
        det = Detector(data_path=..., irf_path=..., bg_path=..., hp_path=..., bit_size=10)
        dataset = det.SS3()      # SwissSPAD3 SPAD array — HDF5 (file or folder)
        dataset = det.ICCD()     # Intensified CCD — folder of TIFF files
        dataset = det.BH_TCSPC() # Becker & Hickl TCSPC — SDT / ASC / MAT
        dataset = det.generic()  # any format, all corrections opt-in

    bit_size : int
        Photon counter depth (SS3) or ADC bit depth (ICCD) used in pile-up
        correction.  Dynamic range N = 2^bit_size − 1.  Default 10.
    """

    def __init__(self,
                 data_path=None,
                 irf_path=None,
                 bg_path=None,
                 mask_path=None,
                 hp_path=None,
                 bit_size=10):
        self.data_path = data_path
        self.irf_path  = irf_path
        self.bg_path   = bg_path
        self.mask_path = mask_path
        self.hp_path   = hp_path
        self.bit_size  = bit_size

    # ================================================================= #
    #  DETECTOR METHODS
    # ================================================================= #

    def SS3(self, name="Experiment_1", sub_bg=True, pile_up=True,
            hot_pixel=True, make_hp_map=True, threshold_sigma=5.0):
        """
        SwissSPAD3 SPAD array detector.

        Mode 1 — Already-processed data (single HDF5 file per stream):
            Reads decay and IRF from individual HDF5 files.
            Applies pile-up correction only.
            Raises if decay.shape != irf.shape when both are provided.

        Mode 2 — SSLive acquisition (folder of HDF5 files per stream):
            sub_bg=True  : bg_path (folder of HDF5) required.
                           Background cube = mean of all bg files → (H, W, T).
            make_hp_map  : derive hot pixel map from background cube (MAD threshold).
            hp_path      : alternatively load mask from PNG / JPEG / TIFF; auto-rotated
                           if shape is (W, H) instead of (H, W).
            Per-file processing order: hot_pixel → pile_up → sub_bg.
            Accumulated frames are summed → (H, W, T).
            Same corrections applied to IRF folder if provided.

        Parameters
        ----------
        sub_bg          : subtract mean background cube from each frame.
        pile_up         : apply photon-counter pile-up correction per frame.
        hot_pixel       : replace flagged pixels with 3×3 nanmedian neighbourhood.
        make_hp_map     : build hot pixel map from bg_path (folder mode only).
        threshold_sigma : MAD rejection threshold for hot pixel detection (default 5).
        """
        if not self.data_path or not os.path.exists(self.data_path):
            raise ValueError("SS3: data_path must be provided and must exist.")

        data_is_folder = os.path.isdir(self.data_path)

        # ---- Mode 1: Already-processed single HDF5 file(s) ------------------
        if not data_is_folder:
            if not self.data_path.lower().endswith(('.h5', '.hdf5')):
                raise ValueError(
                    f"SS3: data_path must be an HDF5 file or folder, got: {self.data_path}")
            decay = self._read_ss3_hdf5(self.data_path, pile_up=pile_up)
            irf = None
            if self.irf_path:
                if not os.path.isfile(self.irf_path) or \
                        not self.irf_path.lower().endswith(('.h5', '.hdf5')):
                    raise ValueError(
                        f"SS3: irf_path must be an HDF5 file, got: {self.irf_path}")
                irf = self._read_ss3_hdf5(self.irf_path, pile_up=pile_up)
            if decay is not None and irf is not None and decay.shape != irf.shape:
                raise ValueError(
                    f"SS3: Decay shape {decay.shape} does not match IRF shape {irf.shape}.")
            mask = self._load_mask()
            return self._package(decay, irf, None, mask, name, source="SwissSPAD3",
                                 sub_bg=False, pile_up=pile_up, hot_pixel=False,
                                 bit_size=self.bit_size)

        # ---- Mode 2: SSLive folder of HDF5 files ----------------------------
        # Background cube: mean over all bg files → (H, W, T)
        bg_cube = None
        if sub_bg:
            if not self.bg_path:
                raise ValueError(
                    "SS3: bg_path (folder of HDF5 files) is required when sub_bg=True.")
            bg_cube = self._load_ss_folder(self.bg_path, self._read_ss3_hdf5, mode='mean')

        # Hot pixel map
        hp_map = None
        if hot_pixel:
            if make_hp_map:
                if not self.bg_path:
                    raise ValueError(
                        "SS3: bg_path is required when make_hp_map=True.")
                from .dataIO_utils import DataIO_utils
                hp_map, _, _, _ = DataIO_utils().detect_hot_pixels(
                    self.bg_path, threshold_sigma=threshold_sigma
                )
            elif self.hp_path:
                ref_shape = self._ss_first_frame_shape(self.data_path, self._read_ss3_hdf5)
                hp_map = ds.load_hp_image(self.hp_path, ref_shape)

        # Process decay folder: hot_pixel → pile_up → sub_bg → sum → (H, W, T)
        decay = self._process_ss_folder(
            self.data_path, self._read_ss3_hdf5, hp_map, pile_up, sub_bg, bg_cube)

        # Process IRF folder with the same corrections
        irf = None
        if self.irf_path:
            if not os.path.isdir(self.irf_path):
                raise ValueError(
                    "SS3: irf_path must be a folder when data_path is a folder.")
            irf = self._process_ss_folder(
                self.irf_path, self._read_ss3_hdf5, hp_map, pile_up, sub_bg, bg_cube)

        mask = self._load_mask()
        return self._package(decay, irf, bg_cube, mask, name, source="SwissSPAD3",
                             sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel,
                             make_hp_map=make_hp_map, bit_size=self.bit_size,
                             threshold_sigma=threshold_sigma)

    def SS2(self, name="Experiment_1", sub_bg=True, pile_up=True,
            hot_pixel=True, make_hp_map=True, threshold_sigma=5.0):
        """
        SwissSPAD2 SPAD array detector.

        Identical pipeline to SS3() — see SS3 docstring for full details.
        The only difference is the HDF5 gate key structure:
            SS2: 'Gate Images' → 'Gate N'  (N = 1, 2, …)
            SS3: 'Gate Images' → 'Bottom G2 Gate N'  (N = 0, 1, …)
        """
        if not self.data_path or not os.path.exists(self.data_path):
            raise ValueError("SS2: data_path must be provided and must exist.")

        data_is_folder = os.path.isdir(self.data_path)

        # ---- Mode 1: Already-processed single HDF5 file(s) ------------------
        if not data_is_folder:
            if not self.data_path.lower().endswith(('.h5', '.hdf5')):
                raise ValueError(
                    f"SS2: data_path must be an HDF5 file or folder, got: {self.data_path}")
            decay = self._read_ss2_hdf5(self.data_path, pile_up=pile_up)
            irf = None
            if self.irf_path:
                if not os.path.isfile(self.irf_path) or \
                        not self.irf_path.lower().endswith(('.h5', '.hdf5')):
                    raise ValueError(
                        f"SS2: irf_path must be an HDF5 file, got: {self.irf_path}")
                irf = self._read_ss2_hdf5(self.irf_path, pile_up=pile_up)
            if decay is not None and irf is not None and decay.shape != irf.shape:
                raise ValueError(
                    f"SS2: Decay shape {decay.shape} does not match IRF shape {irf.shape}.")
            mask = self._load_mask()
            return self._package(decay, irf, None, mask, name, source="SwissSPAD2",
                                 sub_bg=False, pile_up=pile_up, hot_pixel=False,
                                 bit_size=self.bit_size)

        # ---- Mode 2: SSLive folder of HDF5 files ----------------------------
        bg_cube = None
        if sub_bg:
            if not self.bg_path:
                raise ValueError(
                    "SS2: bg_path (folder of HDF5 files) is required when sub_bg=True.")
            bg_cube = self._load_ss_folder(self.bg_path, self._read_ss2_hdf5, mode='mean')

        hp_map = None
        if hot_pixel:
            if make_hp_map:
                if not self.bg_path:
                    raise ValueError(
                        "SS2: bg_path is required when make_hp_map=True.")
                from .dataIO_utils import DataIO_utils
                hp_map, _, _, _ = DataIO_utils().detect_hot_pixels(
                    self.bg_path, threshold_sigma=threshold_sigma
                )
            elif self.hp_path:
                ref_shape = self._ss_first_frame_shape(self.data_path, self._read_ss2_hdf5)
                hp_map = ds.load_hp_image(self.hp_path, ref_shape)

        decay = self._process_ss_folder(
            self.data_path, self._read_ss2_hdf5, hp_map, pile_up, sub_bg, bg_cube)

        irf = None
        if self.irf_path:
            if not os.path.isdir(self.irf_path):
                raise ValueError(
                    "SS2: irf_path must be a folder when data_path is a folder.")
            irf = self._process_ss_folder(
                self.irf_path, self._read_ss2_hdf5, hp_map, pile_up, sub_bg, bg_cube)

        mask = self._load_mask()
        return self._package(decay, irf, bg_cube, mask, name, source="SwissSPAD2",
                             sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel,
                             make_hp_map=make_hp_map, bit_size=self.bit_size,
                             threshold_sigma=threshold_sigma)

    def ICCD(self, name="Experiment_1"):
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
                    f"ICCD: IRF must be 3D (H, W, T), got shape {irf.shape}.")
            if irf.shape != decay.shape:
                raise ValueError(
                    f"ICCD: IRF shape {irf.shape} does not match data shape {decay.shape}.")

        mask = self._load_mask()

        return self._package(decay, irf, None, mask, name, source="ICCD")

    def BH_TCSPC(self, name="Experiment_1", sub_bg=True, channel=0):
        """
        Time-Correlated Single Photon Counting detectors.

        Supported formats: .sdt (Becker & Hickl / PicoQuant), .asc, .mat, .npy, .tif

        Pile-up correction is NOT applied.  TCSPC pile-up follows a dead-time
        model  C_true = C_meas / (1 − C_meas · τ_dead · f_rep)  that is
        detector-dependent and must be applied externally when needed.

        channel : SDT measurement block index for multi-block files (default 0).
        """
        decay = self._dispatch(self.data_path, sub_bg=sub_bg,
                               pile_up=False, hot_pixel=False, channel=channel)
        irf   = self._dispatch(self.irf_path, sub_bg=False,
                               pile_up=False, hot_pixel=False, channel=channel)
        bg    = self._load_background(pile_up=False, hot_pixel=False)
        mask  = self._load_mask()

        return self._package(decay, irf, bg, mask, name, source="BH-TCSPC",
                             sub_bg=sub_bg, pile_up=False,
                             hot_pixel=False, channel=channel)

    def generic(self, name="Experiment_1", sub_bg=True,
                pile_up=False, hot_pixel=False):
        """
        Generic loader: TIFF / NPY / MAT / TXT / HDF5.
        All corrections opt-in.
        """
        decay = self._dispatch(self.data_path, sub_bg=sub_bg,
                               pile_up=pile_up, hot_pixel=hot_pixel)
        irf   = self._dispatch(self.irf_path, sub_bg=False,
                               pile_up=pile_up, hot_pixel=hot_pixel)
        bg    = self._load_background(pile_up=pile_up, hot_pixel=hot_pixel)
        mask  = self._load_mask()

        return self._package(decay, irf, bg, mask, name, source="Generic",
                             sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)

    # ================================================================= #
    #  ICCD-SPECIFIC LOADERS
    # ================================================================= #

    def _load_iccd_folder(self, folder_path):
        """
        Loads sorted TIFF files as gate images → (H, W, N_gates) float32.
        No corrections applied.
        """
        files = sorted(f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.tif', '.tiff')))
        if not files:
            raise FileNotFoundError(f"No TIFF files in: {folder_path}")

        full_paths = [os.path.join(folder_path, f) for f in files]

        first_raw = ds.load_tiff_file(full_paths[0]).astype(np.float32)
        first = first_raw.mean(axis=-1) if first_raw.ndim == 3 else first_raw
        H, W  = first.shape

        stack = np.zeros((H, W, len(files)), dtype=np.float32)
        stack[:, :, 0] = first

        def _read_gate(args):
            idx, path = args
            raw = ds.load_tiff_file(path).astype(np.float32)
            return idx, raw.mean(axis=-1) if raw.ndim == 3 else raw

        if len(files) > 1:
            tasks = list(enumerate(full_paths[1:], start=1))
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                for idx, gate in tqdm(ex.map(_read_gate, tasks),
                                      total=len(tasks),
                                      desc="Loading ICCD gates", leave=False):
                    if gate is not None and gate.shape == (H, W):
                        stack[:, :, idx] = gate

        return stack

    # ================================================================= #
    #  SPAD PIPELINE HELPERS  (shared by SS2 and SS3)
    # ================================================================= #

    def _load_ss_folder(self, folder_path, reader_fn, mode='mean'):
        """Load all HDF5 files via reader_fn, stack (n,H,W,T), return mean or sum."""
        files = sorted(f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.h5', '.hdf5')))
        if not files:
            raise FileNotFoundError(f"No HDF5 files in: {folder_path}")
        accumulated = np.stack(
            [reader_fn(os.path.join(folder_path, f), pile_up=False)
             for f in tqdm(files,
                           desc=f"Loading {os.path.basename(folder_path)}",
                           leave=False)],
            axis=0
        )  # (n, H, W, T)
        return np.mean(accumulated, axis=0) if mode == 'mean' else np.sum(accumulated, axis=0)

    def _process_ss_folder(self, folder_path, reader_fn, hp_map, pile_up, sub_bg, bg_cube):
        """
        Sequential per-file SPAD processing: hot_pixel → pile_up → sub_bg.
        Accumulates processed frames and returns their sum → (H, W, T).
        """
        files = sorted(f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.h5', '.hdf5')))
        if not files:
            raise FileNotFoundError(f"No HDF5 files in: {folder_path}")

        accumulated = []
        for fname in tqdm(files,
                          desc=f"Processing {os.path.basename(folder_path)}",
                          leave=False):
            frame = reader_fn(os.path.join(folder_path, fname), pile_up=False)
            if hp_map is not None:
                frame = self._correct_hotpixels(frame, hp_map)
            if pile_up:
                frame = ds.pileup_correction(frame, bit_size=self.bit_size)
            if sub_bg and bg_cube is not None:
                frame = np.maximum(frame - bg_cube, 0.0)
            accumulated.append(frame)

        return np.sum(np.stack(accumulated, axis=0), axis=0)  # (H, W, T)

    def _ss_first_frame_shape(self, folder_path, reader_fn):
        """Return (H, W) spatial shape from the first HDF5 file using reader_fn."""
        files = sorted(f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.h5', '.hdf5')))
        if not files:
            raise FileNotFoundError(f"No HDF5 files in: {folder_path}")
        frame = reader_fn(os.path.join(folder_path, files[0]), pile_up=False)
        return frame.shape[:2]

    # ================================================================= #
    #  GENERIC ROUTING  (SS3, TCSPC, generic)
    # ================================================================= #

    def _dispatch(self, path, sub_bg, pile_up, hot_pixel,
                  valid_exts=None, **kw):
        if not path:
            return None
        if not os.path.exists(path):
            print(f"[ERROR] Path not found: {os.path.abspath(path)}")
            return None
        _exts = valid_exts or ('.tif', '.tiff', '.hdf5', '.h5',
                               '.sdt', '.mat', '.npy', '.txt', '.asc')
        if os.path.isdir(path):
            return self._load_folder(path, sub_bg=sub_bg, pile_up=pile_up,
                                     hot_pixel=hot_pixel, valid_exts=_exts, **kw)
        return self._load_file(path, pile_up=pile_up, hot_pixel=hot_pixel, **kw)

    # ================================================================= #
    #  GENERIC FOLDER LOADING  (SS3, TCSPC, generic)
    # ================================================================= #

    def _load_folder(self, folder_path, sub_bg=True, pile_up=False,
                     hot_pixel=False, mode='sum',
                     valid_exts=('.tif','.tiff','.hdf5','.h5',
                                 '.sdt','.mat','.npy','.txt','.asc'),
                     **kw):
        files = sorted(f for f in os.listdir(folder_path)
                       if f.lower().endswith(valid_exts))
        if not files:
            raise FileNotFoundError(
                f"No files matching {valid_exts} in: {folder_path}")

        full_paths = [os.path.join(folder_path, f) for f in files]
        bg_avg = (self._load_background(pile_up=pile_up, hot_pixel=hot_pixel,
                                        valid_exts=valid_exts)
                  if sub_bg else None)

        first = self._load_file(full_paths[0], pile_up=pile_up,
                                hot_pixel=hot_pixel, **kw)
        if first is None:
            return None

        stack = np.zeros((*first.shape, len(files)), dtype=np.float32)
        stack[..., 0] = first

        if len(files) > 1:
            tasks = [(i, p, pile_up, hot_pixel, kw)
                     for i, p in enumerate(full_paths[1:], start=1)]
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
                for idx, data in tqdm(ex.map(self._file_task, tasks),
                                      total=len(tasks),
                                      desc="Loading folder", leave=False):
                    if data is not None and data.shape == first.shape:
                        stack[..., idx] = data

        if bg_avg is not None:
            for i in range(stack.shape[-1]):
                if bg_avg.shape == stack[..., i].shape:
                    stack[..., i] -= bg_avg
                else:
                    print(f"[WARN] BG shape {bg_avg.shape} ≠ frame '{files[i]}' "
                          f"shape {stack[..., i].shape} — subtraction skipped.")
            stack = np.maximum(stack, 0)

        # Collapse file axis: 4D (H,W,T,N_files) → 3D (H,W,T)
        # 3D stacks (from 2D files) are returned as-is
        if stack.ndim == 4:
            return np.sum(stack, axis=-1) if mode == 'sum' else np.mean(stack, axis=-1)
        return stack

    def _file_task(self, args):
        idx, path, pile_up, hot_pixel, kw = args
        return idx, self._load_file(path, pile_up=pile_up, hot_pixel=hot_pixel, **kw)

    # ================================================================= #
    #  SINGLE FILE LOADING
    # ================================================================= #

    def _load_file(self, file_path, pile_up=False, hot_pixel=False, channel=0, **_):
        if not file_path or not os.path.exists(file_path):
            return None
        ext = os.path.splitext(file_path)[-1].lower()
        try:
            # SwissSPAD3 — dedicated HDF5 reader, corrections applied inside
            if ext in ('.hdf5', '.h5'):
                return self._read_ss3_hdf5(file_path, pile_up=pile_up,
                                           hot_pixel=hot_pixel)

            # TCSPC SDT: channel selects measurement block
            if ext == '.sdt':
                from sdtfile import SdtFile
                return np.asarray(SdtFile(file_path).data[channel], dtype=np.float32)

            # PicoQuant ASCII (time col 0, counts col 1), tiled to spatial dims
            if ext == '.asc':
                return ds.load_asc_file(file_path).astype(np.float32)

            loaders = {
                '.mat':  ds.load_mat_file,
                '.npy':  ds.load_npy_file,
                '.tif':  ds.load_tiff_file,
                '.tiff': ds.load_tiff_file,
                '.txt':  ds.load_txt_file,
                # '.roiN': ds.load_roiN_file,  # TODO: register once loader is implemented
            }
            loader = loaders.get(ext)
            if loader is None:
                print(f"[WARN] Unsupported format '{ext}': {os.path.basename(file_path)}")
                return None

            data = loader(file_path).astype(np.float32)
            if pile_up:
                data = ds.pileup_correction(data, bit_size=self.bit_size)
            if hot_pixel:
                data = ds.apply_interpolation_mask(data, hp_path=self.hp_path)
            return data

        except Exception as e:
            print(f"[ERROR] {os.path.basename(file_path)}: {e}")
            return None

    # ================================================================= #
    #  SwissSPAD3 HDF5 READER
    # ================================================================= #

    def _read_ss3_hdf5(self, fname, pile_up=True, hot_pixel=False):
        # hot_pixel ignored — applied externally via ds.hotpixel_correct
        return ds.spad_hdf5_read(fname, 'Bottom G2 Gate',
                                 pile_up=pile_up, bit_size=self.bit_size)

    def _read_ss2_hdf5(self, fname, pile_up=True):
        return ds.spad_hdf5_read(fname, 'Gate ',
                                 pile_up=pile_up, bit_size=self.bit_size)

    def _correct_hotpixels(self, data_3d, hot_pixel_map):
        return ds.hotpixel_correct(data_3d, hot_pixel_map)

    # ================================================================= #
    #  BACKGROUND AND MASK
    # ================================================================= #

    def _load_background(self, pile_up=False, hot_pixel=False, valid_exts=None):
        if not self.bg_path:
            return None
        _exts = valid_exts or ('.tif', '.tiff', '.hdf5', '.h5',
                               '.sdt', '.mat', '.npy', '.txt', '.asc')
        if os.path.isdir(self.bg_path):
            return self._load_folder(self.bg_path, sub_bg=False,
                                     pile_up=pile_up, hot_pixel=hot_pixel,
                                     mode='mean', valid_exts=_exts)
        return self._load_file(self.bg_path, pile_up=pile_up, hot_pixel=hot_pixel)

    def _load_mask(self):
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

    def _package(self, decay, irf, background, mask, name, source, **processing):
        if decay is not None and irf is not None:
            if decay.shape[-1] != irf.shape[-1]:
                print(f"[WARN] Temporal mismatch: decay {decay.shape[-1]} bins, "
                      f"IRF {irf.shape[-1]} bins.")
        return {
            "name"    : name,
            "source"  : source,
            "raw_data": {"decay": decay, "irf": irf,
                         "background": background, "mask": mask},
            "metadata": {"shape"     : decay.shape if decay is not None else None,
                         "processing": processing},
            "result"  : {"maps"   : {"tau1_map": None, "tau2_map": None},
                         "TR_maps": {"fit_map": None, "residuals_map": None}},
        }

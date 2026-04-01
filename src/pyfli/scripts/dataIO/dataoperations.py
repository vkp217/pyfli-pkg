# new data operations file class
import os
import numpy as np
import tifffile
from scipy.io import loadmat
from sdtfile import SdtFile
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Import the static logic from your utility file
from .dataops_static import Staticdataops as ds

class DataOperations:
    def __init__(self, fli_path=None, irf_path=None, bg_path=None, mask_path=None, hp_path=None):
        """
        Initializes the DataOperations class with paths to relevant data files.
        """
        self.fli_path = fli_path
        self.irf_path = irf_path
        self.bg_path = bg_path
        self.mask_path = mask_path
        self.hp_path = hp_path 

        # --- MODULAR REGISTRY ---
        self.loader_registry = {
            '.mat': ds.load_mat_file,
            '.sdt': ds.load_sdt_file,
            '.tif': ds.load_tiff_file,
            '.tiff': ds.load_tiff_file,
            '.npy': ds.load_npy_file,
            '.txt': ds.load_txt_file,
            '.asc': ds.load_asc_file
        }

    # --- PUBLIC API ---

    def load_fli(self, sub_bg=True, pile_up=False, hot_pixel=False):
        print(f"Initiating FLI load from: {self.fli_path}")
        return self._general_loader(self.fli_path, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel, label="FLI")

    def load_background(self, pile_up=False, hot_pixel=False):
        """Loads background. If folder, returns the mean average of all files."""
        if self.bg_path and os.path.isdir(self.bg_path):
            print(f'Background FOLDER detected: {self.bg_path}')
            return self._load_from_folder(self.bg_path, 
                                          sub_bg=False, 
                                          pile_up=pile_up, 
                                          hot_pixel=True, 
                                          mode='mean', 
                                          is_background=True, 
                                          label="BG")        
        if self.bg_path:
            print(f'Background FILE detected: {self.bg_path}')
            return self._general_loader(self.bg_path, 
                                        sub_bg=False, 
                                        pile_up=pile_up, 
                                        hot_pixel=hot_pixel, 
                                        label="BG")
        
        print('No background path provided.')
        return None

    def load_irf(self, sub_bg=False, pile_up=False, hot_pixel=False):
        print(f"Initiating IRF load from: {self.irf_path}")
        return self._general_loader(self.irf_path, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel, label="IRF")

    def load_all_parallel(self, sub_bg=True, pile_up=False, hot_pixel=False):
        print("Starting synchronized parallel loading for FLI, IRF, and BG...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            fli_future = executor.submit(self.load_fli, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
            irf_future = executor.submit(self.load_irf, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
            bg_future  = executor.submit(self.load_background, pile_up=pile_up, hot_pixel=hot_pixel)
            
            return fli_future.result(), irf_future.result(), bg_future.result()

    def make_dataset(self, 
                     name="Experiment_1", 
                     source="ICCD", 
                     sub_bg=True, 
                     pile_up=False, 
                     hot_pixel=False):
        if all([self.fli_path, self.irf_path, self.bg_path]):
            fli, irf, background = self.load_all_parallel(sub_bg=sub_bg, 
                                                          pile_up=pile_up, 
                                                          hot_pixel=hot_pixel)
        else:
            background = self.load_background(pile_up=pile_up, hot_pixel=hot_pixel) if self.bg_path else None
            fli = self.load_fli(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel) if self.fli_path else None
            irf = self.load_irf(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel) if self.irf_path else None
        
        mask = self.load_mask()
        return {
            "name": name, 
            "source": source, 
            "raw_data": {"decay": fli, 
                         "irf": irf, 
                         "background": background, 
                         "mask": mask},
            "metadata": {
                "shape": fli.shape if fli is not None else None, 
                "processing": {"bg_sub": sub_bg, 
                               "pile_up": pile_up, 
                               "hot_pixel": hot_pixel}
            },
            "result": {"maps":{
                                "tau1_map": None, 
                                "tau2_map": None
                                }, 
                       "TR_maps":{
                                "fit_map": None,
                                "residuals_maps": None
                                }
                        }
        }

    def load_mask(self):
        if not self.mask_path: return None
        print(f"Loading mask from: {self.mask_path}")
        mask = self._load_single_file(self.mask_path)
        if mask is None: return None
        if mask.ndim == 3: mask = np.mean(mask, axis=-1)
        return (mask > np.min(mask)).astype(bool)

    def _general_loader(self, 
                        path, 
                        sub_bg=True, 
                        pile_up=False, 
                        hot_pixel=False, 
                        hp_path=None, 
                        mode='sum', 
                        label="Data"):
        if not path or not os.path.exists(path):
            return None
        
        if os.path.isfile(path):
            print(f"Loading single file: {os.path.basename(path)}")
            return self._load_single_file(path, pile_up, hot_pixel, hp_path)
        else:
            print(f"Loading folder: {os.path.basename(path)}")
            return self._load_from_folder(path, sub_bg, pile_up, hot_pixel, hp_path, mode, label=label)

    def _load_single_file(self, file_path, pile_up=False, hot_pixel=False, active_hp=None):
        ext = os.path.splitext(file_path)[-1].lower()
        active_hp = active_hp or self.hp_path
        
        # 1. Specialized Logic for HDF5 (due to internal processing flags)
        if ext in ('.hdf5', '.h5'):
            return ds.SS3HDF5read(file_path, pileCorr=pile_up, hot_pixels=hot_pixel, hp_path=active_hp)
        
        # 2. Modular Registry Lookup
        loader_func = self.loader_registry.get(ext)
        if not loader_func:
            print(f"[WARN] No loader registered for extension: {ext}")
            return None
        try:
            data = loader_func(file_path)
            
            # 3. Universal Post-Processing
            if data is not None:
                if pile_up: 
                    data = ds.pileup_correction(data)
                if hot_pixel: 
                    # print(f" Applying hot-pixel removal to: {os.path.basename(file_path)}")
                    data = ds.apply_interpolation_mask(data, hp_path=active_hp)
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            return None

    def _load_from_folder(self, 
                          folder_path, 
                          sub_bg=True, 
                          pile_up=False, 
                          hot_pixel=False, 
                          active_hp=None, 
                          mode='sum', 
                          is_background=False, 
                          label="Data", 
                          progress_callback=None):
        
        # Dynamically determine valid extensions based on the registry
        valid_exts = ('.tif', '.tiff', '.hdf5', '.h5') # at this time these files are uploaded from folder
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
        
        if not files: 
            raise FileNotFoundError(f"No valid files found in {folder_path}")

        full_paths = [os.path.join(folder_path, f) for f in files]
        
        # Handle background subtraction if requested
        bg_avg = self.load_background(pile_up=pile_up, hot_pixel=hot_pixel) if (sub_bg and not is_background) else None

        # Load first file to establish shape
        first = self._load_single_file(full_paths[0], pile_up, hot_pixel, active_hp)
        if first is None: return None
        
        stack = np.zeros((*first.shape, len(files)), dtype=first.dtype)
        stack[..., 0] = first

        # Parallel load remaining files
        if len(files) > 1:
            args = [(i, p, pile_up, hot_pixel, active_hp) for i, p in enumerate(full_paths[1:], start=1)]
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(tqdm(executor.map(self._load_single_file_parallel, args), 
                                    total=len(args), 
                                    desc=f"Loading {label}", 
                                    leave=False))
                for idx, data in results:
                    if data is not None and data.shape == first.shape:
                        stack[..., idx] = data
                    if progress_callback:
                        progress_callback(int((idx / len(files)) * 100))

        # Perform background subtraction across the stack
        if bg_avg is not None:
            for i in range(stack.shape[-1]):
                if bg_avg.shape == stack[..., i].shape:
                    stack[..., i] -= bg_avg

        # Final reduction
        if is_background:
            return np.mean(stack, axis=-1)

        if stack.ndim == 4:
            return np.sum(stack, axis=-1) if mode == 'sum' else np.mean(stack, axis=-1)
        
        return stack

    def _load_single_file_parallel(self, args):
        """Helper for ThreadPoolExecutor mapping."""
        idx, path, pile_up, hot_pixel, active_hp = args
        data = self._load_single_file(path, pile_up, hot_pixel, active_hp)
        return idx, data

    def load_mask(self):
        if not self.mask_path: return None
        print(f"Loading mask from: {self.mask_path}")
        mask = self._load_single_file(self.mask_path)
        if mask is None: return None
        if mask.ndim == 3: mask = np.mean(mask, axis=-1)
        return (mask > np.min(mask)).astype(bool)
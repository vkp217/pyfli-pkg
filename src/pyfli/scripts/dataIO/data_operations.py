import sys
import os
import numpy as np
import tifffile
import h5py
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sdtfile import SdtFile
from concurrent.futures import ThreadPoolExecutor
import importlib.resources
from pathlib import Path
from tqdm import tqdm

class DataOperations:
    def __init__(self, fli_path=None, irf_path=None, bg_path=None, mask_path=None, hp_path=None):
        self.fli_path = fli_path
        self.irf_path = irf_path
        self.bg_path = bg_path 
        self.mask_path = mask_path
        self.hp_path = hp_path 

    # --- PUBLIC LOADERS ---
    
    def load_fli(self, sub_bg=True, pile_up=False, hot_pixel=False):
        print(f"[DEBUG] Initiating FLI load from: {self.fli_path}")
        return self._general_loader(self.fli_path, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel, label="FLI")

    def load_background(self, pile_up=False, hot_pixel=False):
        """Loads background. If folder, returns the mean average of all files."""
        if self.bg_path and os.path.isdir(self.bg_path):
            print(f'[DEBUG] Background FOLDER detected: {self.bg_path}')
            return self._load_from_folder(
                self.bg_path, sub_bg=False, pile_up=pile_up, 
                hot_pixel=hot_pixel, mode='mean', is_background=True, label="BG"
            )
        
        if self.bg_path:
            print(f'[DEBUG] Background FILE detected: {self.bg_path}')
            return self._general_loader(self.bg_path, sub_bg=False, pile_up=pile_up, hot_pixel=hot_pixel, label="BG")
        
        print('[DEBUG] No background path provided.')
        return None

    def load_irf(self, sub_bg=False, pile_up=False, hot_pixel=False):
        print(f"[DEBUG] Initiating IRF load from: {self.irf_path}")
        return self._general_loader(self.irf_path, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel, label="IRF")

    def load_all_parallel(self, sub_bg=True, pile_up=False, hot_pixel=False):
        print("[DEBUG] Starting synchronized parallel loading for FLI, IRF, and BG...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            fli_future = executor.submit(self.load_fli, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
            irf_future = executor.submit(self.load_irf, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
            bg_future  = executor.submit(self.load_background, pile_up=pile_up, hot_pixel=hot_pixel)
            
            return fli_future.result(), irf_future.result(), bg_future.result()

    def load_dataset(self, name="Experiment_1", source="ICCD", sub_bg=True, pile_up=False, hot_pixel=False):
        if all([self.fli_path, self.irf_path, self.bg_path]):
            fli, irf, background = self.load_all_parallel(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
        else:
            background = self.load_background(pile_up=pile_up, hot_pixel=hot_pixel) if self.bg_path else None
            fli = self.load_fli(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel) if self.fli_path else None
            irf = self.load_irf(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel) if self.irf_path else None
        
        mask = self.load_mask()
        return {
            "name": name, "source": source, "raw_data": {"decay": fli, "irf": irf, "background": background, "mask": mask},
            "metadata": {"shape": fli.shape if fli is not None else None, "processing": {"bg_sub": sub_bg, "pile_up": pile_up, "hot_pixel": hot_pixel}},
            "result": {"maps": {"tau1_map": None, "tau2_map": None}, "TR_maps": {"fit": None, "residuals": None}}
        }

    # --- INTERNAL CORE LOGIC ---

    def _general_loader(self, path, sub_bg=True, pile_up=False, hot_pixel=False, hp_path=None, mode='sum', label="Data"):
        if not path or not os.path.exists(path):
            return None
        active_hp = hp_path if hp_path else self.hp_path
        
        if os.path.isfile(path):
            print(f"[DEBUG] Loading single file: {os.path.basename(path)}")
            return self._load_single_file(path, pile_up, hot_pixel, active_hp)
        else:
            print(f"[DEBUG] Loading folder: {os.path.basename(path)}")
            return self._load_from_folder(path, sub_bg, pile_up, hot_pixel, active_hp, mode, label=label)
            
    def _load_single_file(self, file_path, pile_up=False, hot_pixel=False, active_hp=None):
        ext = os.path.splitext(file_path)[-1].lower() 
        try:
            if ext == '.mat':
                mat_data = loadmat(file_path, squeeze_me=True)
                keys = [k for k in mat_data.keys() if not k.startswith('__')]
                data = np.asarray(mat_data[keys[0]])
            elif ext == '.npy':
                data = np.load(file_path)
            elif ext == '.sdt':
                data = np.asarray(SdtFile(file_path).data[0])
            elif ext in ('.hdf5', '.h5'):
                return ss3HDF5read_optimized(file_path, pileCorr=pile_up, hot_pixels=hot_pixel, hp_path=active_hp)
            elif ext in ('.tif', '.tiff'):
                data = np.asarray(tifffile.imread(file_path))
            elif ext == '.txt':
                data = np.loadtxt(file_path)
                if data.ndim == 1: data = np.tile(data.reshape(1, 1, -1), (512, 512, 1))
            elif ext == '.asc':
                data_read = np.genfromtxt(file_path)
                data_1d = data_read[:, 1] if data_read.ndim == 2 else data_read.flatten()
                data = np.tile(data_1d, (512, 512, 1))
            else:
                return None
            
            if data is not None and ext not in ('.hdf5', '.h5'):
                if pile_up: 
                    data = pileup_correction(data)
                if hot_pixel: 
                    print(f"[DEBUG] Applying hot-pixel removal to: {os.path.basename(file_path)}")
                    data = apply_interpolation_mask(data, hp_path=active_hp)
            return data
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
            return None

    def _load_single_file_parallel(self, args):
        idx, path, pile_up, hot_pixel, active_hp = args
        data = self._load_single_file(path, pile_up, hot_pixel, active_hp)
        return idx, data

    def _load_from_folder(self, folder_path, sub_bg=True, pile_up=False, 
                            hot_pixel=False, active_hp=None, mode='sum', 
                            is_background=False, label="Data", progress_callback=None):
        # valid_exts = ('.tif', '.tiff', '.hdf5', '.h5', '.sdt', '.mat', '.npy', '.txt')
        valid_exts = ('.tif', '.tiff', '.hdf5', '.h5')
        files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
        if not files: raise FileNotFoundError(f"No valid files found in {folder_path}")

        full_paths = [os.path.join(folder_path, f) for f in files]
        
        bg_avg = self.load_background(pile_up=pile_up, hot_pixel=hot_pixel) if (sub_bg and not is_background) else None
        if bg_avg is not None:
            print(f"[DEBUG] Subtraction active for folder {os.path.basename(folder_path)}. Master background ready.")

        first = self._load_single_file(full_paths[0], pile_up, hot_pixel, active_hp)
        if first is None: return None
        
        stack = np.zeros((*first.shape, len(files)), dtype=first.dtype)
        stack[..., 0] = first

        if len(files) > 1:
            args = [(i, p, pile_up, hot_pixel, active_hp) for i, p in enumerate(full_paths[1:], start=1)]
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                # Use tqdm to track parallel file loading
                results = list(tqdm(executor.map(self._load_single_file_parallel, args), 
                                    total=len(args), 
                                    desc=f"[DEBUG] Loading {label}", 
                                    leave=False))
                for idx, data in results:
                    if data is not None and data.shape == first.shape:
                        stack[..., idx] = data

                    if progress_callback:
                        current_progress = int((idx / len(files)) * 100)
                        progress_callback(current_progress)

        if bg_avg is not None:
            for i in range(stack.shape[-1]):
                if bg_avg.shape == stack[..., i].shape:
                    stack[..., i] -= bg_avg
                else:
                    print(f"[WARN] Shape mismatch: BG {bg_avg.shape} vs File {files[i]} {stack[..., i].shape}")

        if is_background:
            print(f"[DEBUG] Averaging {len(files)} files to create Master Background map.")
            return np.mean(stack, axis=-1)

        if stack.ndim == 4:
            print(f"[DEBUG] Collapsing 4D folder stack into 3D volume using mode: {mode}")
            return np.sum(stack, axis=-1) if mode == 'sum' else np.mean(stack, axis=-1)
        
        return stack

    def load_mask(self):
        if not self.mask_path: return None
        print(f"[DEBUG] Loading mask from: {self.mask_path}")
        mask = self._load_single_file(self.mask_path)
        if mask is None: return None
        if mask.ndim == 3: mask = np.mean(mask, axis=-1)
        return (mask > np.min(mask)).astype(bool)

# --- STANDALONE UTILITIES ---

def pileup_correction(data, bit_size=10):
    dynamic_range = 2**bit_size - 1    
    safe_data = np.clip(data / dynamic_range, 0, 0.9999)
    return -np.log(1 - safe_data) * dynamic_range

def apply_interpolation_mask(data_3d, bit_size=10, hp_path=None):
    print(f"[DEBUG] --- Starting Hot-Pixel Removal ---")
    cleaned_data = np.copy(data_3d)
    
    try:
        image_path_ref = importlib.resources.files('pyfli.scripts.dataIO').joinpath('Mask_5p100.PNG')
        source_info = "package resources"
    except Exception as e:
        image_path_ref = Path(__file__).parent / "Mask_5p100.PNG"
        source_info = "local directory fallback"
    
    final_mask_path = hp_path if hp_path else image_path_ref
    print(f"[DEBUG] Targeting Hot-pixel Mask file: {final_mask_path} (Source: {source_info})")

    if not os.path.exists(final_mask_path):
        print(f"[ERROR] Hot-pixel Mask file NOT found at {final_mask_path}. Skipping correction.")
        return data_3d

    mask_2d = plt.imread(final_mask_path)
    if mask_2d is None:
        print("[ERROR] Failed to decode Hot-pixel Mask image. Returning raw data.")
        return data_3d
    
    print(f"[DEBUG] Hot-pixel Mask loaded successfully. Shape: {mask_2d.shape}")

    hot_pixels = np.where(np.transpose(mask_2d) > 0) 
    y_coords, x_coords = hot_pixels
    num_hot_pixels = len(y_coords)
    
    print(f"[DEBUG] Detected {num_hot_pixels} hot pixels to interpolate.")

    # Use tqdm to track hot-pixel processing loop
    for i, (y, x) in enumerate(tqdm(zip(y_coords, x_coords), 
                                   total=num_hot_pixels, 
                                   desc="[DEBUG] Removing Hotpixels", 
                                   leave=False)):
        y_min, y_max = max(0, y-1), min(data_3d.shape[0], y+2)
        x_min, x_max = max(0, x-1), min(data_3d.shape[1], x+2)
        neighborhood = data_3d[y_min:y_max, x_min:x_max, :]
        cleaned_data[y, x, :] = np.median(neighborhood, axis=(0, 1))

    print(f"[DEBUG] --- Hot-Pixel Removal Complete ---")
    return cleaned_data

def ss3HDF5read_optimized(fname, pileCorr=True, hot_pixels=True, hp_path=None):
    try:
        with h5py.File(fname, 'r') as f:
            gate_grp = f.get('Gate Images')
            if not gate_grp: return None
            g2_keys = sorted([k for k in gate_grp.keys() if k.startswith('Bottom G2 Gate')], 
                             key=lambda x: int(x.split('Gate ')[-1]))
            tpsfs = np.zeros((*gate_grp[g2_keys[0]].shape, len(g2_keys)), dtype=np.float32)
            for i, key in enumerate(g2_keys):
                tpsfs[:, :, i] = gate_grp[key][:]
            
            if pileCorr: 
                print(f"[DEBUG] HDF5 Internal: Applying Pileup Correction to {os.path.basename(fname)}")
                tpsfs = pileup_correction(tpsfs)
            if hot_pixels: 
                print(f"[DEBUG] HDF5 Internal: Applying Hot-pixel Removal to {os.path.basename(fname)}")
                tpsfs = apply_interpolation_mask(tpsfs, hp_path=hp_path)
            return tpsfs
    except: return None


import matplotlib.pyplot as plt
if __name__ == '__main__':
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../MyPyFli_UI'))
    fold_path = os.path.join(parent_dir,"data/mouseR_740bp")
    print(fold_path)
    loader = DataOperations(
        fli_path= fold_path
    )
    tpsf = loader.load_fli()
    print(f'the shape of the tpsf is {tpsf.shape}')
    plt.imshow(np.sum(tpsf, axis=2))
    plt.show()
    




# import sys
# import os
# import numpy as np
# import tifffile
# from scipy.ndimage import binary_opening, binary_closing
# from scipy.ndimage import generic_filter
# from scipy.io import loadmat
# from sdtfile import SdtFile
# import h5py
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# import matplotlib.pyplot as plt 
# import importlib.resources
# from pathlib import Path

# class DataOperations:
#     """
#     High-level interface for loading:
#     - FLI data
#     - IRF data
#     - Background data
#     - Binary masks

#     Supports:
#     - Single file
#     - Folder-based stack
#     - HDF5 accumulation logic
#     """

#     def __init__(self, fli_path=None, irf_path=None,
#                  bk_path=None, mask_path=None):

#         self.fli_path = fli_path
#         self.irf_path = irf_path
#         self.bk_path = bk_path
#         self.mask_path = mask_path


#     # INTERNAL LOADERS

#     def _general_loader(self, path):
#         if not path:
#             return None

#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Path '{path}' does not exist.")

#         if os.path.isfile(path):
#             return self._load_single_file(path)

#         elif os.path.isdir(path):
#             return self._load_from_folder(path)

#         return None

#     def _load_single_file(self, file_path):
#         ext = os.path.splitext(file_path)[1].lower()

#         try:
#             # MATLAB
#             if ext == '.mat':
#                 mat_data = loadmat(file_path, squeeze_me=True)
#                 keys = [k for k in mat_data.keys()
#                         if not k.startswith('__')]
#                 # print(keys)
#                 if not keys:
#                     raise ValueError("No valid data in MAT file.")
#                 return np.asarray(mat_data[keys[0]])

#             # NumPy
#             elif ext == '.npy':
#                 return np.load(file_path)

#             # SDT
#             elif ext == '.sdt':
#                 if SdtFile is None:
#                     raise ImportError("sdtfile not installed.")
#                 return np.asarray(SdtFile(file_path).data[0])

#             # HDF5
#             # HDF5
#             elif ext in ('.hdf5', '.h5'):
#                 if ss3HDF5read_optimized is None:  
#                     raise ImportError("ss3HDF5read_optimized not available.")          

#                 is_accumulated = '[Accumulated]' in file_path                
#                 use_correction = not is_accumulated                
#                 arr = ss3HDF5read_optimized(file_path, pileCorr=use_correction,  hot_pixels=use_correction )
                
#                 return np.asarray(arr)

#             # TIFF
#             elif ext in ('.tif', '.tiff'):
#                 return np.asarray(tifffile.imread(file_path))

#             # TXT (special handling)
#             elif ext == '.txt':
#                 data = np.loadtxt(file_path)
#                 data = np.asarray(data)

#                 # If 1D → tile to 512x512xT
#                 if data.ndim == 1:
#                     T = data.shape[0]
#                     data = np.tile(
#                         data.reshape(1, 1, T),
#                         (512, 512, 1)
#                     )

#                 elif data.ndim == 2:
#                     pass  # assume image

#                 elif data.ndim == 3:
#                     pass

#                 else:
#                     raise ValueError(
#                         f"Unsupported TXT dimensionality: {data.ndim}"
#                     )

#                 return data

#             else:
#                 raise ValueError(f"Unsupported file format '{ext}'")

#         except Exception as e:
#             raise RuntimeError(
#                 f"Failed loading {file_path}: {e}"
#             )

#     def _load_single_file_parallel(self, args):
#         """Helper for parallel loading: returns (index, data)."""
#         i, file_path = args
#         return i, self._load_single_file(file_path)

#     def _load_from_folder(self, folder_path, is_background=False):
#         """
#         Load all valid files from a folder and accumulate (sum) them.
        
#         OPTIMIZATION: Uses ThreadPoolExecutor for parallel I/O-bound file loading.
        
#         FIX: Added `is_background` flag so background folders average instead of sum,
#         and do NOT apply self-subtraction when loading background files.
#         """
#         valid_exts = ('.tif', '.tiff', '.hdf5', '.h5')
        
#         # 1. Handle Background if path is provided AND we are not already loading the background
#         bg_avg = None
#         if self.bk_path is not None and not is_background:  # FIX: avoid recursive bg subtraction
#             if os.path.isfile(self.bk_path):
#                 bg_avg = self._load_single_file(self.bk_path)
#             elif os.path.isdir(self.bk_path):
#                 bg_files = sorted([f for f in os.listdir(self.bk_path) if f.lower().endswith(valid_exts)])
#                 if bg_files:
#                     first_bg = self._load_single_file(os.path.join(self.bk_path, bg_files[0]))
#                     bg_stack = np.zeros((*first_bg.shape, len(bg_files)), dtype=first_bg.dtype)
#                     for i, f in enumerate(bg_files):
#                         bg_stack[..., i] = self._load_single_file(os.path.join(self.bk_path, f))
#                     bg_avg = np.mean(bg_stack, axis=-1)
#                     print(f'bg_avg shape is {bg_avg.shape}')

#         # 2. Load Main Data Stack
#         files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
#         if not files:
#             raise FileNotFoundError(f"No valid files found in {folder_path}")
            
#         full_paths = [os.path.join(folder_path, f) for f in files]

#         # OPTIMIZATION: Parallel I/O loading using ThreadPoolExecutor
#         first = self._load_single_file(full_paths[0])
#         print(f'shape of first is {first.shape}')
#         stack = np.zeros((*first.shape, len(files)), dtype=first.dtype)
#         stack[..., 0] = first

#         if len(files) > 1:
#             args = [(i, p) for i, p in enumerate(full_paths[1:], start=1)]
#             with ThreadPoolExecutor() as executor:
#                 for i, data in executor.map(self._load_single_file_parallel, args):
#                     stack[..., i] = data
#                     print(f'shape of stack i = {i} is {stack[..., i].shape}')
#         print(f'shape of stack is {stack.shape}')
#         # Apply background subtraction if bg_avg exists
#         if bg_avg is not None:
#             for i in range(stack.shape[-1]):
#                 if bg_avg.shape != stack[..., i].shape:
#                     raise ValueError(f"Background shape {bg_avg.shape} mismatch with data shape {stack[..., i].shape}")
#                 stack[..., i] = stack[..., i] - bg_avg

#         # accumulated (summed) result across the file dimension
#         if stack.ndim ==3:
#             stack_accumulated = stack.copy()
#         else:
#             stack_accumulated = np.sum(stack, axis=-1)
#         print(f'stack accumulated shape is {stack_accumulated.shape}')
#         return stack_accumulated


#     # PUBLIC LOADERS
#     def load_fli(self):
#         data = self._general_loader(self.fli_path)
#         if data is None:
#             return None

#         # HDF5 folder → sum axis=3
#         if self.fli_path and os.path.isdir(self.fli_path):
#             if data.ndim >= 4:
#                 return np.squeeze(np.sum(data, axis=3))

#         return data

#     def load_background(self):
#         # FIX: Pass is_background=True so background folder loading skips self-subtraction
#         if self.bk_path and os.path.isdir(self.bk_path):
#             data = self._load_from_folder(self.bk_path, is_background=True)
#         else:
#             data = self._general_loader(self.bk_path)
#         if data is None:
#             return None

#         # HDF5 folder → average axis=3
#         if self.bk_path and os.path.isdir(self.bk_path):
#             if data.ndim >= 4:
#                 return np.squeeze(np.average(data, axis=3))

#         return data

#     def load_irf(self):
#         data = self._general_loader(self.irf_path)
#         if data is None:
#             return None

#         # HDF5 folder → sum axis=3
#         if self.irf_path and os.path.isdir(self.irf_path):
#             if data.ndim >= 4:
#                 return np.squeeze(np.sum(data, axis=3))

#         return data

#     def load_all_parallel(self):
#         """
#         OPTIMIZATION: Load FLI, IRF, and background concurrently using threads.
#         Returns (fli, irf, background) tuple.
#         """
#         with ThreadPoolExecutor(max_workers=3) as executor:
#             fli_future = executor.submit(self.load_fli)
#             irf_future = executor.submit(self.load_irf)
#             bk_future  = executor.submit(self.load_background)
#             fli = fli_future.result()
#             irf = irf_future.result()
#             bk  = bk_future.result()
#         return fli, irf, bk

#     # MASK LOADER
#     def load_mask(self,
#                   validate_shape=True,
#                   broadcast_2d_to_3d=True,
#                   strict_binary=False):

#         path = self.mask_path
#         if not os.path.isfile(path):
#             raise ValueError("Mask must be a file.")
#         ext = os.path.splitext(path)[1].lower()
#         if ext in ('.tif', '.tiff'):
#             mask = tifffile.imread(path)
#         elif ext == '.npy':
#             mask = np.load(path)
#         elif ext == '.txt':
#             mask = np.loadtxt(path)
#         else:
#             raise ValueError("Unsupported mask format.")
#         mask = np.asarray(mask)

#         if strict_binary:
#             unique_vals = np.unique(mask)
#             if not np.all(np.isin(unique_vals, [0, 1])):
#                 raise ValueError("Mask not binary.")
#         mask = mask.astype(bool)
#         if validate_shape:
#             fli = self.load_fli()
#             if fli is not None:
#                 if mask.shape == fli.shape:
#                     return mask

#                 if (broadcast_2d_to_3d and
#                         mask.ndim == 2 and
#                         fli.ndim == 3 and
#                         mask.shape == fli.shape[:2]):

#                     mask = np.repeat(
#                         mask[..., np.newaxis],
#                         fli.shape[2],
#                         axis=2
#                     )
#                     return mask

#                 raise ValueError(
#                     f"Mask shape {mask.shape} "
#                     f"!= FLI shape {fli.shape}"
#                 )
#         return mask

#     def mask_implementation(self, fli, irf, threshold=None):
#         # fli = self.load_fli()
#         # irf = self.load_irf() 
        
#         if threshold is None:
#             mask = self.load_mask(validate_shape=True,
#                                  broadcast_2d_to_3d=True,
#                                  strict_binary=False)
#         else:
#             intensity = np.sum(fli, axis=-1)
#             mask = intensity > threshold 
        
#         mask = mask.astype(bool)
#         if mask.ndim < fli.ndim:
#             mask_expanded = mask[..., np.newaxis]
#             masked_fli = fli * mask_expanded
#             masked_irf = irf * mask_expanded
#         else:
#             masked_fli = fli * mask
#             masked_irf = irf * mask

#         return masked_fli, masked_irf, mask

# def ss3HDF5read_optimized(fname, pileCorr = True, hot_pixels = True):
#     try:
#         with h5py.File(fname, 'r') as f:
#             gate_grp = f.get('Gate Images')
#             if not gate_grp:
#                 return None

#             # Get keys once to avoid repeated OS calls
#             all_keys = list(gate_grp.keys())
            
#             # Use list comprehension for faster filtering
#             g2_keys = sorted([k for k in all_keys if k.startswith('Bottom G2 Gate')], 
#                              key=lambda x: int(x.split('Gate ')[-1]))
#             int_keys = sorted([k for k in all_keys if k.startswith('Bottom INT Gate')], 
#                               key=lambda x: int(x.split('Gate ')[-1]))

#             if not g2_keys: return None

#             # Get shape from the first dataset
#             sample_shape = gate_grp[g2_keys[0]].shape
            
#             # Pre-allocate using float32 for speed (switch to float64 if necessary)
#             tpsfs1 = np.zeros((*sample_shape, len(g2_keys)), dtype=np.float32)
#             for i, key in enumerate(g2_keys):
#                 tpsfs1[:, :, i] = gate_grp[key][:]
#             if pileCorr and tpsfs1 is not None:
#                 tpsfs1 = pileup_correction(tpsfs1, bit_size=10)
#             if hot_pixels and tpsfs1 is not None:
#                 tpsfs1 = apply_interpolation_mask(tpsfs1, bit_size=10)
#             return tpsfs1

#     except Exception as e:
#         print(f"Error processing {fname}: {e}")
#         return None

# def pileup_correction(data, bit_size=10):
#     dynamic_range = 2**bit_size - 1    
#     # np.clip to avoid log(0) or log(negative) if data reaches dynamic_range
#     safe_data = np.clip(data / dynamic_range, 0, 0.9999)
#     data_output = -np.log(1 - safe_data) * dynamic_range
    
#     return data_output

# def apply_interpolation_mask(data_3d, bit_size=10): # Added bit_size to match call
#     cleaned_data = np.copy(data_3d)
#     image_path_ref = importlib.resources.files('pyfli.scripts.dataIO').joinpath('Mask_5p100.PNG')
#     print(f'the assets_dir is : {image_path_ref}')
#     mask_2d = plt.imread(image_path_ref)
#     if mask_2d is None:
#         print("Warning: Hotpixel mask not found.")
#         return data_3d
    
#     # binary_mask = np.where(mask_2d>0)
#     hot_pixels = np.where(np.transpose(mask_2d) > 0) 
    
#     for y, x in zip(*hot_pixels):
#         # Adjusting limits based on (H, W, T) structure
#         y_min, y_max = max(0, y-1), min(data_3d.shape[0], y+2)
#         x_min, x_max = max(0, x-1), min(data_3d.shape[1], x+2)
        
#         neighborhood = data_3d[y_min:y_max, x_min:x_max, :]
#         replacement_values = np.median(neighborhood, axis=(0, 1))
#         cleaned_data[y, x, :] = replacement_values
        
#     return cleaned_data

# def batch_process_folder(folder_path, max_workers=4):
#     """
#     Processes all HDF5 files in a folder in parallel.
#     """
#     files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.h5', '.hdf5'))]
    
#     # This uses multiple CPU cores to read different files at the same time
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         results = list(executor.map(ss3HDF5read_optimized, files))
    
#     return [r for r in results if r is not None]


# class SShotpixelMask:
#     """
#     Generates a hot pixel mask for SPAD arrays (like SwissSPAD) using log-normal statistics.
#     Hot pixels typically follow a distribution that is distinct from the 
#     log-normal distribution of the 'healthy' Dark Count Rate (DCR).
#     """
#     def __init__(self, bk_path=None):
#         """
#         :param bk_path: Path to a background file (.h5, .npy) or folder of background files.
#         """
#         self.bk_path = bk_path
#         self.dcr_map = None
#         self.mask = None
#         self.valid_exts = ('.h5', '.hdf5', '.npy')

#     def _load_data(self, path):
#         """Internal helper to load data for DCR mapping."""
#         ext = os.path.splitext(path)[1].lower()
#         if ext in ('.h5', '.hdf5'):
#             # Simplified HDF5 read for DCR mapping
#             with h5py.File(path, 'r') as f:
#                 # Adjust 'Gate Images' to match your specific HDF5 structure
#                 gate_grp = f.get('Gate Images')
#                 if gate_grp:
#                     keys = sorted(gate_grp.keys())
#                     # Load all gates and sum them to get total counts
#                     data = np.stack([gate_grp[k][:] for k in keys], axis=-1)
#                     return data
#         elif ext == '.npy':
#             return np.load(path)
#         return None

#     def gen_dcr_map(self):
#         """
#         Loads background data and collapses it into a 2D DCR (Dark Count Rate) map.
#         If a folder is provided, it accumulates all files within.
#         """
#         if self.bk_path is None:
#             raise ValueError("Background path (bk_path) must be provided.")

#         if os.path.isfile(self.bk_path):
#             data = self._load_data(self.bk_path)
#             # Collapse 3D (H, W, Gates) to 2D (H, W)
#             self.dcr_map = np.sum(data, axis=-1) if data.ndim == 3 else data
        
#         elif os.listdir(self.bk_path):
#             files = sorted([f for f in os.listdir(self.bk_path) 
#                             if f.lower().endswith(self.valid_exts)])
            
#             accumulated = None
#             for f in files:
#                 f_path = os.path.join(self.bk_path, f)
#                 data = self._load_data(f_path)
#                 # Sum over gates and then accumulate over files
#                 current_2d = np.sum(data, axis=-1) if data.ndim == 3 else data
                
#                 if accumulated is None:
#                     accumulated = current_2d.astype(np.float64)
#                 else:
#                     accumulated += current_2d
            
#             self.dcr_map = accumulated

#         if self.dcr_map is None:
#             raise RuntimeError(f"Could not generate DCR map from {self.bk_path}")
        
#         print(f"DCR map generated with shape: {self.dcr_map.shape}")
#         return self.dcr_map

#     def generate_hot_pixel_mask(self, sigma_threshold=3.0):
#         """
#         Identifies hot pixels using log-normal statistics.
#         """
#         if self.dcr_map is None:
#             self.gen_dcr_map()

#         # 1. Take log of DCR map, ignoring zeros/negatives
#         flat_dcr = self.dcr_map.flatten()
#         valid_indices = flat_dcr > 0
#         log_dcr = np.log(flat_dcr[valid_indices])

#         # 2. Stats in log-space
#         mean_log = np.mean(log_dcr)
#         std_log = np.std(log_dcr)

#         # 3. Calculate threshold and flag outliers
#         threshold_linear = np.exp(mean_log + (sigma_threshold * std_log))

#         # Create binary mask (1 = hot pixel, 0 = healthy)
#         self.mask = (self.dcr_map > threshold_linear).astype(np.uint8)
        
#         num_hot = np.sum(self.mask)
#         percent_hot = (num_hot / self.mask.size) * 100
#         print(f"Mask created: {num_hot} hot pixels found ({percent_hot:.2f}%)")
        
#         return self.mask, threshold_linear

#     def save_mask(self, output_path):

#         if self.mask is None:
#             raise ValueError("No mask to save. Run generate_hot_pixel_mask first.")
        
#         ext = os.path.splitext(output_path)[1].lower()
        
#         if ext == '.npy':
#             np.save(output_path, self.mask)
#         else:
#             # plt.imsave handles 0-1 (float) or 0-255 (uint8)
#             # We use cmap='gray' to ensure it's saved as a grayscale image
#             plt.imsave(output_path, self.mask, cmap='gray')
            
#         print(f"Mask saved to {output_path}")

#     def apply_mask(self, data, fill_value=0):
#         """
#         Zeros out (or fills) hot pixels in a 2D or 3D data array.
#         """
#         if self.mask is None:
#             raise ValueError("Mask not generated.")
        
#         corrected = np.copy(data)
        
#         # Check if we are masking a 3D stack (H, W, Time) or a 2D image
#         if data.ndim == 3:
#             # Broadcast 2D mask across the 3rd dimension
#             corrected[self.mask == 1, :] = fill_value
#         else:
#             corrected[self.mask == 1] = fill_value
            
#         return corrected

    


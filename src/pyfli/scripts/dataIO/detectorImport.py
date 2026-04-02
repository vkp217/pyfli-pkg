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
from .dataops_static import Staticdataops as ds

class Detector:
    def __init__(self, 
                 data_path = None, 
                 irf_path = None,
                 bg_path = None,
                 mask_path = None,
                 hp_path = None,
                 detector_type = None):
        
        self.data_path = data_path
        self.irf_path = irf_path
        self.bg_path = bg_path
        self.mask_path = mask_path
        self.hp_path = hp_path
        self.detector_type = detector_type

        self.loader_registry = {
            '.mat': ds.load_mat_file,
            '.sdt': ds.load_sdt_file,
            '.tif': ds.load_tiff_file,
            '.tiff': ds.load_tiff_file,
            '.npy': ds.load_npy_file,
            '.txt': ds.load_txt_file,
            '.asc': ds.load_asc_file,
            '.roiN': ds.load_roiN_file ## I have to adde this format
        }

    # def SS3(self, bg_sub = True, pile_up = True ):


    # def _load_single_file_parallel(self, args):
    #     idx, path, pile_up, hot_pixel, active_hp = args
    #     return idx, self._load_single_file(path, pile_up, hot_pixel, active_hp)










# #     def __init__(self, fli_path=None, irf_path=None, bg_path=None, mask_path=None, hp_path=None):
# #         self.fli_path = fli_path
# #         self.irf_path = irf_path
# #         self.bg_path = bg_path 
# #         self.mask_path = mask_path
# #         self.hp_path = hp_path 

# #     # --- PUBLIC LOADERS ---
    
# #     def load_fli(self, sub_bg=True, pile_up=False, hot_pixel=False):
# #         print(f"[DEBUG] Initiating FLI load from: {self.fli_path}")
# #         return self._general_loader(self.fli_path, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel, label="FLI")

# #     def load_background(self, pile_up=False, hot_pixel=False):
# #         """Loads background. If folder, returns the mean average of all files."""
# #         if self.bg_path and os.path.isdir(self.bg_path):
# #             print(f'[DEBUG] Background FOLDER detected: {self.bg_path}')
# #             return self._load_from_folder(
# #                 self.bg_path, sub_bg=False, pile_up=pile_up, 
# #                 hot_pixel=hot_pixel, mode='mean', is_background=True, label="BG"
# #             )
        
# #         if self.bg_path:
# #             print(f'[DEBUG] Background FILE detected: {self.bg_path}')
# #             return self._general_loader(self.bg_path, sub_bg=False, pile_up=pile_up, hot_pixel=hot_pixel, label="BG")
        
# #         print('[DEBUG] No background path provided.')
# #         return None

# #     def load_irf(self, sub_bg=False, pile_up=False, hot_pixel=False):
# #         print(f"[DEBUG] Initiating IRF load from: {self.irf_path}")
# #         return self._general_loader(self.irf_path, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel, label="IRF")

# #     def load_all_parallel(self, sub_bg=True, pile_up=False, hot_pixel=False):
# #         print("[DEBUG] Starting synchronized parallel loading for FLI, IRF, and BG...")
# #         with ThreadPoolExecutor(max_workers=3) as executor:
# #             fli_future = executor.submit(self.load_fli, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
# #             irf_future = executor.submit(self.load_irf, sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
# #             bg_future  = executor.submit(self.load_background, pile_up=pile_up, hot_pixel=hot_pixel)
            
# #             return fli_future.result(), irf_future.result(), bg_future.result()

# #     def load_dataset(self, name="Experiment_1", source="ICCD", sub_bg=True, pile_up=False, hot_pixel=False):
# #         if all([self.fli_path, self.irf_path, self.bg_path]):
# #             fli, irf, background = self.load_all_parallel(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel)
# #         else:
# #             background = self.load_background(pile_up=pile_up, hot_pixel=hot_pixel) if self.bg_path else None
# #             fli = self.load_fli(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel) if self.fli_path else None
# #             irf = self.load_irf(sub_bg=sub_bg, pile_up=pile_up, hot_pixel=hot_pixel) if self.irf_path else None
        
# #         mask = self.load_mask()
# #         return {
# #             "name": name, "source": source, "raw_data": {"decay": fli, "irf": irf, "background": background, "mask": mask},
# #             "metadata": {"shape": fli.shape if fli is not None else None, "processing": {"bg_sub": sub_bg, "pile_up": pile_up, "hot_pixel": hot_pixel}},
# #             "result": {"maps": {"tau1_map": None, "tau2_map": None}, "TR_maps": {"fit": None, "residuals": None}}
# #         }

# #     # --- INTERNAL CORE LOGIC ---

# #     def _general_loader(self, path, sub_bg=True, pile_up=False, hot_pixel=False, hp_path=None, mode='sum', label="Data"):
# #         if not path or not os.path.exists(path):
# #             return None
# #         active_hp = hp_path if hp_path else self.hp_path
        
# #         if os.path.isfile(path):
# #             print(f"[DEBUG] Loading single file: {os.path.basename(path)}")
# #             return self._load_single_file(path, pile_up, hot_pixel, active_hp)
# #         else:
# #             print(f"[DEBUG] Loading folder: {os.path.basename(path)}")
# #             return self._load_from_folder(path, sub_bg, pile_up, hot_pixel, active_hp, mode, label=label)
            
# #     def _load_single_file(self, file_path, pile_up=False, hot_pixel=False, active_hp=None):
# #         ext = os.path.splitext(file_path)[-1].lower() 
# #         try:
# #             if ext == '.mat':
# #                 mat_data = loadmat(file_path, squeeze_me=True)
# #                 keys = [k for k in mat_data.keys() if not k.startswith('__')]
# #                 data = np.asarray(mat_data[keys[0]])
# #             elif ext == '.npy':
# #                 data = np.load(file_path)
# #             elif ext == '.sdt':
# #                 data = np.asarray(SdtFile(file_path).data[0])
# #             elif ext in ('.hdf5', '.h5'):
# #                 return ss3HDF5read_optimized(file_path, pileCorr=pile_up, hot_pixels=hot_pixel, hp_path=active_hp)
# #             elif ext in ('.tif', '.tiff'):
# #                 data = np.asarray(tifffile.imread(file_path))
# #             elif ext == '.txt':
# #                 data = np.loadtxt(file_path)
# #                 if data.ndim == 1: data = np.tile(data.reshape(1, 1, -1), (512, 512, 1))
# #             elif ext == '.asc':
# #                 data_read = np.genfromtxt(file_path)
# #                 data_1d = data_read[:, 1] if data_read.ndim == 2 else data_read.flatten()
# #                 data = np.tile(data_1d, (512, 512, 1))
# #             else:
# #                 return None
            
# #             if data is not None and ext not in ('.hdf5', '.h5'):
# #                 if pile_up: 
# #                     data = pileup_correction(data)
# #                 if hot_pixel: 
# #                     print(f"[DEBUG] Applying hot-pixel removal to: {os.path.basename(file_path)}")
# #                     data = apply_interpolation_mask(data, hp_path=active_hp)
# #             return data
# #         except Exception as e:
# #             print(f"[ERROR] Failed to load {file_path}: {e}")
# #             return None

# #     def _load_single_file_parallel(self, args):
# #         idx, path, pile_up, hot_pixel, active_hp = args
# #         data = self._load_single_file(path, pile_up, hot_pixel, active_hp)
# #         return idx, data

# #     def _load_from_folder(self, folder_path, sub_bg=True, pile_up=False, 
# #                             hot_pixel=False, active_hp=None, mode='sum', 
# #                             is_background=False, label="Data", progress_callback=None):
# #         # valid_exts = ('.tif', '.tiff', '.hdf5', '.h5', '.sdt', '.mat', '.npy', '.txt')
# #         valid_exts = ('.tif', '.tiff', '.hdf5', '.h5')
# #         files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])
# #         if not files: raise FileNotFoundError(f"No valid files found in {folder_path}")

# #         full_paths = [os.path.join(folder_path, f) for f in files]
        
# #         bg_avg = self.load_background(pile_up=pile_up, hot_pixel=hot_pixel) if (sub_bg and not is_background) else None
# #         if bg_avg is not None:
# #             print(f"[DEBUG] Subtraction active for folder {os.path.basename(folder_path)}. Master background ready.")

# #         first = self._load_single_file(full_paths[0], pile_up, hot_pixel, active_hp)
# #         if first is None: return None
        
# #         stack = np.zeros((*first.shape, len(files)), dtype=first.dtype)
# #         stack[..., 0] = first

# #         if len(files) > 1:
# #             args = [(i, p, pile_up, hot_pixel, active_hp) for i, p in enumerate(full_paths[1:], start=1)]
# #             with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
# #                 # Use tqdm to track parallel file loading
# #                 results = list(tqdm(executor.map(self._load_single_file_parallel, args), 
# #                                     total=len(args), 
# #                                     desc=f"[DEBUG] Loading {label}", 
# #                                     leave=False))
# #                 for idx, data in results:
# #                     if data is not None and data.shape == first.shape:
# #                         stack[..., idx] = data

# #                     if progress_callback:
# #                         current_progress = int((idx / len(files)) * 100)
# #                         progress_callback(current_progress)

# #         if bg_avg is not None:
# #             for i in range(stack.shape[-1]):
# #                 if bg_avg.shape == stack[..., i].shape:
# #                     stack[..., i] -= bg_avg
# #                 else:
# #                     print(f"[WARN] Shape mismatch: BG {bg_avg.shape} vs File {files[i]} {stack[..., i].shape}")

# #         if is_background:
# #             print(f"[DEBUG] Averaging {len(files)} files to create Master Background map.")
# #             return np.mean(stack, axis=-1)

# #         if stack.ndim == 4:
# #             print(f"[DEBUG] Collapsing 4D folder stack into 3D volume using mode: {mode}")
# #             return np.sum(stack, axis=-1) if mode == 'sum' else np.mean(stack, axis=-1)
        
# #         return stack

# #     def load_mask(self):
# #         if not self.mask_path: return None
# #         print(f"[DEBUG] Loading mask from: {self.mask_path}")
# #         mask = self._load_single_file(self.mask_path)
# #         if mask is None: return None
# #         if mask.ndim == 3: mask = np.mean(mask, axis=-1)
# #         return (mask > np.min(mask)).astype(bool)

# # # --- STANDALONE UTILITIES ---

# # def pileup_correction(data, bit_size=10):
# #     dynamic_range = 2**bit_size - 1    
# #     safe_data = np.clip(data / dynamic_range, 0, 0.9999)
# #     return -np.log(1 - safe_data) * dynamic_range

# # def apply_interpolation_mask(data_3d, bit_size=10, hp_path=None):
# #     print(f"[DEBUG] --- Starting Hot-Pixel Removal ---")
# #     cleaned_data = np.copy(data_3d)
    
# #     try:
# #         image_path_ref = importlib.resources.files('pyfli.scripts.dataIO').joinpath('Mask_5p100.PNG')
# #         source_info = "package resources"
# #     except Exception as e:
# #         image_path_ref = Path(__file__).parent / "Mask_5p100.PNG"
# #         source_info = "local directory fallback"
    
# #     final_mask_path = hp_path if hp_path else image_path_ref
# #     print(f"[DEBUG] Targeting Hot-pixel Mask file: {final_mask_path} (Source: {source_info})")

# #     if not os.path.exists(final_mask_path):
# #         print(f"[ERROR] Hot-pixel Mask file NOT found at {final_mask_path}. Skipping correction.")
# #         return data_3d

# #     mask_2d = plt.imread(final_mask_path)
# #     if mask_2d is None:
# #         print("[ERROR] Failed to decode Hot-pixel Mask image. Returning raw data.")
# #         return data_3d
    
# #     print(f"[DEBUG] Hot-pixel Mask loaded successfully. Shape: {mask_2d.shape}")

# #     hot_pixels = np.where(np.transpose(mask_2d) > 0) 
# #     y_coords, x_coords = hot_pixels
# #     num_hot_pixels = len(y_coords)
    
# #     print(f"[DEBUG] Detected {num_hot_pixels} hot pixels to interpolate.")

# #     # Use tqdm to track hot-pixel processing loop
# #     for i, (y, x) in enumerate(tqdm(zip(y_coords, x_coords), 
# #                                    total=num_hot_pixels, 
# #                                    desc="[DEBUG] Removing Hotpixels", 
# #                                    leave=False)):
# #         y_min, y_max = max(0, y-1), min(data_3d.shape[0], y+2)
# #         x_min, x_max = max(0, x-1), min(data_3d.shape[1], x+2)
# #         neighborhood = data_3d[y_min:y_max, x_min:x_max, :]
# #         cleaned_data[y, x, :] = np.median(neighborhood, axis=(0, 1))

# #     print(f"[DEBUG] --- Hot-Pixel Removal Complete ---")
# #     return cleaned_data

# # def ss3HDF5read_optimized(fname, pileCorr=True, hot_pixels=True, hp_path=None):
# #     try:
# #         with h5py.File(fname, 'r') as f:
# #             gate_grp = f.get('Gate Images')
# #             if not gate_grp: return None
# #             g2_keys = sorted([k for k in gate_grp.keys() if k.startswith('Bottom G2 Gate')], 
# #                              key=lambda x: int(x.split('Gate ')[-1]))
# #             tpsfs = np.zeros((*gate_grp[g2_keys[0]].shape, len(g2_keys)), dtype=np.float32)
# #             for i, key in enumerate(g2_keys):
# #                 tpsfs[:, :, i] = gate_grp[key][:]
            
# #             if pileCorr: 
# #                 print(f"[DEBUG] HDF5 Internal: Applying Pileup Correction to {os.path.basename(fname)}")
# #                 tpsfs = pileup_correction(tpsfs)
# #             if hot_pixels: 
# #                 print(f"[DEBUG] HDF5 Internal: Applying Hot-pixel Removal to {os.path.basename(fname)}")
# #                 tpsfs = apply_interpolation_mask(tpsfs, hp_path=hp_path)
# #             return tpsfs
# #     except: return None


# # import matplotlib.pyplot as plt
# # if __name__ == '__main__':
# #     parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../MyPyFli_UI'))
# #     fold_path = os.path.join(parent_dir,"data/mouseR_740bp")
# #     print(fold_path)
# #     loader = DataOperations(
# #         fli_path= fold_path
# #     )
# #     tpsf = loader.load_fli()
# #     print(f'the shape of the tpsf is {tpsf.shape}')
# #     plt.imshow(np.sum(tpsf, axis=2))
# #     plt.show()
    


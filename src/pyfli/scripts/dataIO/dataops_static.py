# dataIO/dataops_static.py
import os
import numpy as np
import h5py
import tifffile
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sdtfile import SdtFile
from pathlib import Path

class Staticdataops:
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
    def apply_interpolation_mask(data_3d, hp_path=None):
        """
        Identifies hot pixels from a mask and replaces them with the median 
        of their 3x3 neighborhood (excluding the hot pixel itself).
        """
        if not hp_path:
            raise ValueError("Hotpixel removal mask path (hp_path) is not provided.")
        
        # Load mask and ensure it is 2D grayscale
        hotpixel_mask = plt.imread(hp_path)
        if hotpixel_mask.ndim == 3:
            hotpixel_mask = hotpixel_mask[..., 0]
            
        # Handle shape mismatches or transpositions
        if data_3d.shape[:2] != hotpixel_mask.shape:
            if data_3d.shape[0] == hotpixel_mask.shape[1] and data_3d.shape[1] == hotpixel_mask.shape[0]:
                hotpixel_mask = hotpixel_mask.T
            else:
                raise ValueError(f"Shape mismatch: data {data_3d.shape[:2]} vs mask {hotpixel_mask.shape}")

        cleaned_data = np.copy(data_3d)
        y_coords, x_coords = np.where(hotpixel_mask > 0)
        
        for y, x in zip(y_coords, x_coords):
            # Define 3x3 neighborhood bounds
            y_min, y_max = max(0, y-1), min(data_3d.shape[0], y+2)
            x_min, x_max = max(0, x-1), min(data_3d.shape[1], x+2)
            
            # Extract neighborhood and mask the center pixel to avoid biasing the median
            neighborhood = data_3d[y_min:y_max, x_min:x_max, :].copy()
            local_y, local_x = y - y_min, x - x_min
            neighborhood[local_y, local_x, :] = np.nan 
            
            # Use nanmedian to ignore the masked center pixel
            cleaned_data[y, x, :] = np.nanmedian(neighborhood, axis=(0, 1))
            
        return cleaned_data

    @staticmethod
    def load_mat_file(path):
        mat_data = loadmat(path, squeeze_me=True)
        # Filter out metadata keys
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        return np.asarray(mat_data[keys[0]])

    @staticmethod
    def load_sdt_file(path):
        return np.asarray(SdtFile(path).data[0])

    @staticmethod
    def load_tiff_file(path):
        return np.asarray(tifffile.imread(path))

    @staticmethod
    def load_npy_file(path):
        return np.load(path)

    @staticmethod
    def load_txt_file(path, target_spatial=(512, 512)):
        data = np.loadtxt(path)
        if data.ndim == 1: 
            # Reshape 1D IRF/Trace to 3D and tile across spatial dimensions
            data = np.tile(data.reshape(1, 1, -1), (*target_spatial, 1))
        return data

    @staticmethod
    def load_asc_file(path, target_spatial=(512, 512)):
        data_read = np.genfromtxt(path)
        data_1d = data_read[:, 1] if data_read.ndim == 2 else data_read.flatten()
        return np.tile(data_1d.reshape(1, 1, -1), (*target_spatial, 1))

    @staticmethod
    def SS3HDF5read(fname, pileCorr=True, hot_pixels=True, hp_path=None):
        """
        Reads gated HDF5 data, optionally applying pileup and hotpixel corrections.
        """
        # Critical Check: Prevent loading if correction is requested but path is missing
        if hot_pixels and hp_path is None:
            raise ValueError("hp_path must be provided when hot_pixels=True.")

        try:
            with h5py.File(fname, 'r') as f:
                gate_grp = f.get('Gate Images')
                if not gate_grp:
                    print("Error: 'Gate Images' group not found in HDF5.")
                    return None
                
                # Sort gates numerically (Gate 0, Gate 1, etc.)
                g2_keys = sorted(
                    [k for k in gate_grp.keys() if k.startswith('Bottom G2 Gate')], 
                    key=lambda x: int(x.split('Gate ')[-1])
                )
                
                # Pre-allocate 3D array (Height, Width, Time/Gates)
                first_gate = gate_grp[g2_keys[0]]
                tpsfs = np.zeros((*first_gate.shape, len(g2_keys)), dtype=np.float32)
                
                for i, key in enumerate(g2_keys):
                    tpsfs[:, :, i] = gate_grp[key][:]
                
                # Apply sequence of corrections
                if pileCorr: 
                    tpsfs = Staticdataops.pileup_correction(tpsfs)
                
                if hot_pixels: 
                    tpsfs = Staticdataops.apply_interpolation_mask(tpsfs, hp_path=hp_path)
                
                return tpsfs

        except Exception as e:
            # If our internal logic raised a ValueError, pass it through. 
            # Otherwise, catch unexpected I/O errors.
            if isinstance(e, ValueError):
                raise e
            print(f"HDF5 Load Error: {e}")
            return None
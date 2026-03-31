# dataIO/dataops_static.py
import os
import numpy as np
import h5py
import tifffile
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sdtfile import SdtFile
import importlib.resources
from pathlib import Path
from tqdm import tqdm

class Staticdataops:
    @staticmethod
    def pileup_correction(data, bit_size=10):
        dynamic_range = 2**bit_size - 1    
        safe_data = np.clip(data / dynamic_range, 0, 0.9999)
        return -np.log(1 - safe_data) * dynamic_range

    @staticmethod
    def apply_interpolation_mask(data_3d, bit_size=10, hp_path=None):
        cleaned_data = np.copy(data_3d)
        if not hp_path:
            raise ValueError('Hotpixel removal mask is not provided')
        hotpixel_mask = plt.imread(hp_path)
        hot_pixels = np.where(np.transpose(hotpixel_mask) > 0) 
        y_coords, x_coords = hot_pixels    
        for y, x in zip(y_coords, x_coords):
            y_min, y_max = max(0, y-1), min(data_3d.shape[0], y+2)
            x_min, x_max = max(0, x-1), min(data_3d.shape[1], x+2)
            neighborhood = data_3d[y_min:y_max, x_min:x_max, :]
            cleaned_data[y, x, :] = np.median(neighborhood, axis=(0, 1))
        return cleaned_data

    @staticmethod
    def load_mat_file(path):
        mat_data = loadmat(path, squeeze_me=True)
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
    def load_txt_file(path):
        data = np.loadtxt(path)
        if data.ndim == 1: 
            data = np.tile(data.reshape(1, 1, -1), (512, 512, 1))
        return data

    @staticmethod
    def load_asc_file(path):
        data_read = np.genfromtxt(path)
        data_1d = data_read[:, 1] if data_read.ndim == 2 else data_read.flatten()
        return np.tile(data_1d, (512, 512, 1))

    @staticmethod
    def SS3HDF5read(fname, pileCorr=True, hot_pixels=True, hp_path=None):
        # This remains specialized as it handles its own internal processing
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
                    tpsfs = Staticdataops.pileup_correction(tpsfs)
                if hot_pixels: 
                    tpsfs = Staticdataops.apply_interpolation_mask(tpsfs, hp_path=hp_path)
                return tpsfs
        except Exception as e:
            print(f"HDF5 Load Error: {e}")
            return None
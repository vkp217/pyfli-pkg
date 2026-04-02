# dataIO_utils.py 
import h5py
import numpy as np
import json

class DataIO_utils:
    def __init__(self):
        pass

    def load_phasors_hdf5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            Gc = f['Gc'][:]
            Sc = f['Sc'][:]
            tau = f['tau'][:] if 'tau' in f else None
        if tau is not None:
            if Gc.shape[1:] != tau.shape:
                raise ValueError(f"Dimension mismatch: Phasor spatial size {Gc.shape[1:]} "
                    f"does not match Tau size {tau.shape}.")
            if Gc.shape != Sc.shape:
                raise ValueError("Critical Error: Gc and Sc dimensions do not match.")
        return Gc, Sc, tau
    
    def roiNloader(self, map_array, file_path, visualize=True):
        if map_array.ndim == 3:
            H, W, _ = map_array.shape
        elif map_array.ndim == 2:
            H, W = map_array.shape
        else:
            raise ValueError('Correct data map is not provided')
        mask = np.zeros((H, W), dtype=bool)
        with open(file_path, 'r') as fid:
            J = json.load(fid)
        p = J.get("Named ROI Descriptions", [])
        for roi in p:
            try:
                contours = roi["ROI Descriptor"]["Contours"]
                for contour in contours:
                    coords = contour["Coordinates"]
                    if len(coords) >= 2:
                        x = int(coords[0])
                        y = int(coords[1])
                        if 0 <= y < H and 0 <= x < W:
                            mask[y, x] = True
            except KeyError:
                continue
        return mask
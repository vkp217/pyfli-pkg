# dataIO_utils.py
import os
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

    def detect_hot_pixels(self, bg_path, threshold_sigma=5.0, save_path=None):
        """
        Finding hot pixels from an SS3 background HDF5 file or folder 
            total_counts > median + threshold_sigma × 1.4826 × MAD
        equivalent threshold_sigma=5 ≈ 5σ rejection for Gaussian data. 
        """
        def _read_raw(fname):
            with h5py.File(fname, 'r') as f:
                gate_grp = f.get('Gate Images')
                if gate_grp is None:
                    raise KeyError(f"'Gate Images' not found in {fname}")
                g2_keys = sorted(
                    (k for k in gate_grp.keys() if k.startswith('Bottom G2 Gate')),
                    key=lambda k: int(k.split('Gate ')[-1])
                )
                if not g2_keys:
                    raise KeyError(f"No 'Bottom G2 Gate' datasets in {fname}")
                tpsfs = np.zeros(
                    (*gate_grp[g2_keys[0]].shape, len(g2_keys)), dtype=np.float32
                )
                for i, key in enumerate(g2_keys):
                    tpsfs[:, :, i] = gate_grp[key][:]
            return tpsfs

        if os.path.isfile(bg_path):
            if not bg_path.lower().endswith(('.h5', '.hdf5')):
                raise ValueError(f"Expected an HDF5 file (.h5 / .hdf5), got: {bg_path}")
            accumulated = _read_raw(bg_path)[np.newaxis]   # (1, H, W, T)

        elif os.path.isdir(bg_path):
            files = sorted(
                f for f in os.listdir(bg_path)
                if f.lower().endswith(('.h5', '.hdf5'))
            )
            if not files:
                raise FileNotFoundError(f"No HDF5 files found in: {bg_path}")
            accumulated = np.stack(
                [_read_raw(os.path.join(bg_path, f)) for f in files], axis=0
            )                                               # (n, H, W, T)

        else:
            raise FileNotFoundError(f"Path not found: {bg_path}")

        # All three outputs have shape (H, W, T), operated over the file axis
        bg_sum    = np.sum(accumulated,    axis=0)   # (H, W, T)
        bg_mean   = np.mean(accumulated,   axis=0)   # (H, W, T)
        bg_median = np.median(accumulated, axis=0)   # (H, W, T)

        # Hot pixel detection: collapse time → (H, W), then threshold via MAD
        total_counts = np.sum(bg_sum, axis=-1)       # (H, W)
        median = np.median(total_counts)
        mad    = np.median(np.abs(total_counts - median))
        thresh = median + threshold_sigma * 1.4826 * mad

        hot_pixel_map = total_counts > thresh

        n_hot = int(np.sum(hot_pixel_map))
        print(f"Detected {n_hot} hot pixels "
              f"({100.0 * n_hot / total_counts.size:.3f}% of {total_counts.shape})")
        print(f"Threshold: {thresh:.1f}  "
              f"(median={median:.1f}, MAD={mad:.1f}, σ={threshold_sigma})")

        if save_path:
            import matplotlib.pyplot as plt
            plt.imsave(save_path, hot_pixel_map.astype(np.uint8) * 255, cmap='gray')
            print(f"Hot pixel mask saved to: {save_path}")

        return hot_pixel_map, bg_sum, bg_mean, bg_median
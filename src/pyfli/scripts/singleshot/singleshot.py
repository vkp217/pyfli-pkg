# scripts/singleshot/singleshot.py
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

class SingleShotFLI:
    def __init__(self):
        pass

    def compute_lifetime(step, fname, gate_width, gate):
        """
        Single-shot fluorescence lifetime estimation using lookup table and HDF5 TPSF data.
        Parameters
        ----------
        step : float
            Step size for lookup table generation (e.g., 0.1).
        fname : str
            Path to HDF5 file containing TPSF data.
        gate_width : int
            Gate width in nanoseconds.
        gate : int
            Gate index to use for decay calculation.
        
        Returns
        -------
        output : np.ndarray
            Computed fluorescence lifetime image.
        """
        # --- Lookup Table Generation ---
        t0 = 0
        w = np.arange(1, 10 + step, step)
        T = 12.5
        tau = np.arange(200, 3001) * 1e-3  # ns

        ratio = np.zeros((len(w), len(tau)))
        for i, wi in enumerate(w):
            cal1 = np.exp(-t0 / tau)
            cal2 = np.exp(-(t0 + wi) / tau)
            cal3 = 1 - np.exp(-T / tau)
            cal4 = cal1 - cal2
            ratio[i, :] = cal4 / cal3

        with h5py.File(fname, 'r') as f:

            tpsfs1 = np.array(f['/tpsfs1'])
            inten1 = np.array(f['/inten1'])
            tpsfs2 = np.array(f['/tpsfs2'])
            inten2 = np.array(f['/inten2'])

        # --- Single-shot Lifetime Estimation ---
        ratio_vec1_temp = np.squeeze(tpsfs1[:, :, gate])
        ratio_vec1 = ratio_vec1_temp.flatten()

        ratio_vec2_temp = np.squeeze(tpsfs2[:, :, gate])
        ratio_vec2 = ratio_vec2_temp.flatten()

        rratio = ratio_vec1_temp / ratio_vec2_temp
        rratio[rratio > 2] = 2

        rratio_vec = np.full_like(ratio_vec1, np.nan, dtype=float)
        for i in range(len(ratio_vec1)):
            if ratio_vec1[i] > 3 and ratio_vec2[i] > 5:
                rratio_vec[i] = ratio_vec1[i] / ratio_vec2[i]

        # --- Interpolation ---
        a1, a2 = rratio.shape
        k = int((gate_width - 1) / step) + 1
        interp_func = interp1d(ratio[k, :], tau, kind='nearest', bounds_error=False, fill_value=np.nan)
        vq_vec = interp_func(rratio_vec)
        vq = vq_vec.reshape(a1, a2) - (500 * 1e-3)

        output = np.nan_to_num(vq, nan=0.0)

        # --- Plot Results ---
        fig, axs = plt.subplots(1, 3, figsize=(14, 5))
        im1 = axs[0].imshow(ratio_vec1_temp)
        axs[0].set_title(f"G2 {gate}")
        plt.colorbar(im1, ax=axs[0])

        im2 = axs[1].imshow(ratio_vec2_temp)
        axs[1].set_title(f"INT {gate}")
        plt.colorbar(im2, ax=axs[1])

        im3 = axs[2].imshow(output)
        axs[2].set_title("Computed Lifetime")
        plt.colorbar(im3, ax=axs[2])

        plt.tight_layout()
        plt.show()

        return output
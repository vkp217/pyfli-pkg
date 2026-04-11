# scripts/ss_helpers.py
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
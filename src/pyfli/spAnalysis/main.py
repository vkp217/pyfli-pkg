import numpy as np
from .solvers import NyquistHadamard, TVMinimization

def run_reconstruction(data, mode='nyquist'):
    res = 64
    bins = data.shape[1]
    
    if mode == 'nyquist':
        engine = NyquistHadamard(res, bins)
    elif mode == 'tv':
        engine = TVMinimization(res, bins, reg_param=0.01)
    else:
        raise ValueError("Unknown mode")

    cube = engine.reconstruct(data)
    return engine.post_process(cube)

# Example usage:
# raw_data = np.load("tcspc_counts.npy")
# final_cube = run_reconstruction(raw_data, mode='nyquist')
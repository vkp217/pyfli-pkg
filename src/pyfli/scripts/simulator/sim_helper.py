import numpy as np

def irf_picker(irf_full):
    # IRF Selection Logic
    if irf_full.ndim == 3:
        max_attempts = 1000
        attempts = 0
        while True:
            x = np.random.randint(irf_full.shape[0])
            y = np.random.randint(irf_full.shape[1])
            pixel_data = irf_full[x, y, :]
            
            max_val = np.max(pixel_data)
            min_val = np.min(pixel_data)
            ratio = max_val / min_val if min_val > 0 else 0
            
            if max_val > 700 or ratio > 20: # Ensure a clear peak
                irf = pixel_data
                break
            
            attempts += 1
            if attempts >= max_attempts:
                raise RuntimeError(f"Could not find a valid pixel after {max_attempts} attempts.")          
    elif irf_full.ndim == 1:
        irf = irf_full
    else:
        raise ValueError(f'IRF must be 1-D or 3-D, got shape {irf_full.shape}')
    
    return irf
    

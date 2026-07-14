import numpy as np

def irf_picker(irf_full):
    # IRF Selection Logic
    if irf_full.ndim == 3:
        H, W, T = irf_full.shape
        max_attempts = 1000

        for _ in range(max_attempts):
            x = np.random.randint(H)
            y = np.random.randint(W)
            pixel_data = irf_full[x, y, :]

            peak = np.max(pixel_data)
            if peak <= 500:
                continue

            # Estimate noise from the pre-peak baseline (first quarter before the peak)
            # SNR criterion: noise <= 5% of peak  ↔  SNR = peak/noise >= 20
            peak_idx = int(np.argmax(pixel_data))
            if peak_idx >= 3:
                n_baseline = max(3, peak_idx // 4)
                noise = np.std(pixel_data[:n_baseline])
            else:
                # Peak is at the very start — fall back to the far tail
                noise = np.std(pixel_data[int(0.75 * T):])

            if noise <= 0.05 * peak:  # SNR >= 20
                irf = pixel_data
                break
        else:
            raise RuntimeError(f"Could not find a valid IRF pixel after {max_attempts} attempts.")          
    elif irf_full.ndim == 1:
        irf = irf_full
    else:
        raise ValueError(f'IRF must be 1-D or 3-D, got shape {irf_full.shape}')
    
    return irf
    

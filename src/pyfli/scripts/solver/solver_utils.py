#  solver/solver_utils.py
import numpy as np

def estimate_p0_numeric(t, decay_pixel, model_type, period):
    eps = 1e-12
    d = decay_pixel.copy()
    B_est = max(float(np.percentile(d, 10)), 0.0)
    d_corr = np.clip(d - B_est, 0, None)
    total = np.sum(d_corr) + eps
    
    if model_type == "mono-exponential":
        A_est = max(float(np.max(d_corr)), eps)
        tau_est = float(np.sum(t * d_corr) / total)
        tau_est = float(np.clip(tau_est, 0.05 * period, 0.9 * period))
        return [A_est, tau_est, B_est]
    
    # Bi-exponential logic here...
    # (Truncated for brevity, insert your log-slope logic from the original code)
    return [A1_est, tau1_est, A2_est, tau2_est, B_est]

def compute_stats(y, y_fit, n_free, weighted=False, sigma=None):
    residual = y - y_fit
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-12)
    rmse = np.sqrt(ss_res / len(y))
    
    if weighted:
        _s = sigma if sigma is not None else np.sqrt(np.maximum(y, 1.0))
        _s = np.where(_s < 1e-12, 1e-12, _s)
        chi2 = np.sum((residual / _s) ** 2) / max(len(y) - n_free, 1)
    else:
        chi2 = ss_res / max(len(y) - n_free, 1)
    return residual, r2, rmse, chi2
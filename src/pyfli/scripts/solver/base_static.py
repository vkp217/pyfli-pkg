# solver/base_static.py
import numpy as np

def moment_based_guess(t, decay, T_acq, T_laser, model_type='mono-exponential'):
    """
    Robustly estimates parameters using the 0th and 1st moments (Area and Mean Time).
    Returns a dictionary of parameters.
    """
    # 1. Background (offset): 5th percentile is safer than min() for noise
    offset_guess = np.percentile(decay, 5)
    clean_d = np.clip(decay - offset_guess, 1e-6, None)
    
    # 2. Find peak to define the start of the 'actual' decay
    idx_max = np.argmax(clean_d)
    t_decay = t[idx_max:] - t[idx_max]
    d_decay = clean_d[idx_max:]
    
    # 3. Moment Analysis
    m0 = np.trapezoid(d_decay, t_decay) # Total Area
    if m0 > 0:
        # Mean lifetime <t> = Integral(t * I(t)) / Integral(I(t))
        m1 = np.trapezoid(t_decay * d_decay, t_decay)
        tau_mean = m1 / m0
    else:
        tau_mean = T_laser / 10.0

    # Safety clipping to ensure we stay inside physical bounds of the system
    tau_g = np.clip(tau_mean, 0.05, T_laser * 0.8)
    
    # 4. Intensity S (corrected for window truncation)
    # Area = S * tau * (1 - exp(-T_acq / tau))
    s_guess = m0 / (tau_g * (1 - np.exp(-T_acq / tau_g))) if tau_g > 0 else m0

    if model_type == 'mono-exponential':
        return {
            'amp': float(s_guess),
            'tau': float(tau_g),
            'offset': float(offset_guess)
        }
    else:
        # For bi-exponential, we seed on both sides of the mean
        return {
            'amp': float(s_guess),
            'alpha1': 0.5,
            'tau1': float(tau_g * 0.5),
            'tau2': float(tau_g * 1.5),
            'offset': float(offset_guess)
        }

def rld_based_guess(t, decay, T_acq, T_laser, model_type='mono-exponential'):
    """
    Rapid Lifetime Determination (RLD) using integrated time windows.
    Returns a dictionary of parameters.
    """
    offset_guess = np.percentile(decay, 5)
    clean_d = np.clip(decay - offset_guess, 1e-6, None)
    
    idx_max = np.argmax(clean_d)
    t_fit = t[idx_max:] - t[idx_max]
    y_fit = clean_d[idx_max:]
    
    num_bins = len(y_fit)
    dt = t_fit[1] - t_fit[0] if num_bins > 1 else 1.0

    if model_type == 'mono-exponential':
        mid = num_bins // 2
        a0 = np.sum(y_fit[:mid])
        a1 = np.sum(y_fit[mid:2*mid])
        
        tau_g = (dt * mid) / np.log(a0 / a1) if (a1 > 0 and a0 > a1) else T_laser/10.0
        tau_g = np.clip(tau_g, 0.05, T_laser * 0.8)
        
        return {
            'amp': float(np.max(y_fit)),
            'tau': float(tau_g),
            'offset': float(offset_guess)
        }
    else:
        q = num_bins // 4
        # Early and late window sets
        a0, a1 = np.sum(y_fit[:q]), np.sum(y_fit[q:2*q])
        a2, a3 = np.sum(y_fit[2*q:3*q]), np.sum(y_fit[3*q:4*q])
        
        t1 = (dt * q) / np.log(a0 / a1) if (a1 > 0 and a0 > a1) else 0.5
        t2 = (dt * q) / np.log(a2 / a3) if (a3 > 0 and a2 > a3) else 2.0
        
        tau1_g = np.clip(min(t1, t2), 0.05, T_laser * 0.4)
        tau2_g = np.clip(max(t1, t2), tau1_g * 1.1, T_laser * 0.8)
        
        return {
            'amp': float(np.max(y_fit)),
            'alpha1': 0.5,
            'tau1': float(tau1_g),
            'tau2': float(tau2_g),
            'offset': float(offset_guess)
        }
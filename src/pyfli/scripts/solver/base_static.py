"""Static helper functions for parameter/bounds resolution and initial-guess estimation.

Provides plugin-style initial-guess generators (moment-based and rapid
lifetime determination-based) and logic to merge user-supplied parameter
guesses/bounds with automatically generated defaults for the FLI fitters.
"""
import numpy as np

def resolve_params_and_bounds(user_p0, user_bounds, model_type, t, decay, T_laser, guess_plugin, T_acq):
    """Merge user-supplied initial parameters and bounds with an automatic guess.

    Calls ``guess_plugin`` to obtain a dictionary of default initial parameter
    values, overrides it with any user-supplied ``user_p0`` (dict or
    array-like), builds the initial parameter vector and default bounds for
    the requested model, then applies any user-supplied ``user_bounds``
    overrides. The resulting initial guess is clipped to lie strictly within
    the final bounds.

    Args:
        user_p0: User-supplied initial guess, either a dict keyed by
            parameter name (``'amp'``, ``'tau'``/``'alpha1'``/``'tau1'``/
            ``'tau2'``, ``'v_shift'``, ``'h_shift'``) or a positional
            list/array. May be None.
        user_bounds: User-supplied bounds, either a dict keyed by parameter
            name mapping to ``(low, high)`` tuples, or a positional
            list/array of ``(low, high)`` pairs (or None entries to keep the
            default). May be None.
        model_type: Either ``'mono-exponential'`` or ``'bi-exponential'``.
        t: Time axis (ns) used by the guess plugin.
        decay: Measured decay counts used by the guess plugin.
        T_laser: Laser period (ns), used to set upper bounds on lifetimes and
            default guess ranges.
        guess_plugin: Callable ``(t, decay, T_acq, T_laser, model_type) ->
            dict`` that produces the default parameter guess (e.g.
            ``moment_based_guess`` or ``rld_based_guess``).
        T_acq: Acquisition window period (ns), used to bound the temporal
            shift parameter and passed through to ``guess_plugin``.

    Returns:
        tuple: ``(p0_safe, (low_vec, high_vec))`` where ``p0_safe`` is the
        clipped initial parameter vector and ``(low_vec, high_vec)`` are the
        resolved lower/upper bound arrays.
    """
    smart_dict = guess_plugin(t, decay, T_acq, T_laser, model_type)

    smart_dict.setdefault('h_shift', 0.0)

    if isinstance(user_p0, dict):
        smart_dict.update(user_p0)
    elif isinstance(user_p0, (list, np.ndarray)):
        keys = ['amp', 'tau', 'v_shift', 'h_shift'] if model_type == 'mono-exponential' else \
                ['amp', 'alpha1', 'tau1', 'tau2', 'v_shift', 'h_shift']
        for i, val in enumerate(user_p0):
            if i < len(keys):
                smart_dict[keys[i]] = val

    if model_type == 'mono-exponential':
        p0_vec = np.array([
            smart_dict['amp'],
            smart_dict['tau'],
            smart_dict['v_shift'],
            smart_dict.get('h_shift', 0.0),
        ])
    else:
        p0_vec = np.array([
            smart_dict['amp'],
            smart_dict['alpha1'],
            smart_dict['tau1'],
            smart_dict['tau2'],
            smart_dict['v_shift'],
            smart_dict.get('h_shift', 0.0),
        ])

    n_params = len(p0_vec)

    # h_shift is now in ns (same units as t), so bound by T_acq/4
    shift_bound = T_acq / 4.0

    low_vec  = np.zeros(n_params)
    high_vec = np.full(n_params, np.inf)

    if model_type == 'bi-exponential':
        low_vec[1],  high_vec[1]  = 0.0,  1.0
        low_vec[2],  high_vec[2]  = 1e-4, T_laser
        low_vec[3],  high_vec[3]  = max(float(p0_vec[2]), 1e-4), T_laser
        low_vec[5],  high_vec[5]  = -shift_bound, shift_bound
    else:
        low_vec[1],  high_vec[1]  = 1e-4, T_laser
        low_vec[3],  high_vec[3]  = -shift_bound, shift_bound

    if isinstance(user_bounds, dict):
        key_map = {
            'amp': 0, 'tau': 1, 'v_shift': 2, 'h_shift': 3,
        } if model_type == 'mono-exponential' else {
            'amp': 0, 'alpha1': 1, 'tau1': 2, 'tau2': 3, 'v_shift': 4, 'h_shift': 5,
        }
        for k, v in user_bounds.items():
            if k in key_map:
                low_vec[key_map[k]], high_vec[key_map[k]] = v
    elif isinstance(user_bounds, (list, np.ndarray)):
        for i, b in enumerate(user_bounds):
            if b is not None and i < n_params:
                low_vec[i], high_vec[i] = b

    high_vec = np.maximum(high_vec, low_vec + 1e-6)

    p0_safe = np.clip(p0_vec, low_vec + 1e-7, high_vec - 1e-7)

    return p0_safe, (low_vec, high_vec)

def moment_based_guess(t, decay, T_acq, T_laser, model_type='mono-exponential'):
    """Estimate initial decay parameters from the first moment (mean arrival time) of the decay.

    Estimates a baseline offset from the 5th percentile of the decay, then
    computes the area (zeroth moment) and mean arrival time (first moment /
    zeroth moment) of the background-subtracted, peak-aligned decay tail to
    derive a lifetime guess. For the bi-exponential model, the early/late
    half-split area ratio is used to seed ``alpha1``.

    Args:
        t: Time axis (ns).
        decay: Measured decay counts.
        T_acq: Acquisition window period (ns).
        T_laser: Laser period (ns), used to bound the lifetime guess.
        model_type: Either ``'mono-exponential'`` (default) or
            ``'bi-exponential'``.

    Returns:
        dict: Initial guess dictionary. For ``'mono-exponential'``: ``amp``,
        ``tau``, ``v_shift``. For ``'bi-exponential'``: ``amp``, ``alpha1``,
        ``tau1``, ``tau2``, ``v_shift``.
    """
    offset_guess = np.percentile(decay, 5)
    clean_d = np.clip(decay - offset_guess, 1e-6, None)

    idx_max = np.argmax(clean_d)
    t_decay = t[idx_max:] - t[idx_max]
    d_decay = clean_d[idx_max:]

    m0 = np.trapezoid(d_decay, t_decay)
    if m0 > 0:
        m1 = np.trapezoid(t_decay * d_decay, t_decay)
        tau_mean = m1 / m0
    else:
        tau_mean = T_laser / 10.0

    tau_g = np.clip(tau_mean, 0.05, T_laser * 0.8)

    s_guess = m0 / (1 - np.exp(-T_acq / tau_g)) if tau_g > 0 else m0

    if model_type == 'mono-exponential':
        return {
            'amp': float(s_guess),
            'tau': float(tau_g),
            'v_shift': float(offset_guess)
        }
    else:
        if len(d_decay) > 2:
            half = max(len(d_decay) // 2, 1)
            a_early = float(np.trapezoid(d_decay[:half], t_decay[:half])) + 1e-9
            a_late  = float(np.trapezoid(d_decay[half:], t_decay[half:])) + 1e-9
            alpha1_guess = float(np.clip(a_early / (a_early + a_late), 0.001, 0.999))
        else:
            alpha1_guess = 0.5
        return {
            'amp': float(s_guess),
            'alpha1': alpha1_guess,
            'tau1': float(tau_g * 0.5),
            'tau2': float(tau_g * 1.5),
            'v_shift': float(offset_guess)
        }

def rld_based_guess(t, decay, T_acq, T_laser, model_type='mono-exponential'):
    """Estimate initial decay parameters using Rapid Lifetime Determination (RLD).

    Splits the peak-aligned, background-subtracted decay tail into equal-width
    bins (2 bins for mono-exponential, 4 bins for bi-exponential) and estimates
    lifetimes from the log-ratio of integrated counts in consecutive bins.

    Args:
        t: Time axis (ns).
        decay: Measured decay counts.
        T_acq: Acquisition window period (ns).
        T_laser: Laser period (ns), used to bound the lifetime guesses.
        model_type: Either ``'mono-exponential'`` (default) or
            ``'bi-exponential'``.

    Returns:
        dict: Initial guess dictionary. For ``'mono-exponential'``: ``amp``,
        ``tau``, ``v_shift``. For ``'bi-exponential'``: ``amp``, ``alpha1``
        (fixed at 0.5), ``tau1``, ``tau2``, ``v_shift``.
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
            'v_shift': float(offset_guess)
        }
    else:
        q = num_bins // 4
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
            'v_shift': float(offset_guess)
        }

"""Console/log formatting helpers for FLI fit results.

Provides `Msg_display`, a small presenter class that formats fitted
decay parameters, session configuration, and per-pixel data-map summaries
into readable text blocks, either printing them or forwarding them to a
`DataSaver`-like logger.
"""
import os
import numpy as np

class Msg_display:
    """Formats and displays/logs FLI fitting results and diagnostics.

    If constructed with a `saver` object (exposing `log` and, optionally,
    `save_params`), formatted messages are also written to that saver's
    log in addition to (or instead of) being printed to stdout.

    Attributes:
        saver: Optional logger object used to persist displayed messages.
    """

    def __init__(self, saver=None):
        """Initialize the display helper.

        Args:
            saver: Optional object with a `log(message)` method (and
                optionally `save_params(**kwargs)`) used to persist
                displayed output. If None, output is only printed.
        """
        self.saver = saver

    def _internal_log(self, message):
        if self.saver:
            self.saver.log(message)
        else:
            print(message)

    def disp_params(self, res_px, model_type='bi-exponential'):
        """Format and display fitted parameters for a single pixel.

        Builds a bordered text block listing the fitted parameter values
        with their uncertainties, R^2, chi-squared, reduced chi-squared,
        and convergence status, then prints/logs it via `_internal_log`.

        Args:
            res_px: Sequence of per-pixel fit results, expected to
                contain at least 7 elements where index 0 is the
                parameter array, index 1 the parameter errors, indices
                2-4 are R2/chi2/reduced-chi2, and index 6 is the
                convergence flag.
            model_type: Decay model used for the fit; determines the
                parameter labels shown. Use 'bi-exponential' (default)
                for the 5-parameter model, otherwise the 3-parameter
                mono-exponential labels are used.

        Raises:
            ValueError: If `res_px` is empty or None.
            IndexError: If `res_px` does not contain the expected number
                of elements.
        """
        if not res_px:
            raise ValueError('Data was not provided (res_px is empty or None)')

        try:
            p, err = res_px[0], res_px[1]
            r2, chi2, red_chi2 = res_px[2], res_px[3], res_px[4]
            conv = res_px[6]
        except IndexError:
            raise IndexError("res_px does not have the expected number of elements.")

        # Build output string
        output = []
        output.append('\n' + '='*30)
        output.append(f'FIT PARAMETERS ({model_type.upper()})')
        output.append('-'*30)

        labels = ['photon_counts', 'alpha1', 'tau1', 'tau2', 'v-shift'] if model_type == 'bi-exponential' else ['photon_counts', 'tau', 'v-shift']
        
        for i, label in enumerate(labels):
            output.append(f'{label:8}: {p[i]:.4f} \u00B1 {err[i]:.4f}')

        output.append('-'*30)
        output.append(f'R2           : {r2:.4f}')
        output.append(f'chi2         : {chi2:.4f}')
        output.append(f'Reduced chi2 : {red_chi2:.4f}')
        output.append(f'Convergence  : {conv}')
        output.append('='*30 + '\n')

        # Display and Log
        full_msg = "\n".join(output)
        self._internal_log(full_msg)

    def fit_session(self, **kwargs):
        """Display/log a formatted session-configuration banner.

        Prints (or logs) a bordered header, the given keyword arguments
        as labeled settings (using friendlier labels for known keys),
        and a footer. If a `saver` is set, the raw kwargs are also
        recorded via `saver.save_params`.

        Args:
            **kwargs: Session configuration values to display, such as
                `model_type`, `processor_name`, `fitter_name`, `p0`,
                `use_initial_guess`, and `use_bounds`.
        """
        pretty_labels = {
            'model_type': 'Decay Model',
            'processor_name': 'Processor',
            'fitter_name': 'Fitting Method',
            'p0': 'Initial Guesses (p0)',
            'use_initial_guess': 'Using Guess',
            'use_bounds': 'Using Bounds'
        }

        header = '\n' + '-' * 60 + f"\n{'SESSION CONFIGURATION':^60}\n" + '-' * 60
        self._internal_log(header)

        # Log parameters via save_params if saver exists for structured logging
        if self.saver:
            self.saver.save_params(**kwargs)

        for key, value in kwargs.items():
            label = pretty_labels.get(key, key.replace('_', ' ').capitalize())
            self._internal_log(f"{label:25}: {value}")

        footer = '-' * 60 + f"\n{'Session Initialized':^60}\n" + '-' * 60 + '\n'
        self._internal_log(footer)

    # Fixed display order: label → candidate map keys (first match wins)
    _PIXEL_FIELDS = [
        ('A',        ['photon_count_map']),
        ('α',        ['alpha1_map', 'alpha_map']),
        ('τ₁',       ['tau1_map',   'tau_map']),
        ('τ₂',       ['tau2_map']),
        ('R²',       ['R2_map']),
        ('Red.χ²',   ['reduced_chi2_map']),
        ('Raw.χ²',   ['chi2_map']),
        ('v-shift',  ['v_shift_map']),
        ('h-shift',  ['h_shift_map']),
    ]

    def get_pixel_summary(self, data_maps, px):
        """Display/log a formatted summary of fit-result maps for one pixel.

        Looks up the value at `px` in each of the known 2-D result maps
        (photon count, alpha, tau1/tau2, R2, chi-squared variants, and
        shifts), formats them into an aligned text block, prints it,
        logs it via the saver if present, and returns the raw rows.

        Args:
            data_maps: Dictionary mapping map names (e.g.
                'photon_count_map', 'alpha1_map', 'tau1_map', 'R2_map')
                to 2-D NumPy arrays.
            px: `(x, y)` pixel coordinate to index into each map.

        Returns:
            list[tuple[str, str]]: `(label, formatted_value)` pairs in
            display order, where `formatted_value` is `'—'` if no
            matching map was found, `'error'` if indexing failed, or the
            value formatted to 4 decimal places otherwise.
        """
        x, y = px
        rows = []
        for label, candidates in self._PIXEL_FIELDS:
            val = '—'
            for key in candidates:
                m = data_maps.get(key)
                if isinstance(m, np.ndarray) and m.ndim == 2:
                    try:
                        v = m[x, y]
                        val = f'{float(v):.4f}'
                    except Exception:
                        val = 'error'
                    break
            rows.append((label, val))

        label_w = max(len(lbl) for lbl, _ in rows)
        rule = '─' * (label_w + 14)
        lines = [f'\n  Pixel {px}', f'  {rule}']
        for label, val in rows:
            lines.append(f'  {label:<{label_w}}   {val}')
        lines.append(f'  {rule}\n')

        output = '\n'.join(lines)
        print(output)

        if self.saver:
            self.saver.log(output)

        return rows
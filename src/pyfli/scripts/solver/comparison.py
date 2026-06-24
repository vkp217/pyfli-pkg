# solver/comparison.py
import numpy as np
import time
import matplotlib.pyplot as plt

class FittingComparator:
    def __init__(self, freq, base_fitter_class, mle_fitter_class):
        self.freq = freq
        self.BaseClass = base_fitter_class
        self.MLEClass = mle_fitter_class
        
        self.method_mapping = {
            'least_squares': ('NLSF', self.BaseClass),
            'trust_region':  ('NLSF', self.BaseClass),
            'unconstrained': ('NLSF', self.BaseClass),
            'poisson':       ('MLE',  self.MLEClass),
            'pearson':       ('MLE',  self.MLEClass),
            'neyman':        ('MLE',  self.MLEClass)
        }

    @staticmethod
    def _print_summary_table(rows, model_type):
        """rows: list of [method, category, success, elapsed, r2, stat, red_stat, popt | None]"""
        is_bi = model_type == 'bi-exponential'
        if is_bi:
            cols = [
                ('Method',  14, '<'), ('Type',   4, '^'),
                ('A',        7, '>'), ('α',      7, '>'),
                ('τ₁',       8, '>'), ('τ₂',     8, '>'),
                ('R²',       7, '>'), ('Red.χ²', 8, '>'), ('Raw.χ²', 8, '>'),
                ('v-shift',  7, '>'), ('h-shift', 8, '>'),
            ]
        else:
            cols = [
                ('Method',  14, '<'), ('Type',   4, '^'),
                ('A',        7, '>'), ('τ',      8, '>'),
                ('R²',       7, '>'), ('Red.χ²', 8, '>'), ('Raw.χ²', 8, '>'),
                ('v-shift',  7, '>'), ('h-shift', 8, '>'),
            ]

        def fmt_cell(text, width, align):
            s = str(text)
            if align == '<':  return f' {s:<{width}} '
            if align == '>':  return f' {s:>{width}} '
            return f' {s:^{width}} '

        def row_str(cells):
            return '│' + '│'.join(fmt_cell(c, w, a) for c, (_, w, a) in zip(cells, cols)) + '│'

        def sep(left, mid, right):
            return left + mid.join('─' * (w + 2) for _, w, _ in cols) + right

        print()
        print(sep('┌', '┬', '┐'))
        print(row_str([c[0] for c in cols]))
        print(sep('├', '┼', '┤'))

        for method, category, success, elapsed, r2, stat, red_stat, popt in rows:
            if popt is None:
                filler = ['—'] * (len(cols) - 2)
                cells = [method, category] + filler
            elif is_bi and len(popt) >= 6:
                cells = [
                    method, category,
                    f'{popt[0]:.2f}', f'{popt[1]:.4f}',
                    f'{popt[2]:.3f}', f'{popt[3]:.3f}',
                    f'{r2:.4f}', f'{red_stat:.4f}', f'{stat:.2f}',
                    f'{popt[4]:.2f}', f'{popt[5]:.3f}',
                ]
            elif is_bi and len(popt) >= 5:
                cells = [
                    method, category,
                    f'{popt[0]:.2f}', f'{popt[1]:.4f}',
                    f'{popt[2]:.3f}', f'{popt[3]:.3f}',
                    f'{r2:.4f}', f'{red_stat:.4f}', f'{stat:.2f}',
                    f'{popt[4]:.2f}', '—',
                ]
            else:
                cells = [
                    method, category,
                    f'{popt[0]:.2f}', f'{popt[1]:.3f}',
                    f'{r2:.4f}', f'{red_stat:.4f}', f'{stat:.2f}',
                    f'{popt[2]:.2f}', f'{popt[3]:.3f}' if len(popt) > 3 else '—',
                ]
            print(row_str(cells))

        print(sep('└', '┴', '┘'))
        print()

    @staticmethod
    def _weighted_residual(method, y, model):
        """Return normalised residuals appropriate for each estimator.

        Poisson MLE  → signed deviance residual  sign(y−m)·√(2(m−y+y·ln(y/m)))
        Pearson χ²   → (y − m) / √m
        Neyman  χ²   → (y − m) / √max(y, 1)
        NLSF         → (y − m) / √max(m, 1)   (approx Poisson weight)
        """
        m = np.clip(model, 1e-9, None)
        if method == 'poisson':
            safe_y = np.where(y > 0, y, 1e-9)
            dev = 2.0 * (m - y + y * np.log(safe_y / m))
            return np.sign(y - m) * np.sqrt(np.maximum(dev, 0.0))
        elif method == 'pearson':
            return (y - m) / np.sqrt(m)
        elif method == 'neyman':
            return (y - m) / np.sqrt(np.maximum(y, 1.0))
        else:  # least_squares, trust_region, unconstrained
            return (y - m) / np.sqrt(np.maximum(m, 1.0))

    def compare_selected(self, methods, y_data, irf_data, model_type='bi-exponential',
                         p0=None, bounds=None, yscale='log', plot=True):
        results_table = []
        if y_data.ndim != 1 or irf_data.ndim != 1:
            raise ValueError("compare_selected expects 1D decay and IRF traces")
        y_in   = y_data.astype(np.float32)
        irf_in = irf_data.astype(np.float32)

        plot_data = {'y': y_in, 'irf': irf_in, 'fits': {}, 'residuals': {}, 't': None}

        W = 62
        n_methods = len([m for m in methods if m in self.method_mapping])
        title    = f"FLI Fitting Results  |  {model_type.upper()}"
        subtitle = f"{n_methods} method{'s' if n_methods != 1 else ''} queued"
        print(f'\n┌{"─"*W}┐')
        print(f'│  {title:<{W-2}}│')
        print(f'│  {subtitle:<{W-2}}│')
        print(f'└{"─"*W}┘\n')

        for method in methods:
            if method not in self.method_mapping: continue
            category, Fitter = self.method_mapping[method]

            fitter_inst = Fitter(self.freq, y_in, irf_in)
            if plot_data['t'] is None:
                plot_data['t'] = fitter_inst.t   # physical time axis (ns)
            start_time = time.perf_counter()

            try:
                res = fitter_inst.fit_with_estimator(
                    estimator_type=method, model_type=model_type, p0=p0, bounds=bounds
                )
                elapsed = (time.perf_counter() - start_time) * 1000

                popt     = res[0]
                fit_full = fitter_inst.model_fit(fitter_inst.t, popt, model_type=model_type).astype(np.float32)

                # Normalised residuals only over the fitted region
                idx   = fitter_inst.fit_indices
                resid = np.full_like(y_in, np.nan)
                resid[idx] = self._weighted_residual(method, y_in[idx], fit_full[idx])

                r2       = res[2]
                stat     = res[3]
                red_stat = res[4]
                success  = "YES" if res[6] == 1 else "NO"

                if plot:
                    plot_data['fits'][method]      = fit_full
                    plot_data['residuals'][method] = resid

                results_table.append([
                    method.upper(), category, success, f"{elapsed:.2f} ms",
                    r2, stat, red_stat, popt
                ])

            except Exception as e:
                results_table.append([
                    method.upper(), category, "FAIL", "N/A", 0.0, 0.0, 0.0, None
                ])

        self._print_summary_table(results_table, model_type)

        if plot and plot_data['fits']:
            self._plot_comparison(plot_data, yscale, model_type)

        return results_table

    def run_all(self, y_data, irf_data, model_type='bi-exponential', p0=None, bounds=None, yscale='log', plot=True):
        return self.compare_selected(list(self.method_mapping.keys()), y_data, irf_data, 
                                     model_type, p0, bounds, yscale=yscale, plot=plot)

    def _plot_comparison(self, data, yscale, model_type):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                                       gridspec_kw={'height_ratios': [2.5, 1]})

        t = data['t'] if data.get('t') is not None else np.arange(len(data['y']))
        x_label = 'Time (ns)' if data.get('t') is not None else 'Sample Index'

        ax1.step(t, data['y'], where='mid', color='gray', alpha=0.3, label='Raw Data')

        colors = plt.cm.tab10(np.linspace(0, 1, len(data['fits'])))
        for i, (name, fit) in enumerate(data['fits'].items()):
            c = colors[i]
            ax1.plot(t, fit, label=f'Fit: {name.upper()}', color=c, linewidth=1.5)
            resid = data['residuals'][name]
            valid  = ~np.isnan(resid)
            ax2.plot(t[valid], resid[valid], color=c, alpha=0.7,
                     label=name.upper())

        ax1.set_yscale(yscale)
        ax1.set_ylabel('Photon Counts')
        ax1.set_title(f'FLI Diagnostic Comparison ({model_type.upper()})')
        ax1.legend(loc='upper right', fontsize='x-small', ncol=2)
        ax1.grid(True, which='both', ls='-', alpha=0.05)

        ax2.axhline(0, color='black', linewidth=1.2, alpha=0.8)
        ax2.set_ylabel('Normalised Residuals')
        ax2.set_xlabel(x_label)
        ax2.legend(loc='upper right', fontsize='x-small', ncol=2)
        ax2.grid(True, alpha=0.05)

        plt.tight_layout()
        plt.show()
        return fig
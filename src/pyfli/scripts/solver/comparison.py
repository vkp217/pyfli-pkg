# solver/comparison.py
import numpy as np
import time
import matplotlib.pyplot as plt
from tabulate import tabulate

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

        print(f"\n{'='*150}")
        print(f"FLI DIAGNOSTIC BENCHMARK | Model: {model_type.upper()}")
        print(f"{'='*150}\n")

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

                if model_type == 'bi-exponential':
                    h_s = f", Δh:{popt[5]:.2f}" if len(popt) > 5 else ""
                    p_str = f"A:{popt[0]:.1f}, α:{popt[1]:.2f}, τ1:{popt[2]:.2f}, τ2:{popt[3]:.2f}, B:{popt[4]:.1f}{h_s}"
                else:
                    h_s = f", Δh:{popt[3]:.2f}" if len(popt) > 3 else ""
                    p_str = f"A:{popt[0]:.1f}, τ:{popt[1]:.2f}, B:{popt[2]:.1f}{h_s}"

                results_table.append([
                    method.upper(), category, success, f"{elapsed:.2f} ms", 
                    f"{r2:.4f}", f"{stat:.2f}", f"{red_stat:.3f}", p_str
                ])

            except Exception as e:
                results_table.append([method.upper(), category, "FAIL", "N/A", "N/A", "N/A", "N/A", f"Err: {str(e)[:30]}"])

        # Display Summary Table
        headers = ["Method", "Type", "Conv", "Time", "R2", "Chi2", "Red. Chi2", "Parameters"]
        print(tabulate(results_table, headers=headers, tablefmt="fancy_grid"))

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
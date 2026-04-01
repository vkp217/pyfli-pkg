# solver/comparison.py
import numpy as np
import time
from tabulate import tabulate

class FittingComparator:
    def __init__(self, freq, base_fitter_class, mle_fitter_class):
        """
        freq: [freq_laser, freq_acquisition]
        base_fitter_class: Reference to BaseFLIFitter (NLSF)
        mle_fitter_class: Reference to MLEFLIFitter (MLE)
        """
        self.freq = freq
        self.BaseClass = base_fitter_class
        self.MLEClass = mle_fitter_class
        
        # Comprehensive mapping of all available objective functions
        self.method_mapping = {
            'least_squares': ('NLSF', self.BaseClass),
            'trust_region':  ('NLSF', self.BaseClass),
            'unconstrained': ('NLSF', self.BaseClass),
            'poisson':       ('MLE',  self.MLEClass),
            'pearson':       ('MLE',  self.MLEClass),
            'neyman':        ('MLE',  self.MLEClass)
        }

    def compare_selected(self, methods, y_data, irf_data, model_type='bi-exponential', p0=None, bounds=None):
        """
        methods: List of strings, e.g., ['trust_region', 'poisson']
        y_data: 1D array of decay counts
        irf_data: 1D array of IRF counts
        """
        results = []
        
        print(f"\n{'='*105}")
        print(f"SELECTIVE COMPARISON | Methods: {', '.join(methods)} | Model: {model_type}")
        print(f"{'='*105}\n")

        for method in methods:
            if method not in self.method_mapping:
                print(f"Warning: Method '{method}' not recognized. Skipping.")
                continue

            category, Fitter = self.method_mapping[method]
            # Ensure data is float32 for consistency
            fitter_inst = Fitter(self.freq, y_data.astype(np.float32), irf_data.astype(np.float32))
            
            start_time = time.perf_counter()
            
            try:
                # Unified API call
                # Standard Return: (popt, perr, chi2, red_chi2, model_fit, residuals, success_flag)
                res = list(fitter_inst.fit_with_estimator(
                    estimator_type=method,
                    model_type=model_type,
                    p0=p0,
                    bounds=bounds
                ))
                
                elapsed = (time.perf_counter() - start_time) * 1000  # ms
                popt = res[0]
                chi2 = res[2]     # Based on standard tuple index
                red_chi2 = res[3]  # Based on standard tuple index
                success_val = res[6] if len(res) > 6 else 0
                success = "YES" if success_val == 1 else "NO"

                # Extract key lifetimes for easy viewing
                if model_type == 'bi-exponential' and len(popt) >= 4:
                    params = f"α1:{popt[1]:.2f}, τ1:{popt[2]:.2f}, τ2:{popt[3]:.2f}"
                elif len(popt) >= 2:
                    params = f"τ:{popt[1]:.2f}, Off:{popt[2]:.2f}"
                else:
                    params = "N/A"

                results.append([
                    method.upper(), 
                    category, 
                    success,
                    f"{elapsed:.2f} ms", 
                    f"{chi2:.1f}", 
                    f"{red_chi2:.3f}", 
                    params
                ])

            except Exception as e:
                # Foolproof catch to prevent loop breakage
                results.append([method.upper(), category, "FAIL", "N/A", "N/A", "N/A", str(e)[:20]])

        headers = ["Method", "Type", "Conv", "Time", "Chi2", "Red. Chi2", "Key Parameters"]
        print(tabulate(results, headers=headers, tablefmt="fancy_grid"))
        return results

    def run_all(self, y_data, irf_data, model_type='bi-exponential', p0=None, bounds=None):
        """Helper to run every supported method at once."""
        return self.compare_selected(list(self.method_mapping.keys()), y_data, irf_data, model_type, p0, bounds)
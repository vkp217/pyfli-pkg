import os
import numpy as np
from tabulate import tabulate

class Msg_display:
    def __init__(self, saver=None):
        """
        :param saver: An instance of the DataSaver class.
        """
        self.saver = saver

    def _internal_log(self, message):
        """Helper to print and optionally log via DataSaver."""
        print(message)
        if self.saver:
            # We strip any fancy print characters if logging to file
            self.saver.log(message)

    def disp_params(self, res_px, model_type='bi-exponential'):
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

        labels = ['Ar', 'alpha1', 'tau1', 'tau2', 'Offset'] if model_type == 'bi-exponential' else ['Ar', 'tau', 'Offset']
        
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

    def get_pixel_summary(self, data_maps, px):
        x, y = px
        table_data = []    
        for key, map_2d in data_maps.items():
            try:
                if isinstance(map_2d, np.ndarray) and map_2d.ndim == 2:
                    value = map_2d[x, y]
                    formatted_val = f"{value:.4f}" if isinstance(value, (float, np.float32, np.float64)) else value
                    table_data.append([key, formatted_val])
            except Exception as e:
                table_data.append([key, f"ERROR: {str(e)}"])
                
        headers = ["Parameters", f"Value at {px}"]
        
        # Create Table
        table_output = tabulate(table_data, headers=headers, tablefmt="fancy_grid")
        
        # Display logic
        print(f"\n{'='*50}\nPIXEL DIAGNOSTIC: {px}\n{'='*50}")
        print(table_output)

        # Log logic (using a simpler table format for the .txt file to keep it readable)
        if self.saver:
            clean_table = tabulate(table_data, headers=headers, tablefmt="plain")
            self.saver.log(f"\nPIXEL DIAGNOSTIC: {px}\n{clean_table}")
        
        return table_data
# scripts/data_text/msg_display.py

class Msg_display:
    def __init__(self):
        pass

    def disp_params(self, res_px, model_type='bi-exponential'):
        if not res_px:
            raise ValueError('Data was not provided (res_px is empty or None)')

        # Unpack for readability: assumes [params, errors, r2, chi2, red_chi2, _, convergence]
        try:
            p, err = res_px[0], res_px[1]
            r2, chi2, red_chi2 = res_px[2], res_px[3], res_px[4]
            conv = res_px[6]
        except IndexError:
            raise IndexError("res_px does not have the expected number of elements (min 7).")

        print('\n' + '='*30)
        print(f'FIT PARAMETERS ({model_type.upper()})')
        print('-'*30)

        if model_type == 'bi-exponential':
            # Ar, alpha1, tau1, tau2, Offset
            labels = ['Ar', 'alpha1', 'tau1', 'tau2', 'Offset']
            for i, label in enumerate(labels):
                print(f'{label:8}: {p[i]:.4f} \u00B1 {err[i]:.4f}')

        elif model_type == 'mono-exponential':
            # Ar, tau, Offset
            labels = ['Ar', 'tau', 'Offset']
            for i, label in enumerate(labels):
                print(f'{label:8}: {p[i]:.4f} \u00B1 {err[i]:.4f}')
        
        else:
            raise ValueError(f'Unsupported model type: {model_type}')

        print('-'*30)
        print(f'R2           : {r2:.4f}')
        print(f'chi2         : {chi2:.4f}')
        print(f'Reduced chi2 : {red_chi2:.4f}')
        print(f'Convergence  : {conv}')
        print('='*30 + '\n')

    def fit_session(self, model_type=None, processor_name=None, fitter_name=None, 
                    estimator=None, data_name=None, use_initial_guess=None, 
                    p0=None, use_bounds=None, bounds=None):
        
        print('\n' + '-'*60)
        print('FITTING SESSION INITIALIZED')
        print('-'*60)
        print(f'Decay Model       : {model_type}')
        print(f'Processor         : {processor_name}')
        print(f'Method            : {fitter_name} ({estimator})')
        print(f'Data Source       : {data_name}')
        print(f'Use Initial Guess : {use_initial_guess}')
        print(f'Initial P0        : {p0}')
        print(f'Use Bounds        : {use_bounds}')
        print(f'Bounds            : {bounds}')
        print('-'*60 + '\n')

    def fit_session(self, **kwargs):
        pretty_labels = {
            'model_type': 'Decay Model',
            'processor_name': 'Processor',
            'fitter_name': 'Fitting Method',
            'p0': 'Initial Guesses (p0)',
            'use_initial_guess': 'Using Guess',
            'use_bounds': 'Using Bounds'
        }

        print('\n' + '-' * 60)
        print(f"{'SESSION CONFIGURATION':^60}") # Centered title
        print('-' * 60)

        # Loop through whatever arguments were passed
        for key, value in kwargs.items():
            # Get the pretty label or capitalize the raw key
            label = pretty_labels.get(key, key.replace('_', ' ').capitalize())
            print(f"{label:25}: {value}")

        print('-' * 60)
        print(f"{'Session Initialized':^60}")
        print('-' * 60 + '\n')

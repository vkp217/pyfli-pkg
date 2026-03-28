# solver/comparison.py 

import numpy as np
import matplotlib.pyplot as plt
from .base_fitter import BaseFLIFitter
from .mleFitter import MLEFLIFitter

def run_comparison_simulation(trials=50, photon_count=5000):
    # 1. Setup Simulation Parameters
    freq = [80, 200]  # 80MHz Laser, 200MHz Acq (5ns window)
    true_tau1 = 0.8
    true_tau2 = 3.2
    true_a1 = 0.4
    true_bg = 2.0
    
    # Create a synthetic IRF (Gaussian)
    t = np.linspace(0, 5, 256)
    irf = np.exp(-(t - 0.5)**2 / (2 * 0.1**2))
    irf /= np.sum(irf)
    
    results_wls = []
    results_mle = []

    print(f"Running {trials} trials at ~{photon_count} photons/decay...")

    for i in range(trials):
        # 2. Generate Ground Truth Model
        # S is scaled to hit the target photon count roughly
        S_target = photon_count 
        decay_model = S_target * ((true_a1 / true_tau1) * np.exp(-t / true_tau1) + 
                                  ((1 - true_a1) / true_tau2) * np.exp(-t / true_tau2))
        convolved = np.convolve(decay_model, irf, mode='full')[:len(t)] + true_bg
        
        # 3. Add Poisson Noise (Crucial for MLE vs WLS comparison)
        noisy_decay = np.random.poisson(convolved).astype(float)
        
        # 4. Initialize Fitters
        wls_fitter = BaseFLIFitter(freq, noisy_decay, irf)
        mle_fitter = MLEFLIFitter(freq, noisy_decay, irf)
        
        # 5. Perform Fits
        try:
            p_wls, _, _, _, _, _ = wls_fitter.least_squares_fit(model_type='bi-exponential')
            p_mle, _, _, _, _, _ = mle_fitter.mle_fit(model_type='bi-exponential')
            
            # Store lifetimes (tau1, tau2)
            results_wls.append([p_wls[2], p_wls[3]])
            results_mle.append([p_mle[2], p_mle[3]])
        except Exception as e:
            continue

    # 6. Statistical Analysis
    res_wls = np.array(results_wls)
    res_mle = np.array(results_mle)

    std_wls = np.std(res_wls, axis=0)
    std_mle = np.std(res_mle, axis=0)
    
    mean_wls = np.mean(res_wls, axis=0)
    mean_mle = np.mean(res_mle, axis=0)

    # 7. Print Report
    print("\n" + "="*40)
    print(f"{'Parameter':<10} | {'True':<6} | {'WLS Mean (±STD)':<18} | {'MLE Mean (±STD)':<18}")
    print("-" * 40)
    print(f"{'Tau 1':<10} | {true_tau1:<6} | {mean_wls[0]:.3f} ± {std_wls[0]:.3f} | {mean_mle[0]:.3f} ± {std_mle[0]:.3f}")
    print(f"{'Tau 2':<10} | {true_tau2:<6} | {mean_wls[1]:.3f} ± {std_wls[1]:.3f} | {mean_mle[1]:.3f} ± {std_mle[1]:.3f}")
    print("="*40)

    # 8. Visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot([res_wls[:, 0], res_mle[:, 0]], labels=['WLS', 'MLE'])
    plt.title('Variance in Tau 1')
    plt.ylabel('Lifetime (ns)')

    plt.subplot(1, 2, 2)
    plt.boxplot([res_wls[:, 1], res_mle[:, 1]], labels=['WLS', 'MLE'])
    plt.title('Variance in Tau 2')
    
    plt.tight_layout()
    plt.show()

def run_precision_bias_study(trials=100, photon_count=1000):
    # 1. Setup Simulation Parameters
    freq = [80, 200] 
    true_tau1, true_tau2, true_a1, true_bg = 0.8, 3.2, 0.4, 1.0
    
    t = np.linspace(0, 5, 256)
    irf = np.exp(-(t - 0.5)**2 / (2 * 0.1**2))
    irf /= np.sum(irf)
    
    # Storage for Bias and Variance analysis
    methods = ['poisson', 'pearson', 'neyman']
    results = {m: [] for m in methods}
    errors = {m: [] for m in methods}

    print(f"Simulating {trials} decays at {photon_count} photons...")

    for _ in range(trials):
        # Generate Ground Truth
        decay_model = photon_count * ((true_a1 / true_tau1) * np.exp(-t / true_tau1) + 
                                     ((1 - true_a1) / true_tau2) * np.exp(-t / true_tau2))
        convolved = np.convolve(decay_model, irf, mode='full')[:len(t)] + true_bg
        noisy_decay = np.random.poisson(convolved).astype(float)
        
        fitter = MLEFLIFitter(freq, noisy_decay, irf)
        
        for m in methods:
            try:
                # popt, perr, r_sq, stat, red_stat, ssr, success, msg
                p, perr, _, _, _, _, _, _ = fitter.fit_with_estimator(estimator_type=m)
                results[m].append(p[2]) # Tracking Tau 1 for comparison
                errors[m].append(perr[2])
            except:
                continue

    # 2. Statistical Reporting
    print("\n" + "="*75)
    print(f"{'Method':<12} | {'Mean Tau1':<12} | {'Bias (%)':<12} | {'Mean StdErr':<12} | {'Var (Observed)':<12}")
    print("-" * 75)

    for m in methods:
        data = np.array(results[m])
        errs = np.array(errors[m])
        
        mean_val = np.mean(data)
        bias_pct = ((mean_val - true_tau1) / true_tau1) * 100
        mean_stderr = np.nanmean(errs)
        observed_std = np.std(data)
        
        print(f"{m:<12} | {mean_val:12.4f} | {bias_pct:11.2f}% | {mean_stderr:12.4f} | {observed_std:12.4f}")
    print("="*75)

    # 3. Visualization of Bias and Confidence Intervals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bias Plot (Tau 1 distribution)
    ax1.axhline(true_tau1, color='r', linestyle='--', label='True Value')
    ax1.boxplot([results[m] for m in methods], labels=methods)
    ax1.set_title(f'Bias Comparison (Tau 1)\nTarget: {true_tau1}ns')
    ax1.set_ylabel('Recovered Lifetime (ns)')

    # Variance/Error Bar Plot
    # Shows the "predicted" error bars from the Hessian vs the actual spread
    avg_perr = [np.nanmean(errors[m]) for m in methods]
    ax2.bar(methods, avg_perr, color='skyblue', alpha=0.7)
    ax2.set_title('Average Predicted Confidence Interval (Hessian)')
    ax2.set_ylabel('Standard Deviation (ns)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Lower photon counts (e.g. 500) will highlight the extreme bias of Neyman's method
    run_precision_bias_study(trials=50, photon_count=800)
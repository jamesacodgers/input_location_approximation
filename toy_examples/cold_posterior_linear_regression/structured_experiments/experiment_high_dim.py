import torch
import numpy as np
import matplotlib.pyplot as plt
from src.linear_utils import (compute_mfvi_analytic, compute_exact_posterior, 
                   generate_data, generate_test_data, test_nll_mfvi, 
                   test_nll_exact, compute_analytic_nll_approximations)

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)

def run_high_dim_experiment():
    # Configuration
    N_DIMS = 1024
    N_SAMPLES = 10
    INPUT_STD = 1.0
    NOISE_STD = 0.1
    DIAGONAL_INPUT = True # Highly redundant features to exacerbate CPE
    
    print(f"Running high-dim experiment: D={N_DIMS}, N={N_SAMPLES}, Noise={NOISE_STD}")
    
    # Generate Data
    X_train, y_train, true_weights = generate_data(n_samples=N_SAMPLES, n_dims=N_DIMS, 
                                                   input_std=INPUT_STD, noise_std=NOISE_STD, 
                                                   seed=42, diagonal_input=DIAGONAL_INPUT)
    
    X_test, y_test = generate_test_data(true_weights, n_samples=1000, 
                                        input_std=INPUT_STD, noise_std=NOISE_STD, 
                                        seed=123, diagonal_input=DIAGONAL_INPUT)
    
    # Compute Exact Posterior
    mu_ex, Sigma_ex = compute_exact_posterior(X_train, y_train, noise_std=NOISE_STD)
    exact_nll = test_nll_exact(X_test, y_test, mu_ex, Sigma_ex, noise_std=NOISE_STD, n_samples=1000)
    print(f"Exact posterior test NLL: {exact_nll:.4f}")
    
    # Temperatures to explore
    temperatures = np.logspace(-4, 4, 12)
    
    results = {
        'temps': [],
        'mc_nlls': [],
        'analytic_true_2nd': [],
        'var_ones': [],
        'var_orth': []
    }
    
    # Precompute directions for variance analysis
    ones_vec = torch.ones(N_DIMS) / np.sqrt(N_DIMS) # Normalized ones
    # Orthogonal vector (half 1s, half -1s)
    orth_vec = torch.ones(N_DIMS)
    orth_vec[N_DIMS//2:] = -1.0
    orth_vec = orth_vec / torch.norm(orth_vec)
    
    for T in temperatures:
        mu_vi, sigma_vi = compute_mfvi_analytic(X_train, y_train, temperature=T, noise_std=NOISE_STD)
        
        # MC NLL
        mc_nll = test_nll_mfvi(X_test, y_test, mu_vi, sigma_vi, noise_std=NOISE_STD, n_samples=500)
        
        # Analytic NLL (2nd order, True weights)
        nll_true, _, _ = compute_analytic_nll_approximations(mu_vi, sigma_vi, true_weights, Sigma_ex, INPUT_STD, NOISE_STD)
        
        # Predictive Variances (along ones and orth)
        # Var(y* | x) = x^T S x + noise^2
        # Here we look at x^T S x
        var_ones = torch.sum(sigma_vi * ones_vec**2).item()
        var_orth = torch.sum(sigma_vi * orth_vec**2).item()
        
        results['temps'].append(T)
        results['mc_nlls'].append(mc_nll)
        results['analytic_true_2nd'].append(nll_true[1])
        results['var_ones'].append(var_ones)
        results['var_orth'].append(var_orth)
        
        print(f"T={T:.1e} | MC NLL={mc_nll:.4f} | Var_Ones={var_ones:.4f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: NLL vs Temperature
    ax1.plot(results['temps'], results['mc_nlls'], 'o-', label='MC NLL')
    ax1.plot(results['temps'], results['analytic_true_2nd'], '--', label='Analytic (2nd Order)')
    ax1.axhline(y=exact_nll, color='red', linestyle=':', label='Exact Posterior')
    ax1.axvline(x=1.0, color='gray', linestyle='--')
    ax1.set_xscale('log')
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Test NLL')
    ax1.set_title(f'NLL vs Temperature (D={N_DIMS})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Variance Metrics
    ax2.plot(results['temps'], results['var_ones'], label='Var (Data Span)')
    ax2.plot(results['temps'], results['var_orth'], label='Var (Orthogonal)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Posterior Variance (x^T S x)')
    ax2.set_title('Directional Variance Components')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("high_dim_experiment_results.pdf", bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    run_high_dim_experiment()

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)


def compute_mfvi_analytic(X: torch.Tensor, y: torch.Tensor, temperature: float = 1.0, 
                          noise_std: float = 1.0):
    """
    Compute closed-form mean-field VI solution for Bayesian linear regression with COLD posterior.
    
    For temperature T, we scale both prior and likelihood:
    - Prior: N(0, T·I)
    - Likelihood: N(Xw, T·σ²I)
    
    This gives:
    - Posterior precision: (X^T X / (T σ²) + I/T) = (X^T X + σ²I) / (T σ²)
    - Mean: μ = (X^T X + σ²I)^{-1} X^T y
    - Variances: σ_i² = T · [(X^T X + σ²I)^{-1}]_ii
    
    Note: The posterior mean is INDEPENDENT of T (cold posterior property)!
    
    Args:
        X: Design matrix [n, d]
        y: Target vector [n]
        temperature: Temperature parameter
        noise_std: Noise standard deviation
    
    Returns:
        mu: Posterior mean [d]
        sigma: Posterior standard deviations [d] (diagonal only)
    """
    XTX = X.T @ X
    XTy = X.T @ y
    
    # Posterior precision (after factoring out 1/T)
    # (X^T X + σ²I) / (T σ²)
    precision = XTX + noise_std**2 * torch.eye(X.shape[1])
    
    # Posterior mean (independent of T!)
    mu = torch.linalg.solve(precision, XTy)
    
    # Posterior covariance (full, for MFVI we take diagonal)
    # Sigma = T σ² (X^T X + σ²I)^{-1}
    Sigma_full = temperature * noise_std**2 * torch.inverse(precision)
    sigma_sq = torch.diag(Sigma_full)
    sigma = torch.sqrt(sigma_sq)
    
    return mu, sigma


def compute_exact_posterior(X: torch.Tensor, y: torch.Tensor, noise_std: float = 1.0):
    """
    Compute exact posterior for Bayesian linear regression with spherical prior N(0, I).
    
    Returns:
        mu_post: Posterior mean [d]
        Sigma_post: Posterior covariance [d, d]
    """
    precision_lik = X.T @ X / (noise_std ** 2)
    precision_post = precision_lik + torch.eye(X.shape[1])
    Sigma_post = torch.inverse(precision_post)
    mu_post = Sigma_post @ (X.T @ y / (noise_std ** 2))
    
    return mu_post, Sigma_post


def test_nll_mfvi(X_test: torch.Tensor, y_test: torch.Tensor, 
                  mu: torch.Tensor, sigma: torch.Tensor,
                  noise_std: float = 1.0, n_samples: int = 500):
    """
    Compute test NLL using mean-field VI posterior.
    """
    # Sample from diagonal Gaussian
    eps = torch.randn(n_samples, len(mu))
    weights = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # [n_samples, d]
    
    # Predictions
    y_pred = torch.matmul(weights, X_test.T)  # [n_samples, n_test]
    
    # Log probabilities
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))
    
    # Monte Carlo approximation
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
    nll = -log_pred.mean().item()
    
    return nll


def test_nll_exact(X_test: torch.Tensor, y_test: torch.Tensor, 
                   mu_post: torch.Tensor, Sigma_post: torch.Tensor,
                   noise_std: float = 1.0, n_samples: int = 500):
    """
    Compute test NLL using exact posterior.
    """
    dist = torch.distributions.MultivariateNormal(mu_post, Sigma_post)
    weights = dist.sample((n_samples,))
    
    y_pred = torch.matmul(weights, X_test.T)
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
    nll = -log_pred.mean().item()
    
    return nll


def generate_data(n_samples: int = 100, n_dims: int = 10, input_std: float = 1.0, 
                  noise_std: float = 1.0, seed: int = 42):
    """Generate training data with perfectly correlated inputs (all x_i = x_1)."""
    torch.manual_seed(seed)
    true_weights = torch.randn(n_dims)
    x1 = torch.randn(n_samples) * input_std
    # All dimensions equal to x1
    X = x1.unsqueeze(1).repeat(1, n_dims)
    y = X @ true_weights + torch.randn(n_samples) * noise_std
    return X, y, true_weights


def generate_test_data(true_weights: torch.Tensor, n_samples: int = 100, 
                       input_std: float = 1.0, noise_std: float = 1.0, 
                       seed: int = 123):
    """Generate test data using the same true weights."""
    torch.manual_seed(seed)
    n_dims = len(true_weights)
    x1 = torch.randn(n_samples) * input_std
    # All dimensions equal to x1
    X_test = x1.unsqueeze(1).repeat(1, n_dims)
    y_test = X_test @ true_weights + torch.randn(n_samples) * noise_std
    return X_test, y_test


# Main experiment
if __name__ == "__main__":
    # Configuration
    N_DIMS = 100  # Number of dimensions
    input_stds = [1]
    temperatures = np.logspace(-7,0,10)
    
    NOISE_STD = 0.1  # Fixed noise standard deviation
    
    all_results = []
    
    for input_std in input_stds:
        print(f"\n{'='*60}")
        print(f"INPUT STD = {input_std:.1f} (variance = {input_std**2:.1f})")
        print('='*60)
        
        # Generate data
        X_train, y_train, true_weights = generate_data(n_samples=10, n_dims=N_DIMS, input_std=input_std, noise_std=NOISE_STD)
        X_test, y_test = generate_test_data(true_weights, n_samples=100000, input_std=input_std, noise_std=NOISE_STD)
        
        print(f"True weights (first 5): {true_weights[:5].numpy()}")
        
        # Compute exact posterior
        mu_exact, Sigma_exact = compute_exact_posterior(X_train, y_train, noise_std=NOISE_STD)
        exact_nll = test_nll_exact(X_test, y_test, mu_exact, Sigma_exact, noise_std=NOISE_STD)
        
        print(f"Exact posterior mean (first 5): {mu_exact[:5].numpy()}")
        print(f"Exact posterior test NLL: {exact_nll:.4f}")
        
        # Temperature sweep
        results = {'input_std': input_std, 'temperatures': [], 'test_nlls': [], 
                   'posterior_means': [], 'posterior_stds': [], 'exact_nll': exact_nll,
                   'X_train': X_train, 'X_test': X_test}
        
        for temp in temperatures:
            # Closed-form MFVI solution
            mu, sigma = compute_mfvi_analytic(X_train, y_train, temperature=temp, noise_std=NOISE_STD)
            
            # Compute test NLL
            test_nll = test_nll_mfvi(X_test, y_test, mu, sigma, noise_std=NOISE_STD)
            
            results['temperatures'].append(temp)
            results['test_nlls'].append(test_nll)
            results['posterior_means'].append(mu)
            results['posterior_stds'].append(sigma)
            
            print(f"T = {temp:.1e}: Test NLL = {test_nll:.4f}")
        
        all_results.append(results)
    
    # Create plots
    n_vars = len(input_stds)
    fig = plt.figure(figsize=(16, 4 * ((n_vars + 1) // 2)))
    
    # Choose two random dimensions to plot
    np.random.seed(42)
    dim1, dim2 = np.random.choice(N_DIMS, 2, replace=False)
    
    for i, res in enumerate(all_results):
        # Plot NLL vs temperature
        ax1 = plt.subplot(((n_vars + 1) // 2), 4, i * 2 + 1)
        ax1.plot(res['temperatures'], res['test_nlls'], 'o-', linewidth=2, markersize=8, label='Mean Field VI')
        ax1.axhline(y=res['exact_nll'], color='green', linestyle='--', linewidth=2, label='Exact Posterior')
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='T=1')
        ax1.set_xlabel('Temperature', fontsize=10)
        ax1.set_ylabel('Test NLL', fontsize=10)
        ax1.set_xscale('log')
        ax1.set_title(f'σ_x = {res["input_std"]:.1f} (var = {res["input_std"]**2:.1f})', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot data distribution (two random dimensions)
        ax2 = plt.subplot(((n_vars + 1) // 2), 4, i * 2 + 2)
        ax2.scatter(res['X_train'][:, dim1].numpy(), res['X_train'][:, dim2].numpy(), 
                    alpha=0.6, s=20, label='Train')
        ax2.scatter(res['X_test'][:, dim1].numpy(), res['X_test'][:, dim2].numpy(), 
                    alpha=0.6, s=20, label='Test')
        lim = 3 * res['input_std']
        ax2.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label=f'x{dim1} = x{dim2}')
        ax2.set_xlabel(f'x{dim1}', fontsize=10)
        ax2.set_ylabel(f'x{dim2}', fontsize=10)
        ax2.set_title(f'Input Space (dims {dim1}, {dim2} of {N_DIMS})', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("toy_examples/cold_posterior_linear_regression/analytic_CPE_cold.pdf", bbox_inches="tight")
    plt.show()
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: Best Temperature for Each Input Variance")
    print("="*80)
    print(f"{'Input Std':<12} {'Variance':<12} {'Best Temp':<12} {'Best NLL':<12} {'Exact NLL':<12}")
    print("-"*80)
    for res in all_results:
        best_idx = np.argmin(res['test_nlls'])
        best_temp = res['temperatures'][best_idx]
        best_nll = res['test_nlls'][best_idx]
        print(f"{res['input_std']:<12.1f} {res['input_std']**2:<12.1f} {best_temp:<12.6f} {best_nll:<12.4f} {res['exact_nll']:<12.4f}")
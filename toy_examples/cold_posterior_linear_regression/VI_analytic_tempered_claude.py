import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)


def compute_mfvi_analytic(X: torch.Tensor, y: torch.Tensor, temperature: float = 1.0, 
                          noise_std: float = 1.0):
    """
    Compute closed-form mean-field VI solution for Bayesian linear regression.
    
    For temperature T, prior N(0, I), and likelihood N(Xw, σ²I):
    
    Mean: μ = (X^T X / (T σ²) + I)^{-1} (X^T y / (T σ²))
    Variances: σ_i² = T σ² / ((X^T X)_ii + T σ²)
    
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
    
    # Posterior precision
    precision = XTX / (temperature * noise_std**2) + torch.eye(X.shape[1])
    
    # Posterior mean
    mu = torch.linalg.solve(precision, XTy / (temperature * noise_std**2))
    
    # Posterior variances (diagonal only for mean-field)
    XTX_diag = torch.diag(XTX)
    sigma_sq = temperature * noise_std**2 / (XTX_diag + temperature * noise_std**2)
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


def compute_delta_approximations(mu: torch.Tensor, Sigma_diag: torch.Tensor, 
                                 Sigma_exact: torch.Tensor, noise_std: float = 1.0):
    """
    Compute first and second-order delta method approximations for the expected log density.
    
    For the expression:
    E_p(y|x,β)[q(y|x)] = -1/2 * (Tr((β-m)(β-m)^T xx^T) + σ² / (Tr(S xx^T) + σ²) 
                                  + log(2π(Tr(S xx^T) + σ²)))
    
    We linearize w.r.t. X = xx^T around E[X] = Σ_exact.
    
    Args:
        mu: MFVI posterior mean [d]
        Sigma_diag: MFVI posterior variances (diagonal) [d]
        Sigma_exact: Exact posterior covariance [d, d]
        noise_std: Observation noise standard deviation
        
    Returns:
        first_order: First-order approximation (evaluating at mean)
        second_order: Second-order correction term
    """
    sigma_sq = noise_std ** 2
    
    # C = (β - m)(β - m)^T for mean field: diagonal matrix with Sigma_diag on diagonal
    # For mean field, this is just diag(Sigma_diag)
    C = Sigma_exact
    
    # S = diag(Sigma_diag) for mean field posterior
    S = torch.diag(Sigma_diag)

    cov_x = torch.ones_like(mu).unsqueeze(-1) @ torch.ones_like(mu).unsqueeze(0) 
    
    # Evaluate at X = Sigma_exact
    u = torch.trace(C @ cov_x) + sigma_sq  # Tr(C Σ) + σ²
    v = torch.trace(S @ cov_x) + sigma_sq  # Tr(S Σ) + σ²
    
    # First-order approximation: f(Σ)
    term_A = u / v
    term_B = torch.log(2 * np.pi * v)
    first_order = -0.5 * (term_A + term_B)
    
    # Second-order correction
    # We need covariance terms for x ~ N(0, Sigma_exact)
    # Cov(Tr(A xx^T), Tr(B xx^T)) = Tr(A Σ B Σ) + Tr(A Σ B^T Σ)
    
    # For symmetric matrices, Tr(A Σ B^T Σ) = Tr(A Σ B Σ)
    # So: Cov(Tr(A xx^T), Tr(B xx^T)) = 2 * Tr(A Σ B Σ)
    
    cov_CS = 2 * torch.trace(C @ cov_x @ S @ cov_x)
    var_S = 2 * torch.trace(S @ cov_x @ S @ cov_x)
    
    # Second-order correction
    # Δ = -1/(4v²) * [2*Cov(C,S) - (2u/v + 1)*Var(S)]
    second_order = -1.0 / (4 * v**2) * (4 * cov_CS - (2 * u / v + 1) *2 *var_S)
    # second_order = first_order + second_order
    return first_order.item(), second_order.item()


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


def compute_delta_nll(X_test: torch.Tensor, y_test: torch.Tensor,
                      mu: torch.Tensor, sigma: torch.Tensor,
                      Sigma_exact: torch.Tensor, noise_std: float = 1.0,
                      order: int = 2):
    """
    Compute test NLL using delta method approximation.
    
    Args:
        order: 1 for first-order, 2 for second-order approximation
    """
    n_test = len(y_test)
    total_nll = 0.0
    
    for i in range(n_test):
        x = X_test[i]  # [d]
        y = y_test[i].item()
        
        # Compute Sigma for this test point
        # For input x, we need E[xx^T] which depends on the data distribution
        # Here we use Sigma_exact as the covariance structure
        Sigma_x = torch.outer(x, x)  # This is the observed xx^T
        
        # Use Sigma_exact to approximate the distribution of x
        # First order: evaluate at E[xx^T]
        Sigma_diag = sigma ** 2
        
        first, second = compute_delta_approximations(mu, Sigma_diag, Sigma_exact, noise_std)
        
        if order == 1:
            nll_i = -first
        else:  # order == 2
            nll_i = -(first + second)
        
        total_nll += nll_i
    
    return total_nll / n_test


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
    N_DIMS = 500  # Reduced for faster computation with delta method
    input_stds = [1]
    temperatures = np.logspace(-2, 0, 15)
    
    NOISE_STD = 0.1  # Fixed noise standard deviation
    
    all_results = []
    
    for input_std in input_stds:
        print(f"\n{'='*60}")
        print(f"INPUT STD = {input_std:.1f} (variance = {input_std**2:.1f})")
        print('='*60)
        
        # Generate data
        X_train, y_train, true_weights = generate_data(n_samples=100, n_dims=N_DIMS, 
                                                       input_std=input_std, noise_std=NOISE_STD)
        X_test, y_test = generate_test_data(true_weights, n_samples=200, 
                                           input_std=input_std, noise_std=NOISE_STD)
        
        print(f"True weights (first 5): {true_weights[:5].numpy()}")
        
        # Compute exact posterior
        mu_exact, Sigma_exact = compute_exact_posterior(X_train, y_train, noise_std=NOISE_STD)
        exact_nll = test_nll_exact(X_test, y_test, mu_exact, Sigma_exact, noise_std=NOISE_STD)
        
        print(f"Exact posterior mean (first 5): {mu_exact[:5].numpy()}")
        print(f"Exact posterior test NLL: {exact_nll:.4f}")
        
        # Temperature sweep
        results = {
            'input_std': input_std, 
            'temperatures': [], 
            'test_nlls': [], 
            'delta_nlls_1st': [],
            'delta_nlls_2nd': [],
            'posterior_means': [], 
            'posterior_stds': [], 
            'exact_nll': exact_nll,
            'X_train': X_train, 
            'X_test': X_test
        }
        
        for temp in temperatures:
            # Closed-form MFVI solution
            mu, sigma = compute_mfvi_analytic(X_train, y_train, temperature=temp, noise_std=NOISE_STD)
            
            # Compute test NLL (Monte Carlo)
            test_nll = test_nll_mfvi(X_test, y_test, mu, sigma, noise_std=NOISE_STD)
            
            # Compute delta method approximations
            delta_nll_1st = compute_delta_nll(X_test, y_test, mu, sigma, Sigma_exact, 
                                             noise_std=NOISE_STD, order=1)
            delta_nll_2nd = compute_delta_nll(X_test, y_test, mu, sigma, Sigma_exact,
                                             noise_std=NOISE_STD, order=2)
            
            results['temperatures'].append(temp)
            results['test_nlls'].append(test_nll)
            results['delta_nlls_1st'].append(delta_nll_1st)
            results['delta_nlls_2nd'].append(delta_nll_2nd)
            results['posterior_means'].append(mu)
            results['posterior_stds'].append(sigma)
            
            print(f"T = {temp:.1e}: MC NLL = {test_nll:.4f}, "
                  f"Delta-1st = {delta_nll_1st:.4f}, Delta-2nd = {delta_nll_2nd:.4f}")
        
        all_results.append(results)
    
    # Create plots
    n_vars = len(input_stds)
    fig = plt.figure(figsize=(20, 4 * ((n_vars + 1) // 2)))
    
    # Choose two random dimensions to plot
    np.random.seed(42)
    dim1, dim2 = np.random.choice(N_DIMS, 2, replace=False)
    
    for i, res in enumerate(all_results):
        # Plot NLL vs temperature
        ax1 = plt.subplot(((n_vars + 1) // 2), 5, i * 3 + 1)
        ax1.plot(res['temperatures'], res['test_nlls'], 'o-', linewidth=2, 
                markersize=8, label='Monte Carlo', color='blue')
        ax1.plot(res['temperatures'], res['delta_nlls_1st'], 's--', linewidth=2,
                markersize=6, label='Delta (1st order)', color='orange', alpha=0.7)
        ax1.plot(res['temperatures'], res['delta_nlls_2nd'], '^--', linewidth=2,
                markersize=6, label='Delta (2nd order)', color='red', alpha=0.7)
        ax1.axhline(y=res['exact_nll'], color='green', linestyle='--', 
                   linewidth=2, label='Exact Posterior')
        ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='T=1')
        ax1.set_xlabel('Temperature', fontsize=10)
        ax1.set_ylabel('Test NLL', fontsize=10)
        ax1.set_xscale('log')
        ax1.set_title(f'σ_x = {res["input_std"]:.1f} (var = {res["input_std"]**2:.1f})', 
                     fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot approximation errors
        ax2 = plt.subplot(((n_vars + 1) // 2), 5, i * 3 + 2)
        errors_1st = np.array(res['delta_nlls_1st']) - np.array(res['test_nlls'])
        errors_2nd = np.array(res['delta_nlls_2nd']) - np.array(res['test_nlls'])
        ax2.plot(res['temperatures'], np.abs(errors_1st), 's--', linewidth=2,
                markersize=6, label='|Error| 1st order', color='orange')
        ax2.plot(res['temperatures'], np.abs(errors_2nd), '^--', linewidth=2,
                markersize=6, label='|Error| 2nd order', color='red')
        ax2.set_xlabel('Temperature', fontsize=10)
        ax2.set_ylabel('Absolute Error', fontsize=10)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_title('Delta Method Approximation Error', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # Plot data distribution (two random dimensions)
        ax3 = plt.subplot(((n_vars + 1) // 2), 5, i * 3 + 3)
        ax3.scatter(res['X_train'][:, dim1].numpy(), res['X_train'][:, dim2].numpy(), 
                   alpha=0.6, s=20, label='Train')
        ax3.scatter(res['X_test'][:, dim1].numpy(), res['X_test'][:, dim2].numpy(), 
                   alpha=0.6, s=20, label='Test')
        lim = 3 * res['input_std']
        ax3.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label=f'x{dim1} = x{dim2}')
        ax3.set_xlabel(f'x{dim1}', fontsize=10)
        ax3.set_ylabel(f'x{dim2}', fontsize=10)
        ax3.set_title(f'Input Space (dims {dim1}, {dim2} of {N_DIMS})', fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=8)
        ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig("analytic_CPE_with_delta.pdf", bbox_inches="tight")
    plt.show()
    
    # Summary table
    print("\n" + "="*100)
    print("SUMMARY: Best Temperature for Each Method")
    print("="*100)
    print(f"{'Input Std':<12} {'Method':<20} {'Best Temp':<12} {'Best NLL':<12} {'Exact NLL':<12}")
    print("-"*100)
    for res in all_results:
        # Monte Carlo
        best_idx_mc = np.argmin(res['test_nlls'])
        best_temp_mc = res['temperatures'][best_idx_mc]
        best_nll_mc = res['test_nlls'][best_idx_mc]
        
        # Delta 1st order
        best_idx_d1 = np.argmin(res['delta_nlls_1st'])
        best_temp_d1 = res['temperatures'][best_idx_d1]
        best_nll_d1 = res['delta_nlls_1st'][best_idx_d1]
        
        # Delta 2nd order
        best_idx_d2 = np.argmin(res['delta_nlls_2nd'])
        best_temp_d2 = res['temperatures'][best_idx_d2]
        best_nll_d2 = res['delta_nlls_2nd'][best_idx_d2]
        
        print(f"{res['input_std']:<12.1f} {'Monte Carlo':<20} {best_temp_mc:<12.6f} "
              f"{best_nll_mc:<12.4f} {res['exact_nll']:<12.4f}")
        print(f"{'':<12} {'Delta (1st order)':<20} {best_temp_d1:<12.6f} "
              f"{best_nll_d1:<12.4f} {res['exact_nll']:<12.4f}")
        print(f"{'':<12} {'Delta (2nd order)':<20} {best_temp_d2:<12.6f} "
              f"{best_nll_d2:<12.4f} {res['exact_nll']:<12.4f}")
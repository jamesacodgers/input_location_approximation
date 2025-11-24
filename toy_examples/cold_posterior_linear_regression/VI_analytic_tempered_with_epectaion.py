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


def compute_analytic_nll_approximations(mu: torch.Tensor, sigma: torch.Tensor, 
                                      true_weights: torch.Tensor, 
                                      input_std: float, noise_std: float):
    """
    Computes 1st and 2nd order Delta Method approximations of the Expected NLL.
    
    Optimized for the specific case where Input Covariance Sigma = input_std^2 * Ones_Matrix.
    This avoids O(D^3) matrix operations.
    """
    D = len(mu)
    var_x = input_std**2
    
    # 1. Setup variables
    # beta_diff = (beta_true - m)
    beta_diff = true_weights - mu
    
    # S is diagonal posterior covariance (sigma is std dev)
    S_diag = sigma**2
    
    # 2. Compute Traces using Rank-1 shortcut
    # Because Sigma = var_x * 1 * 1.T:
    # Tr(S @ Sigma) = var_x * sum(S_diag)
    tr_S_Sigma = var_x * torch.sum(S_diag)
    
    # (beta-m).T @ Sigma @ (beta-m) = var_x * (sum(beta_diff))^2
    quad_M_Sigma = var_x * (torch.sum(beta_diff))**2
    
    # Tr((S @ Sigma)^2) = var_x^2 * (sum(S_diag))^2
    tr_S_Sigma_sq = (var_x**2) * (torch.sum(S_diag))**2
    
    # Tr(M @ Sigma @ S @ Sigma) = var_x^2 * (sum(beta_diff))^2 * sum(S_diag)
    tr_M_Sigma_S_Sigma = (var_x**2) * (torch.sum(beta_diff)**2) * torch.sum(S_diag)

    # 3. Define Numerator (N) and Denominator (K) terms
    # K = Tr(S \Sigma) + \sigma^2
    K = tr_S_Sigma + noise_std**2
    
    # N = Tr(M \Sigma) + \sigma^2
    N = quad_M_Sigma + noise_std**2

    # 4. First Order Approximation
    # NLL ≈ 0.5 * (N/K + log(2*pi*K))
    nll_1st = 0.5 * (N / K + torch.log(2 * torch.tensor(np.pi) * K))

    # 5. Second Order Correction
    # Based on derived expansion: Correction = (K-2N)/(2K^3) * Var(b) + 1/(K^2) * Cov(a,b)
    # Note: The expansion was for Expected Log Likelihood. NLL is negative of that.
    # So NLL_2nd = NLL_1st - Correction
    
    correction = (
        ((K - 2*N) / (2 * K**3)) * tr_S_Sigma_sq + 
        (1 / K**2) * tr_M_Sigma_S_Sigma
    )
    
    nll_2nd = nll_1st - correction
    
    return nll_1st.item(), nll_2nd.item()


def compute_exact_posterior(X: torch.Tensor, y: torch.Tensor, noise_std: float = 1.0):
    precision_lik = X.T @ X / (noise_std ** 2)
    precision_post = precision_lik + torch.eye(X.shape[1])
    Sigma_post = torch.inverse(precision_post)
    mu_post = Sigma_post @ (X.T @ y / (noise_std ** 2))
    return mu_post, Sigma_post


def test_nll_mfvi(X_test: torch.Tensor, y_test: torch.Tensor, 
                  mu: torch.Tensor, sigma: torch.Tensor,
                  noise_std: float = 1.0, n_samples: int = 500):
    eps = torch.randn(n_samples, len(mu))
    weights = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    y_pred = torch.matmul(weights, X_test.T)
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
    nll = -log_pred.mean().item()
    return nll


def test_nll_exact(X_test: torch.Tensor, y_test: torch.Tensor, 
                   mu_post: torch.Tensor, Sigma_post: torch.Tensor,
                   noise_std: float = 1.0, n_samples: int = 500):
    dist = torch.distributions.MultivariateNormal(mu_post, Sigma_post)
    weights = dist.sample((n_samples,))
    y_pred = torch.matmul(weights, X_test.T)
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
    nll = -log_pred.mean().item()
    return nll


def generate_data(n_samples: int = 100, n_dims: int = 10, input_std: float = 1.0, 
                  noise_std: float = 1.0, seed: int = 42):
    torch.manual_seed(seed)
    true_weights = torch.randn(n_dims)
    x1 = torch.randn(n_samples) * input_std
    X = x1.unsqueeze(1).repeat(1, n_dims)
    y = X @ true_weights + torch.randn(n_samples) * noise_std
    return X, y, true_weights


def generate_test_data(true_weights: torch.Tensor, n_samples: int = 100, 
                       input_std: float = 1.0, noise_std: float = 1.0, 
                       seed: int = 123):
    torch.manual_seed(seed)
    n_dims = len(true_weights)
    x1 = torch.randn(n_samples) * input_std
    X_test = x1.unsqueeze(1).repeat(1, n_dims)
    y_test = X_test @ true_weights + torch.randn(n_samples) * noise_std
    return X_test, y_test


# Main experiment
if __name__ == "__main__":
    # Configuration
    N_DIMS = 5000
    input_stds = [1.0]
    temperatures = np.logspace(-7, 0, 20)
    NOISE_STD = 0.1
    
    all_results = []
    
    for input_std in input_stds:
        print(f"\n{'='*60}")
        print(f"INPUT STD = {input_std:.1f} (variance = {input_std**2:.1f})")
        print('='*60)
        
        X_train, y_train, true_weights = generate_data(n_samples=100, n_dims=N_DIMS, input_std=input_std, noise_std=NOISE_STD)
        X_test, y_test = generate_test_data(true_weights, n_samples=2000, input_std=input_std, noise_std=NOISE_STD)
        
        mu_exact, Sigma_exact = compute_exact_posterior(X_train, y_train, noise_std=NOISE_STD)
        exact_nll = test_nll_exact(X_test, y_test, mu_exact, Sigma_exact, noise_std=NOISE_STD)
        
        print(f"Exact posterior test NLL: {exact_nll:.4f}")
        
        results = {
            'input_std': input_std, 
            'temperatures': [], 
            'test_nlls': [], 
            'analytic_1st': [], 
            'analytic_2nd': [],
            'exact_nll': exact_nll,
            'X_train': X_train,
            'X_test': X_test
        }
        
        for temp in temperatures:
            mu, sigma = compute_mfvi_analytic(X_train, y_train, temperature=temp, noise_std=NOISE_STD)
            
            # Monte Carlo Test NLL
            test_nll = test_nll_mfvi(X_test, y_test, mu, sigma, noise_std=NOISE_STD)
            
            # Analytic Approximations
            nll_1, nll_2 = compute_analytic_nll_approximations(mu, sigma, true_weights, input_std, NOISE_STD)
            
            results['temperatures'].append(temp)
            results['test_nlls'].append(test_nll)
            results['analytic_1st'].append(nll_1)
            results['analytic_2nd'].append(nll_2)
            
            print(f"T = {temp:.1e}: MC NLL = {test_nll:.4f} | 2nd Order = {nll_2:.4f}")
        
        all_results.append(results)
    
    # Create plots
    n_vars = len(input_stds)
    fig = plt.figure(figsize=(16, 6 * ((n_vars + 1) // 2)))
    
    np.random.seed(42)
    dim1, dim2 = np.random.choice(N_DIMS, 2, replace=False)
    
    for i, res in enumerate(all_results):
        ax1 = plt.subplot(((n_vars + 1) // 2), 2, i * 2 + 1)
        
        # Plot MC Estimate
        ax1.plot(res['temperatures'], res['test_nlls'], 'o-', color='blue', 
                 linewidth=2, markersize=6, label='MC Estimate')
        
        # Plot Analytic Approximations
        ax1.plot(res['temperatures'], res['analytic_1st'], '--', color='orange', 
                 linewidth=2, label='1st Order Approx')
        ax1.plot(res['temperatures'], res['analytic_2nd'], '-', color='red', 
                 linewidth=2, alpha=0.7, label='2nd Order Approx')
        
        ax1.axhline(y=res['exact_nll'], color='green', linestyle=':', 
                    linewidth=2, label='Exact Posterior')
        ax1.axvline(x=1.0, color='grey', linestyle='--', alpha=0.5, label='T=1')
        
        ax1.set_xlabel('Temperature', fontsize=10)
        ax1.set_ylabel('Test NLL', fontsize=10)
        ax1.set_xscale('log')
        ax1.set_title(f'NLL Approximations (σ_x = {res["input_std"]:.1f})', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot data
        ax2 = plt.subplot(((n_vars + 1) // 2), 2, i * 2 + 2)
        ax2.scatter(res['X_train'][:, dim1].numpy(), res['X_train'][:, dim2].numpy(), 
                    alpha=0.6, s=20, label='Train')
        lim = 3 * res['input_std']
        ax2.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3)
        ax2.set_xlabel(f'x{dim1}')
        ax2.set_ylabel(f'x{dim2}')
        ax2.set_title(f'Input Space (Correlated)', fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("toy_examples/cold_posterior_linear_regression/analytic_CPE_with_approx.pdf", bbox_inches="tight")
    plt.show()
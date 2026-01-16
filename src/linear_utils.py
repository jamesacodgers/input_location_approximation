import torch
import numpy as np
from torch.distributions import Normal

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)

def compute_mfvi_analytic(X: torch.Tensor, y: torch.Tensor, temperature: float = 1.0, 
                          noise_std: float = 1.0):
    """
    Compute closed-form mean-field VI solution for Cold Posterior Bayesian linear regression.
    Both likelihood and prior are scaled by 1/T.
    """
    XTX = X.T @ X
    XTy = X.T @ y
    
    # Posterior precision (Cold: 1/T * (Likelihood Precision + Prior Precision))
    # Total precision = 1/T * (X^T X / sigma^2 + I)
    precision = (XTX / (noise_std**2) + torch.eye(X.shape[1])) / temperature
    
    # Posterior mean (Independent of T)
    mu = torch.linalg.solve(precision, XTy / (temperature * noise_std**2))
    
    # Posterior variances (diagonal only for mean-field)
    XTX_diag = torch.diag(XTX)
    sigma_sq = (temperature * noise_std**2) / (XTX_diag + noise_std**2)
    sigma = sigma_sq
    
    return mu, sigma

def compute_exact_posterior(X: torch.Tensor, y: torch.Tensor, noise_std: float = 1.0):
    """Compute the exact analytical posterior (not tempered)."""
    precision_lik = X.T @ X / (noise_std ** 2)
    precision_post = precision_lik + torch.eye(X.shape[1])
    Sigma_post = torch.inverse(precision_post)
    mu_post = Sigma_post @ (X.T @ y / (noise_std ** 2))
    return mu_post, Sigma_post

def test_nll_mfvi(X_test: torch.Tensor, y_test: torch.Tensor, 
                  mu: torch.Tensor, sigma: torch.Tensor,
                  noise_std: float = 1.0, n_samples: int = 500):
    """Monte Carlo estimation of NLL for MFVI posterior."""
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
    """Monte Carlo estimation of NLL for Exact posterior."""
    dist = torch.distributions.MultivariateNormal(mu_post, Sigma_post)
    weights = dist.sample((n_samples,))
    y_pred = torch.matmul(weights, X_test.T)
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
    nll = -log_pred.mean().item()
    return nll

def generate_data(n_samples: int = 100, n_dims: int = 10, input_std: float = 1.0, 
                  noise_std: float = 1.0, seed: int = 42, diagonal_input: bool = True):
    """Generate linear regression data with a specific input structure."""
    torch.manual_seed(seed)
    true_weights = torch.randn(n_dims)
    if diagonal_input:
        # All features are copies of the first one (high redundancy)
        x1 = torch.randn(n_samples) * input_std
        X = x1.unsqueeze(1).repeat(1, n_dims)
    else:
        # Standard independent features
        X = torch.randn(n_samples, n_dims) * input_std
        
    y = X @ true_weights + torch.randn(n_samples) * noise_std
    return X, y, true_weights

def generate_test_data(true_weights: torch.Tensor, n_samples: int = 100, 
                       input_std: float = 1.0, noise_std: float = 1.0, 
                       seed: int = 123, diagonal_input: bool = True):
    """Generate test data consistent with training data structure."""
    torch.manual_seed(seed)
    n_dims = len(true_weights)
    if diagonal_input:
        x1 = torch.randn(n_samples) * input_std
        X_test = x1.unsqueeze(1).repeat(1, n_dims)
    else:
        X_test = torch.randn(n_samples, n_dims) * input_std
        
    y_test = X_test @ true_weights + torch.randn(n_samples) * noise_std
    return X_test, y_test

def compute_analytic_nll_approximations(mu: torch.Tensor, sigma: torch.Tensor, 
                                      true_weights: torch.Tensor, Sigma_exact: torch.Tensor,
                                      input_std: float, noise_std: float):
    """
    Computes 1st and 2nd order Delta Method approximations of the Expected NLL.
    Optimized for the specific case where Input Covariance Sigma = input_std^2 * Ones_Matrix.
    """
    D = len(mu)
    var_x = input_std**2
    S_diag = sigma
    
    tr_S_Sigma = var_x * torch.sum(S_diag)
    tr_S_Sigma_sq = (var_x**2) * (torch.sum(S_diag))**2
    K = tr_S_Sigma + noise_std**2
    
    def compute_nll_internal(N, correction_term):
        nll_1 = 0.5 * (N / K + torch.log(2 * torch.tensor(np.pi) * K))
        corr = ((K - 2*N) / (2 * K**3)) * tr_S_Sigma_sq + (1 / K**2) * correction_term
        nll_2 = nll_1 - corr
        return nll_1.item(), nll_2.item()

    # 1. True Weights
    beta_diff = true_weights - mu
    tr_M_Sigma_true = var_x * (torch.sum(beta_diff))**2
    tr_M_Sigma_S_Sigma_true = (var_x**2) * (torch.sum(beta_diff)**2) * torch.sum(S_diag)
    nll_true = compute_nll_internal(tr_M_Sigma_true + noise_std**2, tr_M_Sigma_S_Sigma_true)

    # 2. Approx Posterior
    tr_M_Sigma_approx = tr_S_Sigma
    tr_M_Sigma_S_Sigma_approx = tr_S_Sigma_sq
    nll_approx = compute_nll_internal(tr_M_Sigma_approx + noise_std**2, tr_M_Sigma_S_Sigma_approx)

    # 3. Exact Posterior
    tr_M_Sigma_exact = var_x * torch.sum(Sigma_exact)
    tr_M_Sigma_S_Sigma_exact = (var_x**2) * torch.sum(S_diag) * torch.sum(Sigma_exact)
    nll_exact = compute_nll_internal(tr_M_Sigma_exact + noise_std**2, tr_M_Sigma_S_Sigma_exact)
    
    return nll_true, nll_approx, nll_exact

def polynomial_basis(X: torch.Tensor, degree: int = 1):
    """Apply polynomial basis transformation."""
    if degree == 1:
        return X
    
    feats = [X]
    for d in range(2, degree + 1):
        feats.append(X**d)
    return torch.cat(feats, dim=1)

def rbf_basis(X: torch.Tensor, centers: torch.Tensor, lengthscale: float = 1.0):
    """Apply RBF basis transformation: exp(-||x - c||^2 / (2 * l^2))."""
    # X: (N, D), centers: (M, D)
    # Output: (N, M)
    dist_sq = torch.cdist(X, centers)**2
    return torch.exp(-dist_sq / (2 * lengthscale**2))

def apply_basis(X: torch.Tensor, basis_type: str = 'identity', **kwargs):
    """Generic basis application wrapper."""
    if basis_type == 'identity' or basis_type is None:
        return X
    elif basis_type == 'polynomial':
        return polynomial_basis(X, degree=kwargs.get('degree', 1))
    elif basis_type == 'rbf':
        return rbf_basis(X, centers=kwargs['centers'], lengthscale=kwargs.get('lengthscale', 1.0))
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

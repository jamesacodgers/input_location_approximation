import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal
from typing import Tuple, Dict, List

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)


class MeanFieldVI:
    """Mean Field Variational Inference for Bayesian Linear Regression."""
    
    def __init__(self, input_dim: int = 2, temperature: float = 1.0):
        """
        Args:
            input_dim: Dimension of input features
            temperature: Temperature for tempering the posterior (1.0 = standard, <1.0 = cold)
        """
        self.input_dim = input_dim
        self.temperature = temperature
        
        # Variational parameters (mean and log std for each weight)
        self.q_mean = nn.Parameter(torch.zeros(input_dim))
        self.q_log_std = nn.Parameter(torch.zeros(input_dim))
        
        # Prior: standard normal N(0, I)
        self.prior_mean = torch.zeros(input_dim)
        self.prior_std = torch.ones(input_dim)
        
    def sample_weights(self, n_samples: int = 1) -> torch.Tensor:
        """Sample weights from variational distribution."""
        q_std = torch.exp(self.q_log_std)
        eps = torch.randn(n_samples, self.input_dim)
        return self.q_mean + q_std * eps
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between q and prior (analytical for Gaussians)."""
        q_std = torch.exp(self.q_log_std)
        
        # KL(q||p) for Gaussian = 0.5 * (σ²/σ₀² + (μ-μ₀)²/σ₀² - 1 - log(σ²/σ₀²))
        var_ratio = (q_std / self.prior_std) ** 2
        mean_diff_sq = ((self.q_mean - self.prior_mean) / self.prior_std) ** 2
        
        kl = 0.5 * (var_ratio + mean_diff_sq - 1 - torch.log(var_ratio))
        return kl.sum()
    
    def elbo(self, X: torch.Tensor, y: torch.Tensor, n_samples: int = 100000, 
             noise_std: float = 0.01) -> torch.Tensor:
        """Compute the Evidence Lower Bound (ELBO) with temperature."""
        # Sample weights
        weights = self.sample_weights(n_samples)
        
        # Compute predictions for each weight sample
        y_pred = torch.matmul(weights, X.T)  # [n_samples, n_data]
        
        # Log likelihood
        log_lik = Normal(y_pred, noise_std).log_prob(y.unsqueeze(0)).sum(dim=-1)
        expected_log_lik = log_lik.mean() 
        
        # KL divergence
        kl = self.kl_divergence()
        
        # ELBO = E[log p(y|x,w)]/T - KL[q||p]
        return (expected_log_lik - kl)/ self.temperature
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, n_iterations: int = 5000, 
            lr: float = 0.01, n_samples: int = 1000, noise_std: float = 1.0,
            verbose: bool = False):
        """Fit the model using gradient ascent on ELBO."""
        optimizer = optim.Adam([self.q_mean, self.q_log_std], lr=lr)
        ll = []
        for i in range(n_iterations):
            optimizer.zero_grad()
            loss = -self.elbo(X, y, n_samples, noise_std)
            loss.backward()
            optimizer.step()
            ll.append(loss.detach().item())
            
        
        plt.plot(ll)
        plt.show()
                
    
    def test_nll(self, X: torch.Tensor, y: torch.Tensor, n_samples: int = 1000, 
                 noise_std: float = 1.0) -> float:
        """Compute negative log-likelihood on test data."""
        weights = self.sample_weights(n_samples)  # [n_samples, input_dim]
        y_pred = torch.matmul(weights, X.T)  # [n_samples, n_data]
        log_probs = Normal(y_pred, noise_std).log_prob(y.unsqueeze(0))  # [n_samples, n_data]
        log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)  # [n_data]
        nll = -log_pred.mean().item()
        return nll


def generate_data(n_samples: int = 100, input_std: float = 1.0, 
                  noise_std: float = 1.0, seed: int = 42):
    """Generate training data with perfectly correlated inputs (x1 = x2)."""
    torch.manual_seed(seed)
    true_weights = torch.randn(2)
    x1 = torch.randn(n_samples) * input_std
    X = torch.stack([x1, x1], dim=1)
    y = X @ true_weights + torch.randn(n_samples) * noise_std
    return X, y, true_weights


def generate_test_data(true_weights: torch.Tensor, n_samples: int = 100, 
                       input_std: float = 1.0, noise_std: float = 1.0, 
                       seed: int = 123):
    """Generate test data using the same true weights."""
    torch.manual_seed(seed)
    x1 = torch.randn(n_samples) * input_std
    X_test = torch.stack([x1, x1], dim=1)
    y_test = X_test @ true_weights + torch.randn(n_samples) * noise_std
    return X_test, y_test


def compute_exact_posterior(X: torch.Tensor, y: torch.Tensor, noise_std: float = 1.0):
    """
    Compute exact posterior for Bayesian linear regression with spherical prior N(0, I).
    
    Posterior is Gaussian: p(w|X,y) = N(w | mu_post, Sigma_post)
    where:
        Sigma_post = (X^T X / sigma^2 + I)^{-1}
        mu_post = Sigma_post @ (X^T y / sigma^2)
    
    Returns:
        mu_post: Posterior mean [2]
        Sigma_post: Posterior covariance [2, 2]
    """
    # Precision of likelihood
    precision_lik = X.T @ X / (noise_std ** 2)
    
    # Posterior precision = prior precision + likelihood precision
    # Prior precision is I for N(0, I)
    precision_post = precision_lik + torch.eye(2)
    
    # Posterior covariance
    Sigma_post = torch.inverse(precision_post)
    
    # Posterior mean
    mu_post = Sigma_post @ (X.T @ y / (noise_std ** 2))
    
    return mu_post, Sigma_post


def exact_test_nll(X_test: torch.Tensor, y_test: torch.Tensor, 
                   mu_post: torch.Tensor, Sigma_post: torch.Tensor, 
                   noise_std: float = 1.0, n_samples: int = 5000):
    """
    Compute test NLL using exact posterior.
    Uses Monte Carlo with samples from exact posterior.
    """
    # Sample from exact posterior
    dist = torch.distributions.MultivariateNormal(mu_post, Sigma_post)
    weights = dist.sample((n_samples,))  # [n_samples, 2]
    
    # Compute predictions
    y_pred = torch.matmul(weights, X_test.T)  # [n_samples, n_data]
    
    # Compute log likelihood for each sample
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))  # [n_samples, n_data]
    
    # Monte Carlo approximation
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)  # [n_data]
    nll = -log_pred.mean().item()
    
    return nll


# Main experiment
if __name__ == "__main__":
    # Variance sweep
    input_stds = [1e10]  
    temperatures = [1e-5, 0.0001, 0.001, 0.01, 0.1, 1]
    
    all_results = []
    
    for input_std in input_stds:
        print(f"\n{'='*60}")
        print(f"INPUT STD = {input_std:.1f} (variance = {input_std**2:.1f})")
        print('='*60)
        
        # Generate data
        X_train, y_train, true_weights = generate_data(n_samples=200, input_std=input_std, noise_std=0.01)
        X_test, y_test = generate_test_data(true_weights, n_samples=200, input_std=input_std, noise_std=0.01)
        
        print(f"True weights: {true_weights.numpy()}")
        
        # Compute exact posterior
        # mu_post, Sigma_post = compute_exact_posterior(X_train, y_train, noise_std=1.0)
        # exact_nll = exact_test_nll(X_test, y_test, mu_post, Sigma_post, noise_std=1.0)
        
        # print(f"Exact posterior mean: {mu_post.numpy()}")
        # print(f"Exact posterior test NLL: {exact_nll:.4f}")
        
        # Temperature sweep
        results = {'input_std': input_std, 'temperatures': [], 'test_nlls': [], 
                   'posterior_means': [], 'posterior_stds': [], 
                #    'exact_nll': exact_nll,
                   'X_train': X_train, 'X_test': X_test}
        
        for temp in temperatures:
            print(f"\nTemperature T = {temp:.2f}")
            
            model = MeanFieldVI(input_dim=2, temperature=temp)
            model.fit(X_train, y_train, n_iterations=40000, verbose=True, lr=2e-4, noise_std=0.01)
            
            test_nll = model.test_nll(X_test, y_test, n_samples=50000, noise_std=0.01)
            
            results['temperatures'].append(temp)
            results['test_nlls'].append(test_nll)
            results['posterior_means'].append(model.q_mean.detach().clone())
            results['posterior_stds'].append(torch.exp(model.q_log_std).detach().clone())
            
            print(f"Test NLL: {test_nll:.4f}, Posterior mean: {model.q_mean.detach().numpy()}")
        
        all_results.append(results)
    
    # Create plots
    n_vars = len(input_stds)
    fig = plt.figure(figsize=(16, 4 * ((n_vars + 1) // 2)))
    
    for i, res in enumerate(all_results):
        # Plot NLL vs temperature
        ax1 = plt.subplot(((n_vars + 1) // 2), 4, i * 2 + 1)
        ax1.plot(np.log(res['temperatures']), res['test_nlls'], 'o-', linewidth=2, markersize=8, label='Mean Field VI')
        # ax1.axhline(y=res['exact_nll'], color='green', linestyle='--', linewidth=2, label='Exact Posterior')
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='T=1')
        ax1.set_xlabel('Temperature', fontsize=10)
        ax1.set_ylabel('Test NLL', fontsize=10)
        ax1.set_title(f'σ_x = {res["input_std"]:.1f} (var = {res["input_std"]**2:.1f})', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # Plot data distribution
        ax2 = plt.subplot(((n_vars + 1) // 2), 4, i * 2 + 2)
        ax2.scatter(res['X_train'][:, 0].numpy(), res['X_train'][:, 1].numpy(), 
                    alpha=0.6, s=20, label='Train')
        ax2.scatter(res['X_test'][:, 0].numpy(), res['X_test'][:, 1].numpy(), 
                    alpha=0.6, s=20, label='Test')
        lim = 3 * res['input_std']
        ax2.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='x1 = x2')
        ax2.set_xlabel('x1', fontsize=10)
        ax2.set_ylabel('x2', fontsize=10)
        ax2.set_title(f'Input Space', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.set_aspect('equal')
    
    plt.tight_layout()
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
        # print(f"{res['input_std']:<12.1f} {res['input_std']**2:<12.1f} {best_temp:<12.2f} {best_nll:<12.4f} {res['exact_nll']:<12.4f}")
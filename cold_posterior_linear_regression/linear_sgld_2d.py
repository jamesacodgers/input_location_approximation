import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, MultivariateNormal
import seaborn as sns
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

class BayesianLinearRegression:
    """
    Bayesian Linear Regression with analytical solution (same as original)
    """
    
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # precision of prior
        self.beta = beta    # precision of likelihood (inverse noise variance)
        
    def analytical_posterior(self, X, y):
        """Compute the analytical posterior for Bayesian linear regression"""
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Posterior covariance
        S_n_inv = self.alpha * torch.eye(X.shape[1]) + self.beta * X.T @ X
        S_n = torch.inverse(S_n_inv)
        
        # Posterior mean
        mu_n = self.beta * S_n @ X.T @ y
        
        return mu_n, S_n

class SGLDSampler:
    """
    Stochastic Gradient Langevin Dynamics sampler for Bayesian Linear Regression
    Supports cold and tempered posteriors with correct definitions:
    - Cold: scales both likelihood and prior by 1/T
    - Tempered: scales only likelihood by 1/λ, prior unchanged
    """
    
    def __init__(self, n_features, alpha=1.0, beta=1.0, temperature=1.0, posterior_type="standard"):
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.posterior_type = posterior_type
        
        # Initialize weights
        self.weights = torch.randn(n_features, requires_grad=True)
        
    def log_prior(self, weights):
        """
        Log prior: p(w) = N(0, alpha^-1 * I)
        For cold posteriors: prior is also scaled by 1/T (equivalent to changing prior precision)
        """
        if self.posterior_type == "cold":
            # Cold posterior: scale prior by 1/T (equivalent to alpha_cold = alpha/T)
            alpha_effective = self.alpha / self.temperature
        else:
            # Standard or tempered: use original prior
            alpha_effective = self.alpha
            
        # log p(w) = -0.5 * alpha * ||w||^2 + const
        return -0.5 * alpha_effective * torch.sum(weights**2)
    
    def log_likelihood(self, weights, X, y):
        """
        Log likelihood: p(y|X,w) = N(X@w, beta^-1 * I)
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        y_pred = X @ weights
        # log p(y|X,w) = -0.5 * beta * ||y - X@w||^2 + const
        return -0.5 * self.beta * torch.sum((y - y_pred)**2)
    
    def log_posterior(self, weights, X, y):
        """
        Log posterior with temperature scaling:
        
        Standard: log p(w|X,y) = log p(y|X,w) + log p(w)
        Cold: log p(w|X,y) = (1/T) * log p(y|X,w) + (1/T) * log p(w)
        Tempered: log p(w|X,y) = (1/λ) * log p(y|X,w) + log p(w)
        """
        log_lik = self.log_likelihood(weights, X, y)
        log_pri = self.log_prior(weights)
        
        if self.posterior_type == "cold":
            # Cold posterior: scale both likelihood and prior by 1/T
            return (1.0 / self.temperature) * log_lik + log_pri  # prior scaling handled in log_prior
        elif self.posterior_type == "tempered":
            # Tempered posterior: scale only likelihood by 1/λ
            return (1.0 / self.temperature) * log_lik + log_pri
        else:
            # Standard posterior: no scaling
            return log_lik + log_pri
    
    def sample(self, X, y, n_samples=5000, step_size=0.001, burn_in=1000, thin=5):
        """
        SGLD sampling with Langevin dynamics:
        θ_{t+1} = θ_t + ε/2 * ∇log p(θ|D) + √ε * N(0,I)
        """
        # Storage for samples
        samples = []
        log_posteriors = []
        
        # Initialize weights if not already done
        weights = self.weights.clone().detach().requires_grad_(True)
        
        print(f"\nStarting SGLD sampling for {self.posterior_type} posterior (T={self.temperature})")
        print(f"Initial weights: {weights.detach().numpy()}")
        print(f"Step size: {step_size}")
        
        # Check initial log posterior
        initial_log_post = self.log_posterior(weights, X, y)
        print(f"Initial log posterior: {initial_log_post.item()}")
        
        pbar = tqdm(range(n_samples + burn_in), desc=f"SGLD {self.posterior_type.capitalize()}")
        
        for i in pbar:
            # Zero gradients
            if weights.grad is not None:
                weights.grad.zero_()
            
            # Compute log posterior and gradients
            log_post = self.log_posterior(weights, X, y)
            
            # Check for NaN in log posterior
            if torch.isnan(log_post):
                print(f"\nERROR: NaN in log posterior at iteration {i}")
                print(f"Weights: {weights.detach().numpy()}")
                print(f"Log likelihood: {self.log_likelihood(weights, X, y).item()}")
                print(f"Log prior: {self.log_prior(weights).item()}")
                break
            
            log_post.backward()
            
            # Check for NaN in gradients
            if torch.isnan(weights.grad).any():
                print(f"\nERROR: NaN in gradients at iteration {i}")
                print(f"Weights: {weights.detach().numpy()}")
                print(f"Gradients: {weights.grad.detach().numpy()}")
                break
            
            with torch.no_grad():
                # Check for exploding gradients
                grad_norm = torch.norm(weights.grad)
                if grad_norm > 1000:
                    print(f"\nWARNING: Large gradient norm at iteration {i}: {grad_norm:.2f}")
                    print(f"Weights: {weights.detach().numpy()}")
                    print(f"Gradients: {weights.grad.detach().numpy()}")
                
                # Langevin dynamics update
                # θ_{t+1} = θ_t + ε/2 * ∇log p(θ|D) + √ε * N(0,I)
                noise = torch.randn_like(weights) * np.sqrt(step_size)
                weights += step_size / 2.0 * weights.grad + noise
                
                # Check for NaN in weights after update
                if torch.isnan(weights).any():
                    print(f"\nERROR: NaN in weights after update at iteration {i}")
                    print(f"Previous weights: {(weights - step_size / 2.0 * weights.grad - noise).detach().numpy()}")
                    print(f"Gradient step: {(step_size / 2.0 * weights.grad).detach().numpy()}")
                    print(f"Noise: {noise.detach().numpy()}")
                    break
                
                # Store samples after burn-in and with thinning
                if i >= burn_in and (i - burn_in) % thin == 0:
                    samples.append(weights.clone())
                    log_posteriors.append(log_post.item())
                
                # Update progress bar
                if i % 100 == 0:
                    pbar.set_postfix({'Log Post': f'{log_post.item():.2f}', 'Grad Norm': f'{grad_norm:.2f}'})
            
            # Refresh gradients for next iteration
            weights = weights.detach().requires_grad_(True)
        
        print(f"\nCompleted sampling. Collected {len(samples)} samples")
        
        if len(samples) == 0:
            print("ERROR: No samples collected!")
            return None, None
        
        # Convert to tensors
        samples = torch.stack(samples)
        
        # Check for NaN in final samples
        nan_mask = torch.isnan(samples)
        if nan_mask.any():
            print(f"WARNING: Found {nan_mask.sum()} NaN values in samples")
            print(f"Sample shape: {samples.shape}")
            # Remove NaN samples
            valid_samples = samples[~torch.isnan(samples).any(dim=1)]
            if len(valid_samples) > 0:
                samples = valid_samples
                print(f"Using {len(samples)} valid samples after removing NaNs")
            else:
                print("ERROR: All samples contain NaN!")
                return None, None
        
        # Store final samples in the sampler
        self.samples = samples
        self.log_posteriors = log_posteriors
        
        # Compute sample statistics
        self.sample_mean = torch.mean(samples, dim=0)
        if len(samples) > 1:
            self.sample_cov = torch.cov(samples.T)
        else:
            self.sample_cov = torch.eye(samples.shape[1])
        
        print(f"Final sample mean: {self.sample_mean.detach().numpy()}")
        print(f"Sample std: {torch.sqrt(torch.diag(self.sample_cov)).detach().numpy()}")
        
        return samples, log_posteriors

def generate_synthetic_data(n_samples=100, n_features=3, noise_std=0.1, seed=42):
    """Generate synthetic regression data with correlated features (same as original)"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # True weights
    w_true = torch.randn(n_features)
    
    # Generate correlated features using multivariate normal
    correlation = 0.8
    cov_matrix = np.array([[1.0, correlation],
                          [correlation, 1.0]])
    
    # Generate correlated features (excluding bias)
    features_corr = np.random.multivariate_normal(
        mean=[0, 0], 
        cov=cov_matrix, 
        size=n_samples
    )
    
    # Create design matrix with bias term
    X = np.column_stack([np.ones(n_samples), features_corr])
    X = torch.tensor(X, dtype=torch.float32)
    
    # Targets with noise
    y = X @ w_true + noise_std * torch.randn(n_samples)
    
    print(f"Generated data with feature correlation: {correlation}")
    print(f"Actual correlation: {np.corrcoef(features_corr.T)[0,1]:.3f}")
    
    return X.numpy(), y.numpy(), w_true.numpy()

def fit_sgld_regression(X, y, alpha=1.0, beta=100.0, temperature=1.0, posterior_type="standard", 
                       n_samples=5000, step_size=0.001, burn_in=1000, thin=5, model_name="SGLD"):
    """Fit Bayesian linear regression using SGLD"""
    
    n_features = X.shape[1]
    sampler = SGLDSampler(n_features, alpha=alpha, beta=beta, 
                         temperature=temperature, posterior_type=posterior_type)
    
    print(f"\n{'='*60}")
    print(f"Fitting {model_name}")
    print(f"Data shape: {X.shape}")
    print(f"Alpha (prior precision): {alpha}")
    print(f"Beta (likelihood precision): {beta}")
    print(f"Temperature: {temperature}")
    print(f"Posterior type: {posterior_type}")
    print(f"{'='*60}")
    
    samples, log_posteriors = sampler.sample(X, y, n_samples=n_samples, 
                                           step_size=step_size, burn_in=burn_in, thin=thin)
    
    # Additional NaN checking
    if samples is not None:
        nan_count = torch.isnan(samples).sum()
        if nan_count > 0:
            print(f"WARNING: Found {nan_count} NaN values in final samples!")
        else:
            print(f"✓ All {len(samples)} samples are valid (no NaNs)")
    else:
        print("ERROR: Sampling failed, no samples returned!")
    
    return sampler, samples, log_posteriors

def compare_posteriors_sgld(X, y, w_true, alpha=1.0, beta=100.0):
    """Compare analytical, standard, cold, and tempered SGLD posteriors"""
    
    # Analytical posterior
    analytical = BayesianLinearRegression(alpha=alpha, beta=beta)
    mu_true, Sigma_true = analytical.analytical_posterior(X, y)
    
    # SGLD parameters - increased samples for better statistics
    sgld_params = {
        'n_samples': 20000,  # Increased from 8000
        'step_size': 0.0001,  # Much smaller step size for numerical stability
        'burn_in': 5000,      # Increased burn-in
        'thin': 2             # Reduced thinning to keep more samples
    }
    
    # Standard SGLD (no temperature scaling)
    sgld_standard, samples_standard, log_post_standard = fit_sgld_regression(
        X, y, alpha=alpha, beta=beta, temperature=1.0, 
        posterior_type="standard", model_name="Standard SGLD", **sgld_params
    )
    
    # Cold posterior SGLD (T=0.8: scales both likelihood and prior by 1/T)
    sgld_cold, samples_cold, log_post_cold = fit_sgld_regression(
        X, y, alpha=alpha, beta=beta, temperature=0.8, 
        posterior_type="cold", model_name="Cold SGLD (T=0.8)", **sgld_params
    )
    
    # Tempered posterior SGLD (λ=0.8: scales only likelihood by 1/λ)
    sgld_tempered, samples_tempered, log_post_tempered = fit_sgld_regression(
        X, y, alpha=alpha, beta=beta, temperature=0.8, 
        posterior_type="tempered", model_name="Tempered SGLD (λ=0.8)", **sgld_params
    )
    
    samplers = {
        'Standard': sgld_standard,
        'Cold': sgld_cold, 
        'Tempered': sgld_tempered
    }
    
    all_samples = {
        'Standard': samples_standard,
        'Cold': samples_cold,
        'Tempered': samples_tempered
    }
    
    # Check if any sampling failed
    failed_models = []
    for model_name, samples in all_samples.items():
        if samples is None:
            failed_models.append(model_name)
            
    if failed_models:
        print(f"\nERROR: Sampling failed for models: {failed_models}")
        print("Try reducing step_size or checking data scaling")
        return None, None, None, None, None
    
    log_posteriors = {
        'Standard': log_post_standard,
        'Cold': log_post_cold,
        'Tempered': log_post_tempered
    }
    
    print("\n" + "="*80)
    print("POSTERIOR COMPARISON - STANDARD vs COLD vs TEMPERED (SGLD)")
    print("="*80)
    print("DEFINITIONS:")
    print("  Standard:  log p(w|D) = log p(y|X,w) + log p(w)")
    print("  Cold:      log p(w|D) = (1/T) * log p(y|X,w) + (1/T) * log p(w)")
    print("  Tempered:  log p(w|D) = (1/λ) * log p(y|X,w) + log p(w)")
    print("="*80)
    print(f"{'Parameter':<12} {'True':<10} {'Analytical':<12} {'Standard SGLD':<14} {'Cold SGLD':<14} {'Tempered SGLD':<14}")
    print(f"{'Name':<12} {'Value':<10} {'Mean±Std':<12} {'Mean±Std':<14} {'Mean±Std':<14} {'Mean±Std':<14}")
    print("-" * 88)
    
    param_names = ['Bias', 'Weight 1', 'Weight 2']
    sigma_true = np.sqrt(np.diag(Sigma_true.numpy()))
    
    for i in range(len(mu_true)):
        row = f"{param_names[i]:<12} {w_true[i]:<10.4f} {mu_true[i].item():.3f}±{sigma_true[i]:.3f}   "
        
        for sampler_name in ['Standard', 'Cold', 'Tempered']:
            sampler = samplers[sampler_name]
            mu_sgld = sampler.sample_mean[i].item()
            sigma_sgld = torch.sqrt(sampler.sample_cov[i, i]).item()
            row += f"{mu_sgld:.3f}±{sigma_sgld:.3f}     "
        
        print(row)
    
    print("="*80)
    
    # Print SGLD diagnostics
    print("\nSGLD DIAGNOSTICS:")
    print("="*50)
    for sampler_name, sampler in samplers.items():
        n_eff_samples = len(sampler.samples)
        acceptance_rate = "N/A"  # SGLD doesn't have acceptance/rejection
        
        # ESS estimation using autocorrelation (simplified)
        samples_np = sampler.samples.numpy()
        autocorr_lags = min(100, len(samples_np) // 4)
        
        print(f"{sampler_name} SGLD:")
        print(f"  Effective samples: {n_eff_samples}")
        print(f"  Final log posterior: {sampler.log_posteriors[-1]:.2f}")
        print(f"  Sample std (bias): {torch.sqrt(sampler.sample_cov[0,0]):.4f}")
    
    return mu_true, Sigma_true, samplers, all_samples, log_posteriors

def plot_sgld_results(X, y, w_true, mu_true, Sigma_true, samplers, all_samples, log_posteriors):
    """Plot comparison results for SGLD models"""
    
    n_params = len(w_true)
    
    # Create output directory
    output_dir = "figs/sgld_cold_posterior_linear_regression"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving figures to: {output_dir}")
    
    # Extract features (skip bias column)
    X_features = X[:, 1:]
    
    # 1. Interactive 3D plot of posterior samples - increased sample counts
    if n_params >= 3:
        # Sample from analytical posterior - more samples
        n_viz_samples = 2000  # Increased from 800
        analytical_dist = MultivariateNormal(mu_true, Sigma_true)
        analytical_samples = analytical_dist.sample((n_viz_samples,))
        
        # Create interactive 3D scatter plot
        fig = go.Figure()
        
        # Add analytical samples
        fig.add_trace(go.Scatter3d(
            x=analytical_samples[:, 0].numpy(),
            y=analytical_samples[:, 1].numpy(), 
            z=analytical_samples[:, 2].numpy(),
            mode='markers',
            marker=dict(size=2, color='blue', opacity=0.4),  # Smaller, more transparent
            name='Analytical Posterior',
            hovertemplate='<b>Analytical</b><br>Bias: %{x:.3f}<br>Weight 1: %{y:.3f}<br>Weight 2: %{z:.3f}<extra></extra>'
        ))
        
        # Add SGLD samples for each model
        colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
        for sampler_name, samples in all_samples.items():
            sampler = samplers[sampler_name]
            
            # Use more samples for visualization - subsample from larger pool
            n_total = len(samples)
            n_viz = min(2000, n_total)  # Increased visualization samples
            indices = np.random.choice(n_total, n_viz, replace=False)
            viz_samples = samples[indices]
            
            print(f"Plotting {len(viz_samples)} {sampler_name} SGLD samples (from {n_total} total)")
            
            # Create label with correct notation
            if sampler.posterior_type == "cold":
                label = f'{sampler_name} (Cold T={sampler.temperature})'
            elif sampler.posterior_type == "tempered":
                label = f'{sampler_name} (Tempered λ={sampler.temperature})'
            else:
                label = f'{sampler_name} (Standard)'
                
            fig.add_trace(go.Scatter3d(
                x=viz_samples[:, 0].detach().numpy(),
                y=viz_samples[:, 1].detach().numpy(),
                z=viz_samples[:, 2].detach().numpy(),
                mode='markers',
                marker=dict(size=2, color=colors[sampler_name], opacity=0.4),  # Smaller, more transparent
                name=label,
                hovertemplate=f'<b>{sampler_name} SGLD</b><br>Bias: %{{x:.3f}}<br>Weight 1: %{{y:.3f}}<br>Weight 2: %{{z:.3f}}<extra></extra>'
            ))
        
        # Add true values
        fig.add_trace(go.Scatter3d(
            x=[w_true[0]], y=[w_true[1]], z=[w_true[2]],
            mode='markers',
            marker=dict(size=12, color='gold', symbol='diamond', line=dict(color='black', width=2)),
            name='True Values',
            hovertemplate='<b>True Values</b><br>Bias: %{x:.3f}<br>Weight 1: %{y:.3f}<br>Weight 2: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='3D Posterior Samples: Analytical vs SGLD (Standard/Cold/Tempered)<br><sub>Cold: scales likelihood & prior by 1/T | Tempered: scales only likelihood by 1/λ | 2000 samples each</sub>',
            scene=dict(
                xaxis_title='Bias (w[0])',
                yaxis_title='Weight 1 (w[1])',
                zaxis_title='Weight 2 (w[2])',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1000, height=800, margin=dict(l=0, r=0, b=0, t=60)
        )
        fig.show()
        
        # Save interactive plot
        fig.write_html(f"{output_dir}/3d_posterior_samples_sgld.html")
        print(f"Saved: {output_dir}/3d_posterior_samples_sgld.html")
    
    # 2. SGLD trace plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    param_names = ['Bias', 'Weight 1', 'Weight 2']
    colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
    
    for i, param_name in enumerate(param_names):
        for j, (sampler_name, samples) in enumerate(all_samples.items()):
            sampler = samplers[sampler_name]
            
            # Trace plot - samples is already a tensor
            sample_values = samples[:, i].detach().numpy()
            axes[i, j].plot(sample_values, color=colors[sampler_name], alpha=0.7, linewidth=0.8)
            axes[i, j].axhline(y=w_true[i], color='gold', linestyle='--', linewidth=2, label='True Value')
            axes[i, j].axhline(y=mu_true[i].item(), color='blue', linestyle='--', linewidth=2, label='Analytical Mean')
            
            if sampler.posterior_type == "cold":
                title = f'{sampler_name} (Cold T={sampler.temperature})'
            elif sampler.posterior_type == "tempered":
                title = f'{sampler_name} (Tempered λ={sampler.temperature})'
            else:
                title = f'{sampler_name} (Standard)'
            
            axes[i, j].set_title(f'{title}\n{param_name}', fontsize=10, fontweight='bold')
            axes[i, j].grid(True, alpha=0.3)
            
            if i == 0 and j == 0:  # Only show legend once
                axes[i, j].legend(fontsize=8)
            
            if i == 2:  # Bottom row
                axes[i, j].set_xlabel('SGLD Iteration', fontsize=9)
            if j == 0:  # Left column
                axes[i, j].set_ylabel('Parameter Value', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sgld_trace_plots.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/sgld_trace_plots.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/sgld_trace_plots.png")
    plt.show()
    
    # 3. Log posterior traces
    plt.figure(figsize=(12, 6))
    
    for sampler_name, log_posts in log_posteriors.items():
        sampler = samplers[sampler_name]
        
        if sampler.posterior_type == "cold":
            label = f'{sampler_name} (Cold T={sampler.temperature})'
        elif sampler.posterior_type == "tempered":
            label = f'{sampler_name} (Tempered λ={sampler.temperature})'
        else:
            label = f'{sampler_name} (Standard)'
        
        plt.plot(log_posts, linewidth=2, color=colors[sampler_name], 
                label=label, alpha=0.8)
    
    plt.title('SGLD Log Posterior Traces: Standard vs Cold vs Tempered', fontsize=14, fontweight='bold')
    plt.xlabel('SGLD Iteration (after burn-in)', fontsize=12)
    plt.ylabel('Log Posterior', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/sgld_log_posterior_traces.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/sgld_log_posterior_traces.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/sgld_log_posterior_traces.png")
    plt.show()
    
    # 4. Sample autocorrelation analysis
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (sampler_name, samples) in enumerate(all_samples.items()):
        # Compute autocorrelation for first parameter (bias)
        param_samples = samples[:, 0].detach().numpy()  # Bias parameter
        n_lags = min(200, len(param_samples) // 4)
        
        autocorr = np.correlate(param_samples - np.mean(param_samples), 
                              param_samples - np.mean(param_samples), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        autocorr = autocorr[:n_lags]
        
        sampler = samplers[sampler_name]
        
        if sampler.posterior_type == "cold":
            label = f'{sampler_name} (Cold T={sampler.temperature})'
        elif sampler.posterior_type == "tempered":
            label = f'{sampler_name} (Tempered λ={sampler.temperature})'
        else:
            label = f'{sampler_name} (Standard)'
        
        axes[i].plot(autocorr, color=colors[sampler_name], linewidth=2)
        axes[i].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[i].axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='10% threshold')
        axes[i].set_title(f'Autocorrelation: {label}', fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Lag', fontsize=10)
        axes[i].set_ylabel('Autocorrelation', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sgld_autocorrelation.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/sgld_autocorrelation.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/sgld_autocorrelation.png")
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*70}")
    print("SGLD SUMMARY STATISTICS")
    print(f"{'='*70}")
    
    for sampler_name, sampler in samplers.items():
        samples = all_samples[sampler_name]
        
        # Compute effective sample size (simplified)
        param_samples = samples[:, 0].numpy()
        autocorr = np.correlate(param_samples - np.mean(param_samples), 
                              param_samples - np.mean(param_samples), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        # Find where autocorr drops below 0.1
        below_threshold = np.where(autocorr < 0.1)[0]
        tau = below_threshold[0] if len(below_threshold) > 0 else len(autocorr)
        ess = len(param_samples) / (2 * tau + 1)
        
        print(f"{sampler_name} SGLD:")
        print(f"  Total samples: {len(samples)}")
        print(f"  Autocorr. time (τ): {tau}")
        print(f"  ESS (approx): {ess:.1f}")
        print(f"  Final log post: {sampler.log_posteriors[-1]:.2f}")
        print(f"  Sample mean: [{', '.join([f'{x:.3f}' for x in sampler.sample_mean])}]")
        print()
    
    print(f"{'='*70}")
    
    # 5. Test data KL analysis with PCA visualization (same as VI version)
    print(f"\n{'='*60}")
    print("GENERATING TEST DATA FOR KL ANALYSIS (SGLD)")
    print(f"{'='*60}")
    
    # Generate test data with same correlation structure - more test points
    n_test = 500  # Increased from 200
    test_features_corr = np.random.multivariate_normal(
        mean=[0, 0], 
        cov=np.array([[1.0, 0.8], [0.8, 1.0]]), 
        size=n_test
    )
    X_test = np.column_stack([np.ones(n_test), test_features_corr])
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Perform PCA on training data features (excluding bias)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Fit PCA on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_features)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # Transform test data to PCA space
    X_test_scaled = scaler.transform(test_features_corr)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Generated {n_test} test points")
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Compute KL divergences for all SGLD models
    all_test_kl_sgld = {}
    colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
    
    for sampler_name, sampler in samplers.items():
        print(f"Computing test KL divergences for {sampler_name} SGLD...")
        
        test_kl_divergences = np.zeros(n_test)
        
        # Get noise precision (same as in VI)
        beta = sampler.beta
        
        for i in tqdm(range(n_test), desc=f"Test KL for {sampler_name}"):
            x_point = X_test_tensor[i:i+1]
            
            # True posterior predictive
            pred_mean_true = (x_point @ mu_true).item()
            pred_var_true = (x_point @ Sigma_true @ x_point.T).item() + 1.0/beta
            
            # SGLD posterior predictive (using sample statistics)
            pred_mean_sgld = (x_point @ sampler.sample_mean).item()
            
            # For SGLD, compute predictive variance using sample covariance
            pred_var_sgld = (x_point @ sampler.sample_cov @ x_point.T).item() + 1.0/beta
            
            # KL divergence between two Gaussians
            mean_diff_sq = (pred_mean_true - pred_mean_sgld)**2
            kl_div = 0.5 * (
                np.log(pred_var_sgld / pred_var_true) +
                pred_var_true / pred_var_sgld +
                mean_diff_sq / pred_var_sgld - 1
            )
            test_kl_divergences[i] = kl_div
        
        all_test_kl_sgld[sampler_name] = test_kl_divergences
        
        # Print test statistics
        print(f"\n{sampler_name} SGLD Test KL Statistics:")
        print(f"  Mean: {test_kl_divergences.mean():.6f}")
        print(f"  Max:  {test_kl_divergences.max():.6f}")
        print(f"  Min:  {test_kl_divergences.min():.6f}")
        print(f"  Std:  {test_kl_divergences.std():.6f}")
    
    # Create two plots: PC1 vs KL and PC2 vs KL
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: PC1 vs KL Divergence
    for sampler_name, test_kl in all_test_kl_sgld.items():
        sampler = samplers[sampler_name]
        
        # Create proper labels
        if sampler.posterior_type == "cold":
            label = f'{sampler_name} (Cold T={sampler.temperature})'
        elif sampler.posterior_type == "tempered":
            label = f'{sampler_name} (Tempered λ={sampler.temperature})'
        else:
            label = f'{sampler_name} (Standard)'
        
        axes[0].scatter(X_test_pca[:, 0], test_kl, 
                       c=colors[sampler_name], s=25, alpha=0.7, 
                       label=label, edgecolor='black', linewidth=0.3)
    
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('KL[True || SGLD] Predictive', fontsize=12)
    axes[0].set_title('Test KL Divergence vs PC1 (SGLD)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: PC2 vs KL Divergence
    for sampler_name, test_kl in all_test_kl_sgld.items():
        sampler = samplers[sampler_name]
        
        # Create proper labels
        if sampler.posterior_type == "cold":
            label = f'{sampler_name} (Cold T={sampler.temperature})'
        elif sampler.posterior_type == "tempered":
            label = f'{sampler_name} (Tempered λ={sampler.temperature})'
        else:
            label = f'{sampler_name} (Standard)'
        
        axes[1].scatter(X_test_pca[:, 1], test_kl, 
                       c=colors[sampler_name], s=25, alpha=0.7, 
                       label=label, edgecolor='black', linewidth=0.3)
    
    axes[1].set_xlabel('PC2', fontsize=12)
    axes[1].set_ylabel('KL[True || SGLD] Predictive', fontsize=12)
    axes[1].set_title('Test KL Divergence vs PC2 (SGLD)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('SGLD: KL Divergence vs Principal Components\n(Lower KL = Better Approximation)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/sgld_kl_vs_pca_components.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/sgld_kl_vs_pca_components.pdf", bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print("PCA INTERPRETATION (SGLD)")
    print(f"{'='*60}")
    print(f"PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    print(f"PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
    print(f"Total explained: {sum(pca.explained_variance_ratio_)*100:.1f}%")
    print(f"\nPCA Loadings (how original features contribute to PCs):")
    print(f"{'Component':<12} {'Feature 1':<12} {'Feature 2':<12}")
    print("-" * 40)
    for i, pc in enumerate(['PC1', 'PC2']):
        print(f"{pc:<12} {pca.components_[i,0]:<12.3f} {pca.components_[i,1]:<12.3f}")
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("SGLD MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    overall_stats_sgld = {}
    for sampler_name, test_kl in all_test_kl_sgld.items():
        overall_stats_sgld[sampler_name] = {
            'mean_kl': test_kl.mean(),
            'median_kl': np.median(test_kl),
            'std_kl': test_kl.std()
        }
    
    print(f"{'Model':<15} {'Mean KL':<12} {'Median KL':<12} {'Std KL':<12}")
    print("-" * 55)
    for sampler_name, stats in overall_stats_sgld.items():
        print(f"{sampler_name:<15} {stats['mean_kl']:<12.6f} {stats['median_kl']:<12.6f} {stats['std_kl']:<12.6f}")
    
    print(f"{'='*60}")
    
    # 6. Input data visualization with KL divergence backgrounds (SGLD version)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create grid for KL divergence computation - higher resolution
    x1_min, x1_max = X_features[:, 0].min() - 0.5, X_features[:, 0].max() + 0.5
    x2_min, x2_max = X_features[:, 1].min() - 0.5, X_features[:, 1].max() + 0.5
    grid_resolution = 60  # Increased from 40 for higher resolution
    x1_grid = np.linspace(x1_min, x1_max, grid_resolution)
    x2_grid = np.linspace(x2_min, x2_max, grid_resolution)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    n_grid_points = len(grid_points)
    X_grid = np.column_stack([np.ones(n_grid_points), grid_points])
    X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)
    
    print("Computing KL divergences for all SGLD models...")
    
    # First pass: compute all KL divergences to determine global min/max
    all_kl_divergences_sgld = []
    
    for idx, (sampler_name, sampler) in enumerate(samplers.items()):
        print(f"Computing KL divergences for {sampler_name} SGLD...")
        
        kl_divergences = np.zeros(n_grid_points)
        
        for i in tqdm(range(n_grid_points), desc=f"KL for {sampler_name}"):
            x_point = X_grid_tensor[i:i+1]
            
            # True posterior predictive
            pred_mean_true = (x_point @ mu_true).item()
            pred_var_true = (x_point @ Sigma_true @ x_point.T).item() + 1.0/sampler.beta
            
            # SGLD posterior predictive
            pred_mean_sgld = (x_point @ sampler.sample_mean).item()
            pred_var_sgld = (x_point @ sampler.sample_cov @ x_point.T).item() + 1.0/sampler.beta
            
            # KL divergence
            mean_diff_sq = (pred_mean_true - pred_mean_sgld)**2
            kl_div = 0.5 * (
                np.log(pred_var_sgld / pred_var_true) +
                pred_var_true / pred_var_sgld +
                mean_diff_sq / pred_var_sgld - 1
            )
            kl_divergences[i] = kl_div
        
        all_kl_divergences_sgld.append(kl_divergences)
    
    # Determine global min and max for consistent color scaling
    global_kl_min = min([kl.min() for kl in all_kl_divergences_sgld])
    global_kl_max = max([kl.max() for kl in all_kl_divergences_sgld])
    
    print(f"\nGlobal SGLD KL range: [{global_kl_min:.6f}, {global_kl_max:.6f}]")
    
    # Second pass: create plots with consistent color scale
    for idx, (sampler_name, sampler) in enumerate(samplers.items()):
        kl_divergences = all_kl_divergences_sgld[idx]
        
        # Plot for this model with consistent color scale
        KL_grid = kl_divergences.reshape(X1_grid.shape)
        im = axes[idx].imshow(KL_grid, extent=[x1_min, x1_max, x2_min, x2_max], 
                            origin='lower', cmap='Reds', alpha=0.7, aspect='auto',
                            vmin=global_kl_min, vmax=global_kl_max)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], shrink=0.8)
        cbar.set_label('KL[True || SGLD]', fontsize=10)
        
        # Overlay data points
        scatter = axes[idx].scatter(X_features[:, 0], X_features[:, 1], c=y, s=30, 
                                  cmap='viridis', alpha=0.9, edgecolor='white', linewidth=0.8)
        
        axes[idx].set_xlabel('Feature 1', fontsize=11)
        axes[idx].set_ylabel('Feature 2', fontsize=11)
        axes[idx].set_title(f'{sampler_name} SGLD\n{sampler.posterior_type.title()} ({"T" if sampler.posterior_type == "cold" else "λ"}={sampler.temperature})', 
                          fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Print statistics
        print(f"\n{sampler_name} SGLD KL Statistics:")
        print(f"  Mean: {kl_divergences.mean():.6f}")
        print(f"  Max:  {kl_divergences.max():.6f}")
        print(f"  Std:  {kl_divergences.std():.6f}")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sgld_kl_divergence_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/sgld_kl_divergence_heatmaps.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/sgld_kl_divergence_heatmaps.png")
    print(f"Saved: {output_dir}/sgld_kl_divergence_heatmaps.pdf")
    plt.show()
    
    print(f"\nAll SGLD figures saved to: {output_dir}")
    print(f"Files saved:")
    print(f"  - 3d_posterior_samples_sgld.html (interactive)")
    print(f"  - sgld_trace_plots.png/.pdf")
    print(f"  - sgld_log_posterior_traces.png/.pdf")
    print(f"  - sgld_autocorrelation.png/.pdf")
    print(f"  - sgld_kl_vs_pca_components.png/.pdf")
    print(f"  - sgld_kl_divergence_heatmaps.png/.pdf")

def main():
    """Main execution function for SGLD comparison"""
    print("Bayesian Linear Regression: SGLD Standard vs Cold vs Tempered")
    print("="*70)
    
    # Generate synthetic data with correlated features (same as VI version)
    X, y, w_true = generate_synthetic_data(n_samples=100, n_features=3, noise_std=0.1)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"True weights: {w_true}")
    
    # Hyperparameters (same as VI version)
    alpha = 1.0   # Prior precision
    beta = 100.0  # Likelihood precision (1/noise_variance)
    
    # Compare all posteriors using SGLD
    mu_true, Sigma_true, samplers, all_samples, log_posteriors = compare_posteriors_sgld(
        X, y, w_true, alpha=alpha, beta=beta
    )
    
    # Plot SGLD results
    plot_sgld_results(X, y, w_true, mu_true, Sigma_true, samplers, all_samples, log_posteriors)
    
    # Predictive performance comparison
    print(f"\nPREDICTIVE PERFORMANCE COMPARISON (SGLD):")
    print(f"{'='*55}")
    X_test = torch.tensor(X[:10], dtype=torch.float32)  # Use first 10 samples as test
    y_test = torch.tensor(y[:10], dtype=torch.float32)
    
    # Analytical predictions
    y_pred_analytical = X_test @ mu_true
    mse_analytical = torch.mean((y_test - y_pred_analytical)**2)
    
    print(f"{'Model':<20} {'MSE':<12} {'Temperature':<12}")
    print("-" * 45)
    print(f"{'Analytical':<20} {mse_analytical:.6f} {'N/A':<12}")
    
    # SGLD predictions for each model
    for sampler_name, sampler in samplers.items():
        y_pred_sgld = X_test @ sampler.sample_mean
        mse_sgld = torch.mean((y_test - y_pred_sgld)**2)
        print(f"{sampler_name + ' SGLD':<20} {mse_sgld:.6f} {sampler.temperature:<12}")
    
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
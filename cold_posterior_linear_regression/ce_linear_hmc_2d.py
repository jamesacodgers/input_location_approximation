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

# For HMC sampling
import pyro
import pyro.distributions as dist
from pyro.infer import HMC, MCMC

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

class HMCSampler:
    """
    Hamiltonian Monte Carlo sampler for Bayesian Linear Regression using Pyro
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
        
        # Store data for model
        self.X = None
        self.y = None
        
    def model(self):
        """
        Pyro model for Bayesian linear regression with temperature scaling
        """
        # Prior precision adjustment for cold posteriors
        if self.posterior_type == "cold":
            # Cold posterior: scale prior by 1/T (equivalent to alpha_cold = alpha/T)
            alpha_effective = self.alpha / self.temperature
        else:
            # Standard or tempered: use original prior
            alpha_effective = self.alpha
        
        # Prior: w ~ N(0, alpha_effective^-1 * I)
        weights = pyro.sample("weights", 
                             dist.MultivariateNormal(
                                 torch.zeros(self.n_features),
                                 torch.eye(self.n_features) / alpha_effective
                             ))
        
        # Likelihood scaling for temperature
        if self.posterior_type == "cold":
            # Cold posterior: scale likelihood by 1/T
            beta_effective = self.beta / self.temperature
        elif self.posterior_type == "tempered":
            # Tempered posterior: scale likelihood by 1/λ
            beta_effective = self.beta / self.temperature
        else:
            # Standard posterior: no scaling
            beta_effective = self.beta
        
        # Likelihood: y|X,w ~ N(X@w, beta_effective^-1 * I)
        mean = self.X @ weights
        sigma = 1.0/torch.sqrt(torch.tensor(beta_effective))
        with pyro.plate("data", len(self.y)):
            pyro.sample("obs", dist.Normal(mean, sigma), obs=self.y)
        
        return weights
    
    def sample(self, X, y, n_samples=3000, n_warmup=1000, step_size=0.01, num_steps=10):
        """
        HMC sampling using Pyro
        """
        # Store data
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        print(f"\nStarting HMC sampling for {self.posterior_type} posterior (T={self.temperature})")
        print(f"Samples: {n_samples}, Warmup: {n_warmup}")
        
        # Set up HMC kernel
        kernel = HMC(self.model, 
                     step_size=step_size, 
                     num_steps=num_steps,
                     adapt_step_size=True,
                     adapt_mass_matrix=True)
        
        # Set up MCMC
        mcmc = MCMC(kernel, 
                    num_samples=n_samples, 
                    warmup_steps=n_warmup,
                    disable_progbar=False)
        
        # Run sampling
        try:
            mcmc.run()
            
            # Get samples
            samples_dict = mcmc.get_samples()
            samples = samples_dict["weights"]  # Shape: [n_samples, n_features]
            
            print(f"Completed HMC sampling. Collected {len(samples)} samples")
            
            # Check for NaN in samples
            nan_mask = torch.isnan(samples)
            if nan_mask.any():
                print(f"WARNING: Found {nan_mask.sum()} NaN values in samples")
                valid_samples = samples[~torch.isnan(samples).any(dim=1)]
                if len(valid_samples) > 0:
                    samples = valid_samples
                    print(f"Using {len(samples)} valid samples after removing NaNs")
                else:
                    print("ERROR: All samples contain NaN!")
                    return None, None
            
            # Store samples and compute statistics
            self.samples = samples
            self.sample_mean = torch.mean(samples, dim=0)
            if len(samples) > 1:
                self.sample_cov = torch.cov(samples.T)
            else:
                self.sample_cov = torch.eye(samples.shape[1])
            
            # Get diagnostics
            self.mcmc = mcmc
            try:
                self.diagnostics = mcmc.diagnostics()
                print(f"HMC Diagnostics available")
            except:
                print("Diagnostics not available")
                self.diagnostics = {}
            
            print(f"Final sample mean: {self.sample_mean.detach().numpy()}")
            print(f"Sample std: {torch.sqrt(torch.diag(self.sample_cov)).detach().numpy()}")
            
            # Create dummy log posteriors for compatibility
            log_posteriors = [0.0] * len(samples)  # Placeholder
            
            return samples, log_posteriors
            
        except Exception as e:
            print(f"ERROR in HMC sampling: {e}")
            return None, None

def compute_cross_entropy(mu_true, var_true, mu_pred, var_pred):
    """
    Compute cross entropy H(p,q) = -E_p[log q] for two Gaussians
    where p ~ N(mu_true, var_true) and q ~ N(mu_pred, var_pred)
    
    For Gaussians: H(p,q) = 0.5 * [log(2π*var_pred) + (var_true + (mu_true - mu_pred)²)/var_pred]
    """
    mean_diff_sq = (mu_true - mu_pred)**2
    cross_entropy = 0.5 * (
        np.log(2 * np.pi * var_pred) + 
        (var_true + mean_diff_sq) / var_pred
    )
    return cross_entropy

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

def fit_hmc_regression(X, y, alpha=1.0, beta=100.0, temperature=1.0, posterior_type="standard", 
                      n_samples=3000, n_warmup=1000, step_size=0.01, num_steps=10, model_name="HMC"):
    """Fit Bayesian linear regression using HMC"""
    
    n_features = X.shape[1]
    sampler = HMCSampler(n_features, alpha=alpha, beta=beta, 
                        temperature=temperature, posterior_type=posterior_type)
    
    print(f"\n{'='*60}")
    print(f"Fitting {model_name}")
    print(f"Temperature: {temperature}, Type: {posterior_type}")
    print(f"{'='*60}")
    
    samples, log_posteriors = sampler.sample(X, y, n_samples=n_samples, 
                                           n_warmup=n_warmup, step_size=step_size, 
                                           num_steps=num_steps)
    
    # Check success
    if samples is not None:
        print(f"✓ All {len(samples)} samples are valid")
    else:
        print("ERROR: Sampling failed!")
    
    return sampler, samples, log_posteriors

def compare_posteriors_hmc(X, y, w_true, alpha=1.0, beta=100.0):
    """Compare analytical, standard, cold, and tempered HMC posteriors"""
    
    # Analytical posterior
    analytical = BayesianLinearRegression(alpha=alpha, beta=beta)
    mu_true, Sigma_true = analytical.analytical_posterior(X, y)
    
    # HMC parameters
    hmc_params = {
        'n_samples': 3000,
        'n_warmup': 1000,
        'step_size': 0.01,
        'num_steps': 10
    }
    
    # Standard HMC (no temperature scaling)
    hmc_standard, samples_standard, log_post_standard = fit_hmc_regression(
        X, y, alpha=alpha, beta=beta, temperature=1.0, 
        posterior_type="standard", model_name="Standard HMC", **hmc_params
    )
    
    # Cold posterior HMC (T=0.8: scales both likelihood and prior by 1/T)
    hmc_cold, samples_cold, log_post_cold = fit_hmc_regression(
        X, y, alpha=alpha, beta=beta, temperature=0.8, 
        posterior_type="cold", model_name="Cold HMC (T=0.8)", **hmc_params
    )
    
    # Tempered posterior HMC (λ=0.8: scales only likelihood by 1/λ)
    hmc_tempered, samples_tempered, log_post_tempered = fit_hmc_regression(
        X, y, alpha=alpha, beta=beta, temperature=0.8, 
        posterior_type="tempered", model_name="Tempered HMC (λ=0.8)", **hmc_params
    )
    
    samplers = {
        'Standard': hmc_standard,
        'Cold': hmc_cold, 
        'Tempered': hmc_tempered
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
        return None, None, None, None, None
    
    log_posteriors = {
        'Standard': log_post_standard,
        'Cold': log_post_cold,
        'Tempered': log_post_tempered
    }
    
    print("\n" + "="*80)
    print("POSTERIOR COMPARISON - STANDARD vs COLD vs TEMPERED (HMC)")
    print("="*80)
    print("DEFINITIONS:")
    print("  Standard:  p(w|D) ∝ p(y|X,w) * p(w)")
    print("  Cold:      p(w|D) ∝ [p(y|X,w)]^(1/T) * [p(w)]^(1/T)")
    print("  Tempered:  p(w|D) ∝ [p(y|X,w)]^(1/λ) * p(w)")
    print("="*80)
    print(f"{'Parameter':<12} {'True':<10} {'Analytical':<12} {'Standard HMC':<14} {'Cold HMC':<14} {'Tempered HMC':<14}")
    print(f"{'Name':<12} {'Value':<10} {'Mean±Std':<12} {'Mean±Std':<14} {'Mean±Std':<14} {'Mean±Std':<14}")
    print("-" * 88)
    
    param_names = ['Bias', 'Weight 1', 'Weight 2']
    sigma_true = np.sqrt(np.diag(Sigma_true.numpy()))
    
    for i in range(len(mu_true)):
        row = f"{param_names[i]:<12} {w_true[i]:<10.4f} {mu_true[i].item():.3f}±{sigma_true[i]:.3f}   "
        
        for sampler_name in ['Standard', 'Cold', 'Tempered']:
            sampler = samplers[sampler_name]
            mu_hmc = sampler.sample_mean[i].item()
            sigma_hmc = torch.sqrt(sampler.sample_cov[i, i]).item()
            row += f"{mu_hmc:.3f}±{sigma_hmc:.3f}     "
        
        print(row)
    
    print("="*80)
    
    return mu_true, Sigma_true, samplers, all_samples, log_posteriors

def plot_hmc_results(X, y, w_true, mu_true, Sigma_true, samplers, all_samples, log_posteriors):
    """Plot comparison results for HMC models using cross entropy instead of KL divergence"""
    
    n_params = len(w_true)
    
    # Create output directory
    output_dir = "figs/hmc_cold_posterior_linear_regression_cross_entropy"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving figures to: {output_dir}")
    
    # Extract features (skip bias column)
    X_features = X[:, 1:]
    
    # 1. Interactive 3D plot of posterior samples
    if n_params >= 3:
        # Sample from analytical posterior
        n_viz_samples = 2000
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
            marker=dict(size=2, color='blue', opacity=0.4),
            name='Analytical Posterior',
            hovertemplate='<b>Analytical</b><br>Bias: %{x:.3f}<br>Weight 1: %{y:.3f}<br>Weight 2: %{z:.3f}<extra></extra>'
        ))
        
        # Add HMC samples for each model
        colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
        for sampler_name, samples in all_samples.items():
            sampler = samplers[sampler_name]
            
            # Use samples for visualization
            n_total = len(samples)
            n_viz = min(2000, n_total)
            indices = np.random.choice(n_total, n_viz, replace=False)
            viz_samples = samples[indices]
            
            print(f"Plotting {len(viz_samples)} {sampler_name} HMC samples")
            
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
                marker=dict(size=2, color=colors[sampler_name], opacity=0.4),
                name=label,
                hovertemplate=f'<b>{sampler_name} HMC</b><br>Bias: %{{x:.3f}}<br>Weight 1: %{{y:.3f}}<br>Weight 2: %{{z:.3f}}<extra></extra>'
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
            title='3D Posterior Samples: Analytical vs HMC (Standard/Cold/Tempered)<br><sub>Cold: scales likelihood & prior by 1/T | Tempered: scales only likelihood by 1/λ</sub>',
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
        fig.write_html(f"{output_dir}/3d_posterior_samples_hmc.html")
        print(f"Saved: {output_dir}/3d_posterior_samples_hmc.html")
    
    # 2. HMC trace plots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    param_names = ['Bias', 'Weight 1', 'Weight 2']
    colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
    
    for i, param_name in enumerate(param_names):
        for j, (sampler_name, samples) in enumerate(all_samples.items()):
            sampler = samplers[sampler_name]
            
            # Trace plot
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
                axes[i, j].set_xlabel('HMC Iteration', fontsize=9)
            if j == 0:  # Left column
                axes[i, j].set_ylabel('Parameter Value', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hmc_trace_plots.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/hmc_trace_plots.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/hmc_trace_plots.png")
    plt.show()
    
    # 3. Test data Cross Entropy analysis with PCA visualization
    print(f"\n{'='*60}")
    print("GENERATING TEST DATA FOR CROSS ENTROPY ANALYSIS (HMC)")
    print(f"{'='*60}")
    
    # Generate test data with same correlation structure
    n_test = 500
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
    
    # Compute Cross Entropies for all HMC models
    all_test_ce_hmc = {}
    colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
    
    for sampler_name, sampler in samplers.items():
        print(f"Computing test Cross Entropies for {sampler_name} HMC...")
        
        test_cross_entropies = np.zeros(n_test)
        beta = sampler.beta
        
        for i in tqdm(range(n_test), desc=f"Test CE for {sampler_name}"):
            x_point = X_test_tensor[i:i+1]
            
            # True posterior predictive
            pred_mean_true = (x_point @ mu_true).item()
            pred_var_true = (x_point @ Sigma_true @ x_point.T).item() + 1.0/beta
            
            # HMC posterior predictive (using sample statistics)
            pred_mean_hmc = (x_point @ sampler.sample_mean).item()
            pred_var_hmc = (x_point @ sampler.sample_cov @ x_point.T).item() + 1.0/beta
            
            # Cross entropy H(p_true, p_hmc) where p_true is the reference
            cross_entropy = compute_cross_entropy(pred_mean_true, pred_var_true, 
                                                pred_mean_hmc, pred_var_hmc)
            test_cross_entropies[i] = cross_entropy
        
        all_test_ce_hmc[sampler_name] = test_cross_entropies
        
        # Print test statistics
        print(f"\n{sampler_name} HMC Test Cross Entropy Statistics:")
        print(f"  Mean: {test_cross_entropies.mean():.6f}")
        print(f"  Max:  {test_cross_entropies.max():.6f}")
        print(f"  Min:  {test_cross_entropies.min():.6f}")
        print(f"  Std:  {test_cross_entropies.std():.6f}")
    
    # Create two plots: PC1 vs CE and PC2 vs CE
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: PC1 vs Cross Entropy
    for sampler_name, test_ce in all_test_ce_hmc.items():
        sampler = samplers[sampler_name]
        
        # Create proper labels
        if sampler.posterior_type == "cold":
            label = f'{sampler_name} (Cold T={sampler.temperature})'
        elif sampler.posterior_type == "tempered":
            label = f'{sampler_name} (Tempered λ={sampler.temperature})'
        else:
            label = f'{sampler_name} (Standard)'
        
        axes[0].scatter(X_test_pca[:, 0], test_ce, 
                       c=colors[sampler_name], s=25, alpha=0.7, 
                       label=label, edgecolor='black', linewidth=0.3)
    
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('H(True, HMC) Predictive', fontsize=12)
    axes[0].set_title('Test Cross Entropy vs PC1 (HMC)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: PC2 vs Cross Entropy
    for sampler_name, test_ce in all_test_ce_hmc.items():
        sampler = samplers[sampler_name]
        
        # Create proper labels
        if sampler.posterior_type == "cold":
            label = f'{sampler_name} (Cold T={sampler.temperature})'
        elif sampler.posterior_type == "tempered":
            label = f'{sampler_name} (Tempered λ={sampler.temperature})'
        else:
            label = f'{sampler_name} (Standard)'
        
        axes[1].scatter(X_test_pca[:, 1], test_ce, 
                       c=colors[sampler_name], s=25, alpha=0.7, 
                       label=label, edgecolor='black', linewidth=0.3)
    
    axes[1].set_xlabel('PC2', fontsize=12)
    axes[1].set_ylabel('H(True, HMC) Predictive', fontsize=12)
    axes[1].set_title('Test Cross Entropy vs PC2 (HMC)', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('HMC: Cross Entropy vs Principal Components\n(Lower CE = Better Approximation)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/hmc_ce_vs_pca_components.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/hmc_ce_vs_pca_components.pdf", bbox_inches='tight')
    plt.show()
    
    # 4. Input data visualization with Cross Entropy backgrounds (HMC version)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create grid for Cross Entropy computation
    x1_min, x1_max = X_features[:, 0].min() - 0.5, X_features[:, 0].max() + 0.5
    x2_min, x2_max = X_features[:, 1].min() - 0.5, X_features[:, 1].max() + 0.5
    grid_resolution = 60
    x1_grid = np.linspace(x1_min, x1_max, grid_resolution)
    x2_grid = np.linspace(x2_min, x2_max, grid_resolution)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    n_grid_points = len(grid_points)
    X_grid = np.column_stack([np.ones(n_grid_points), grid_points])
    X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)
    
    print("Computing Cross Entropies for all HMC models...")
    
    # First pass: compute all Cross Entropies to determine global min/max
    all_cross_entropies_hmc = []
    
    for idx, (sampler_name, sampler) in enumerate(samplers.items()):
        print(f"Computing Cross Entropies for {sampler_name} HMC...")
        
        cross_entropies = np.zeros(n_grid_points)
        
        for i in tqdm(range(n_grid_points), desc=f"CE for {sampler_name}"):
            x_point = X_grid_tensor[i:i+1]
            
            # True posterior predictive
            pred_mean_true = (x_point @ mu_true).item()
            pred_var_true = (x_point @ Sigma_true @ x_point.T).item() + 1.0/sampler.beta
            
            # HMC posterior predictive
            pred_mean_hmc = (x_point @ sampler.sample_mean).item()
            pred_var_hmc = (x_point @ sampler.sample_cov @ x_point.T).item() + 1.0/sampler.beta
            
            # Cross entropy H(p_true, p_hmc)
            cross_entropy = compute_cross_entropy(pred_mean_true, pred_var_true, 
                                                pred_mean_hmc, pred_var_hmc)
            cross_entropies[i] = cross_entropy
        
        all_cross_entropies_hmc.append(cross_entropies)
    
    # Determine global min and max for consistent color scaling
    global_ce_min = min([ce.min() for ce in all_cross_entropies_hmc])
    global_ce_max = max([ce.max() for ce in all_cross_entropies_hmc])
    
    print(f"\nGlobal HMC Cross Entropy range: [{global_ce_min:.6f}, {global_ce_max:.6f}]")
    
    # Second pass: create plots with consistent color scale
    for idx, (sampler_name, sampler) in enumerate(samplers.items()):
        cross_entropies = all_cross_entropies_hmc[idx]
        
        # Plot for this model with consistent color scale
        CE_grid = cross_entropies.reshape(X1_grid.shape)
        im = axes[idx].imshow(CE_grid, extent=[x1_min, x1_max, x2_min, x2_max], 
                            origin='lower', cmap='Reds', alpha=0.7, aspect='auto',
                            vmin=global_ce_min, vmax=global_ce_max)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], shrink=0.8)
        cbar.set_label('H(True, HMC)', fontsize=10)
        
        # Overlay data points
        scatter = axes[idx].scatter(X_features[:, 0], X_features[:, 1], c=y, s=30, 
                                  cmap='viridis', alpha=0.9, edgecolor='white', linewidth=0.8)
        
        axes[idx].set_xlabel('Feature 1', fontsize=11)
        axes[idx].set_ylabel('Feature 2', fontsize=11)
        axes[idx].set_title(f'{sampler_name} HMC\n{sampler.posterior_type.title()} ({"T" if sampler.posterior_type == "cold" else "λ"}={sampler.temperature})', 
                          fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Print statistics
        print(f"\n{sampler_name} HMC Cross Entropy Statistics:")
        print(f"  Mean: {cross_entropies.mean():.6f}")
        print(f"  Max:  {cross_entropies.max():.6f}")
        print(f"  Std:  {cross_entropies.std():.6f}")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hmc_cross_entropy_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/hmc_cross_entropy_heatmaps.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/hmc_cross_entropy_heatmaps.png")
    plt.show()
    
    print(f"\nAll HMC figures saved to: {output_dir}")
    print(f"Files saved:")
    print(f"  - 3d_posterior_samples_hmc.html (interactive)")
    print(f"  - hmc_trace_plots.png/.pdf")
    print(f"  - hmc_ce_vs_pca_components.png/.pdf")
    print(f"  - hmc_cross_entropy_heatmaps.png/.pdf")

def main():
    """Main execution function for HMC comparison using Cross Entropy"""
    print("Bayesian Linear Regression: HMC Standard vs Cold vs Tempered (Cross Entropy Analysis)")
    print("="*80)
    
    # Generate synthetic data with correlated features (same as VI version)
    X, y, w_true = generate_synthetic_data(n_samples=100, n_features=3, noise_std=0.1)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"True weights: {w_true}")
    
    # Hyperparameters (same as VI version)
    alpha = 1.0   # Prior precision
    beta = 100.0  # Likelihood precision (1/noise_variance)
    
    # Compare all posteriors using HMC
    mu_true, Sigma_true, samplers, all_samples, log_posteriors = compare_posteriors_hmc(
        X, y, w_true, alpha=alpha, beta=beta
    )
    
    # Check if sampling succeeded
    if samplers is None:
        print("ERROR: HMC sampling failed. Exiting.")
        return
    
    # Plot HMC results with Cross Entropy analysis
    plot_hmc_results(X, y, w_true, mu_true, Sigma_true, samplers, all_samples, log_posteriors)
    
    # Predictive performance comparison
    print(f"\nPREDICTIVE PERFORMANCE COMPARISON (HMC):")
    print(f"{'='*55}")
    X_test = torch.tensor(X[:10], dtype=torch.float32)  # Use first 10 samples as test
    y_test = torch.tensor(y[:10], dtype=torch.float32)
    
    # Analytical predictions
    y_pred_analytical = X_test @ mu_true
    mse_analytical = torch.mean((y_test - y_pred_analytical)**2)
    
    print(f"{'Model':<20} {'MSE':<12} {'Temperature':<12}")
    print("-" * 45)
    print(f"{'Analytical':<20} {mse_analytical:.6f} {'N/A':<12}")
    
    # HMC predictions for each model
    for sampler_name, sampler in samplers.items():
        y_pred_hmc = X_test @ sampler.sample_mean
        mse_hmc = torch.mean((y_test - y_pred_hmc)**2)
        print(f"{sampler_name + ' HMC':<20} {mse_hmc:.6f} {sampler.temperature:<12}")
    
    print(f"{'='*55}")
    
    # Cross Entropy summary
    print(f"\nCROSS ENTROPY ANALYSIS SUMMARY:")
    print(f"{'='*55}")
    print("Cross Entropy H(p_true, p_approx) measures the expected negative")
    print("log-likelihood of the approximate distribution when the true")
    print("distribution is the reference. Lower values indicate better")
    print("approximation quality.")
    print(f"{'='*55}")

if __name__ == "__main__":
    main()
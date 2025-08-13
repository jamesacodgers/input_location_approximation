import torch
import torch.nn as nn
import torch.optim as optim
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
    Bayesian Linear Regression with both analytical solution and MFVI approximation
    
    Model: y = X @ w + noise
    Prior: w ~ N(0, alpha^-1 * I)
    Likelihood: y|X,w ~ N(X @ w, beta^-1 * I)
    """
    
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # precision of prior
        self.beta = beta    # precision of likelihood (inverse noise variance)
        
    def analytical_posterior(self, X, y):
        """
        Compute the analytical posterior for Bayesian linear regression
        
        Posterior: w|X,y ~ N(mu_n, Sigma_n)
        where:
        Sigma_n = (alpha*I + beta*X^T*X)^-1
        mu_n = beta * Sigma_n * X^T * y
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Posterior covariance
        S_n_inv = self.alpha * torch.eye(X.shape[1]) + self.beta * X.T @ X
        S_n = torch.inverse(S_n_inv)
        
        # Posterior mean
        mu_n = self.beta * S_n @ X.T @ y
        
        return mu_n, S_n

class VariationalLinearRegression(nn.Module):
    """
    Mean Field Variational Inference for Bayesian Linear Regression
    
    Approximating posterior: q(w) = N(mu, diag(sigma^2))
    Supports cold and tempered posteriors with correct definitions:
    - Cold: scales both likelihood and prior by 1/T
    - Tempered: scales only likelihood by 1/λ, prior unchanged
    """
    
    def __init__(self, n_features, alpha=1.0, beta=1.0, temperature=1.0, posterior_type="standard"):
        super().__init__()
        self.n_features = n_features
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.posterior_type = posterior_type  # "standard", "cold", or "tempered"
        
        # Variational parameters
        self.mu = nn.Parameter(torch.randn(n_features))
        self.log_sigma = nn.Parameter(torch.randn(n_features))
        
    @property
    def sigma(self):
        return torch.exp(self.log_sigma)
    
    def sample_weights(self, n_samples=1):
        """Sample weights from the variational distribution"""
        eps = torch.randn(n_samples, self.n_features)
        return self.mu + self.sigma * eps
    
    def kl_divergence(self):
        """
        KL divergence between q(w) and prior p(w)
        For cold posteriors: prior is also scaled by 1/T (equivalent to changing prior precision)
        For tempered posteriors: prior remains unchanged
        """
        if self.posterior_type == "cold":
            # Cold posterior: scale prior by 1/T (equivalent to alpha_cold = alpha/T)
            alpha_effective = self.alpha / self.temperature
        else:
            # Standard or tempered: use original prior
            alpha_effective = self.alpha
            
        # KL[q(w) || p(w)] where p(w) = N(0, alpha_effective^-1 * I)
        kl = 0.5 * (alpha_effective * (self.mu**2 + self.sigma**2) - 
                   2 * self.log_sigma - torch.log(torch.tensor(alpha_effective)) - 1)
        return kl.sum()
    
    def elbo(self, X, y, n_samples=10):
        """
        Evidence Lower Bound with different posterior types:
        
        Standard: ELBO = E_q[log p(y|X,w)] - KL[q(w)||p(w)]
        Cold: ELBO = (1/T) * E_q[log p(y|X,w)] - KL[q(w)||p_cold(w)]
              where p_cold(w) has precision alpha/T
        Tempered: ELBO = (1/λ) * E_q[log p(y|X,w)] - KL[q(w)||p(w)]
                  where λ is the temperature parameter
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Sample weights and compute log likelihood
        w_samples = self.sample_weights(n_samples)  # [n_samples, n_features]
        
        # Predictions for each sample
        y_pred = X @ w_samples.T  # [n_data, n_samples]
        
        # Log likelihood: log p(y|X,w) = -0.5 * beta * ||y - X@w||^2 + const
        log_likelihood = -0.5 * self.beta * ((y.unsqueeze(1) - y_pred)**2).sum(dim=0)
        expected_log_likelihood = log_likelihood.mean()
        
        # Apply temperature scaling to likelihood
        if self.posterior_type == "cold":
            # Cold posterior: scale likelihood by 1/T
            scaled_log_likelihood = (1.0 / self.temperature) * expected_log_likelihood
        elif self.posterior_type == "tempered":
            # Tempered posterior: scale likelihood by 1/λ (using temperature as λ)
            scaled_log_likelihood = (1.0 / self.temperature) * expected_log_likelihood
        else:
            # Standard posterior: no scaling
            scaled_log_likelihood = expected_log_likelihood
        
        # KL divergence (handles cold vs tempered internally)
        kl_div = self.kl_divergence()
        
        # ELBO
        elbo = scaled_log_likelihood - kl_div
        
        return -elbo  # Return negative ELBO for minimization

def generate_synthetic_data(n_samples=100, n_features=3, noise_std=0.1, seed=42):
    """Generate synthetic regression data with correlated features"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # True weights
    w_true = torch.randn(n_features)
    
    # Generate correlated features using multivariate normal
    # Create covariance matrix with strong correlation between features
    correlation = 0.8  # Strong positive correlation
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

def fit_variational_regression(X, y, n_epochs=20000, lr=0.01, alpha=1.0, beta=100.0, temperature=1.0, posterior_type="standard", model_name="VI"):
    """Fit the variational Bayesian linear regression"""
    
    n_features = X.shape[1]
    model = VariationalLinearRegression(n_features, alpha=alpha, beta=beta, 
                                      temperature=temperature, posterior_type=posterior_type)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    pbar = tqdm(range(n_epochs), desc=f"Training {model_name}")
    for epoch in pbar:
        optimizer.zero_grad()
        loss = model.elbo(X, y, n_samples=10)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Update progress bar with loss
        if epoch % 100 == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return model, losses

def compare_posteriors(X, y, w_true, alpha=1.0, beta=100.0):
    """Compare analytical, standard, cold, and tempered variational posteriors"""
    
    # Analytical posterior
    analytical = BayesianLinearRegression(alpha=alpha, beta=beta)
    mu_true, Sigma_true = analytical.analytical_posterior(X, y)
    
    # Standard VI (no temperature scaling)
    vi_model_standard, losses_standard = fit_variational_regression(
        X, y, alpha=alpha, beta=beta, temperature=1.0, 
        posterior_type="standard", model_name="Standard VI"
    )
    
    # Cold posterior VI (T=0.8: scales both likelihood and prior by 1/T)
    vi_model_cold, losses_cold = fit_variational_regression(
        X, y, alpha=alpha, beta=beta, temperature=0.8, 
        posterior_type="cold", model_name="Cold VI (T=0.8)"
    )
    
    # Tempered posterior VI (λ=0.8: scales only likelihood by 1/λ)
    vi_model_tempered, losses_tempered = fit_variational_regression(
        X, y, alpha=alpha, beta=beta, temperature=0.8, 
        posterior_type="tempered", model_name="Tempered VI (λ=0.8)"
    )
    
    models = {
        'Standard': vi_model_standard,
        'Cold': vi_model_cold, 
        'Tempered': vi_model_tempered
    }
    
    losses_dict = {
        'Standard': losses_standard,
        'Cold': losses_cold,
        'Tempered': losses_tempered
    }
    
    print("\n" + "="*80)
    print("POSTERIOR COMPARISON - STANDARD vs COLD vs TEMPERED")
    print("="*80)
    print("DEFINITIONS:")
    print("  Standard:  ELBO = E_q[log p(y|X,w)] - KL[q(w)||p(w)]")
    print("  Cold:      ELBO = (1/T)E_q[log p(y|X,w)] - KL[q(w)||p_cold(w)], p_cold has α/T")
    print("  Tempered:  ELBO = (1/λ)E_q[log p(y|X,w)] - KL[q(w)||p(w)]")
    print("="*80)
    print(f"{'Parameter':<12} {'True':<10} {'Analytical':<12} {'Standard VI':<12} {'Cold VI':<12} {'Tempered VI':<12}")
    print(f"{'Name':<12} {'Value':<10} {'Mean±Std':<12} {'Mean±Std':<12} {'Mean±Std':<12} {'Mean±Std':<12}")
    print("-" * 82)
    
    param_names = ['Bias', 'Weight 1', 'Weight 2']
    sigma_true = np.sqrt(np.diag(Sigma_true.numpy()))
    
    for i in range(len(mu_true)):
        row = f"{param_names[i]:<12} {w_true[i]:<10.4f} {mu_true[i].item():.3f}±{sigma_true[i]:.3f}   "
        
        for model_name in ['Standard', 'Cold', 'Tempered']:
            model = models[model_name]
            mu_vi = model.mu.detach().numpy()[i]
            sigma_vi = model.sigma.detach().numpy()[i]
            row += f"{mu_vi:.3f}±{sigma_vi:.3f}   "
        
        print(row)
    
    print("="*80)
    
    return mu_true, Sigma_true, models, losses_dict

def plot_results(X, y, w_true, mu_true, Sigma_true, models, losses_dict):
    """Plot comparison results for all models"""
    
    n_params = len(w_true)
    
    # Create output directory
    output_dir = "figs/cold_posterior_linear_regression"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving figures to: {output_dir}")
    
    # Extract features (skip bias column) - needed for multiple plots
    X_features = X[:, 1:]
    
    # 1. Interactive 3D plot of posterior samples
    if n_params >= 3:
        # Sample from analytical posterior
        n_samples = 800
        analytical_dist = MultivariateNormal(mu_true, Sigma_true)
        analytical_samples = analytical_dist.sample((n_samples,))
        
        # Create interactive 3D scatter plot
        fig = go.Figure()
        
        # Add analytical samples
        fig.add_trace(go.Scatter3d(
            x=analytical_samples[:, 0].numpy(),
            y=analytical_samples[:, 1].numpy(), 
            z=analytical_samples[:, 2].numpy(),
            mode='markers',
            marker=dict(size=3, color='blue', opacity=0.6),
            name='Analytical Posterior',
            hovertemplate='<b>Analytical</b><br>Bias: %{x:.3f}<br>Weight 1: %{y:.3f}<br>Weight 2: %{z:.3f}<extra></extra>'
        ))
        
        # Add variational samples for each model
        colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
        for model_name, model in models.items():
            vi_samples = model.sample_weights(n_samples).detach()
            
            # Create label with correct notation
            if model.posterior_type == "cold":
                label = f'{model_name} (Cold T={model.temperature})'
            elif model.posterior_type == "tempered":
                label = f'{model_name} (Tempered λ={model.temperature})'
            else:
                label = f'{model_name} (Standard)'
                
            fig.add_trace(go.Scatter3d(
                x=vi_samples[:, 0].numpy(),
                y=vi_samples[:, 1].numpy(),
                z=vi_samples[:, 2].numpy(),
                mode='markers',
                marker=dict(size=3, color=colors[model_name], opacity=0.6),
                name=label,
                hovertemplate=f'<b>{model_name} VI</b><br>Bias: %{{x:.3f}}<br>Weight 1: %{{y:.3f}}<br>Weight 2: %{{z:.3f}}<extra></extra>'
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
            title='3D Posterior Samples: Analytical vs Standard/Cold/Tempered VI<br><sub>Cold: scales likelihood & prior by 1/T | Tempered: scales only likelihood by 1/λ</sub>',
            scene=dict(
                xaxis_title='Bias (w[0])',
                yaxis_title='Weight 1 (w[1])',
                zaxis_title='Weight 2 (w[2])',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=900, height=700, margin=dict(l=0, r=0, b=0, t=60)
        )
        fig.show()
        
        # Save interactive plot
        fig.write_html(f"{output_dir}/3d_posterior_samples.html")
        print(f"Saved: {output_dir}/3d_posterior_samples.html")
    
    # 2. Training loss curves comparison
    plt.figure(figsize=(12, 8))
    
    # Plot losses for all models
    colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
    for model_name, losses in losses_dict.items():
        model = models[model_name]
        
        # Create proper labels
        if model.posterior_type == "cold":
            label = f'{model_name} (Cold T={model.temperature})'
        elif model.posterior_type == "tempered":
            label = f'{model_name} (Tempered λ={model.temperature})'
        else:
            label = f'{model_name} (Standard)'
            
        plt.plot(losses, linewidth=2, color=colors[model_name], 
                label=label, alpha=0.8)
    
    plt.yscale('log')
    plt.title('Training Loss Comparison: Standard vs Cold vs Tempered VI', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Negative ELBO (log scale)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save training loss comparison plots BEFORE showing
    plt.savefig(f"{output_dir}/training_loss_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/training_loss_comparison.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/training_loss_comparison.png")
    print(f"Saved: {output_dir}/training_loss_comparison.pdf")
    
    plt.show()
    
    
    # 3. Test data KL analysis with PCA visualization
    print(f"\n{'='*60}")
    print("GENERATING TEST DATA FOR KL ANALYSIS")
    print(f"{'='*60}")
    
    print(f"\nAll figures saved to: {output_dir}")
    print(f"Files saved:")
    print(f"  - 3d_posterior_samples.html (interactive)")
    print(f"  - training_loss_comparison.png/.pdf")
    print(f"  - kl_divergence_heatmaps.png/.pdf") 
    print(f"  - kl_vs_pca_components.png/.pdf")
    
    # Generate test data with same correlation structure
    n_test = 200
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
    
    # Compute KL divergences for all models
    all_test_kl = {}
    colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
    
    for model_name, model in models.items():
        print(f"Computing test KL divergences for {model_name} VI...")
        
        test_kl_divergences = np.zeros(n_test)
        
        for i in tqdm(range(n_test), desc=f"Test KL for {model_name}"):
            x_point = X_test_tensor[i:i+1]
            
            # True posterior predictive
            pred_mean_true = (x_point @ mu_true).item()
            pred_var_true = (x_point @ Sigma_true @ x_point.T).item() + 1.0/model.beta
            
            # Variational posterior predictive
            mu_vi = model.mu.detach()
            sigma_vi = model.sigma.detach()
            pred_mean_vi = (x_point @ mu_vi).item()
            pred_var_vi = ((x_point**2) @ (sigma_vi**2)).item() + 1.0/model.beta
            
            # KL divergence
            mean_diff_sq = (pred_mean_true - pred_mean_vi)**2
            kl_div = 0.5 * (
                np.log(pred_var_vi / pred_var_true) +
                pred_var_true / pred_var_vi +
                mean_diff_sq / pred_var_vi - 1
            )
            test_kl_divergences[i] = kl_div
        
        all_test_kl[model_name] = test_kl_divergences
        
        # Print test statistics
        print(f"\n{model_name} VI Test KL Statistics:")
        print(f"  Mean: {test_kl_divergences.mean():.6f}")
        print(f"  Max:  {test_kl_divergences.max():.6f}")
        print(f"  Min:  {test_kl_divergences.min():.6f}")
        print(f"  Std:  {test_kl_divergences.std():.6f}")
    
    # Create two plots: PC1 vs KL and PC2 vs KL
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: PC1 vs KL Divergence
    for model_name, test_kl in all_test_kl.items():
        model = models[model_name]
        
        # Create proper labels
        if model.posterior_type == "cold":
            label = f'{model_name} (Cold T={model.temperature})'
        elif model.posterior_type == "tempered":
            label = f'{model_name} (Tempered λ={model.temperature})'
        else:
            label = f'{model_name} (Standard)'
        
        axes[0].scatter(X_test_pca[:, 0], test_kl, 
                       c=colors[model_name], s=25, alpha=0.7, 
                       label=label, edgecolor='black', linewidth=0.3)
    
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('KL[True || VI] Predictive', fontsize=12)
    axes[0].set_title('Test KL Divergence vs PC1', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: PC2 vs KL Divergence
    for model_name, test_kl in all_test_kl.items():
        model = models[model_name]
        
        # Create proper labels
        if model.posterior_type == "cold":
            label = f'{model_name} (Cold T={model.temperature})'
        elif model.posterior_type == "tempered":
            label = f'{model_name} (Tempered λ={model.temperature})'
        else:
            label = f'{model_name} (Standard)'
        
        axes[1].scatter(X_test_pca[:, 1], test_kl, 
                       c=colors[model_name], s=25, alpha=0.7, 
                       label=label, edgecolor='black', linewidth=0.3)
    
    axes[1].set_xlabel('PC2', fontsize=12)
    axes[1].set_ylabel('KL[True || VI] Predictive', fontsize=12)
    axes[1].set_title('Test KL Divergence vs PC2', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('KL Divergence vs Principal Components\n(Lower KL = Better Approximation)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/kl_vs_pca_components.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/kl_vs_pca_components.pdf", bbox_inches='tight')
    plt.show()
    
    print(f"\n{'='*60}")
    print("PCA INTERPRETATION")
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
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    overall_stats = {}
    for model_name, test_kl in all_test_kl.items():
        overall_stats[model_name] = {
            'mean_kl': test_kl.mean(),
            'median_kl': np.median(test_kl),
            'std_kl': test_kl.std()
        }
    
    print(f"{'Model':<15} {'Mean KL':<12} {'Median KL':<12} {'Std KL':<12}")
    print("-" * 55)
    for model_name, stats in overall_stats.items():
        print(f"{model_name:<15} {stats['mean_kl']:<12.6f} {stats['median_kl']:<12.6f} {stats['std_kl']:<12.6f}")
    
    print(f"{'='*60}")
    
    # 3. Input data visualization with KL divergence backgrounds
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create grid for KL divergence computation
    x1_min, x1_max = X_features[:, 0].min() - 0.5, X_features[:, 0].max() + 0.5
    x2_min, x2_max = X_features[:, 1].min() - 0.5, X_features[:, 1].max() + 0.5
    grid_resolution = 40  # Reduced for faster computation with 3 models
    x1_grid = np.linspace(x1_min, x1_max, grid_resolution)
    x2_grid = np.linspace(x2_min, x2_max, grid_resolution)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    n_grid_points = len(grid_points)
    X_grid = np.column_stack([np.ones(n_grid_points), grid_points])
    X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)
    
    print("Computing KL divergences for all models...")
    
    # First pass: compute all KL divergences to determine global min/max
    all_kl_divergences = []
    
    for idx, (model_name, model) in enumerate(models.items()):
        print(f"Computing KL divergences for {model_name} VI...")
        
        kl_divergences = np.zeros(n_grid_points)
        
        for i in tqdm(range(n_grid_points), desc=f"KL for {model_name}"):
            x_point = X_grid_tensor[i:i+1]
            
            # True posterior predictive
            pred_mean_true = (x_point @ mu_true).item()
            pred_var_true = (x_point @ Sigma_true @ x_point.T).item() + 1.0/model.beta
            
            # Variational posterior predictive
            mu_vi = model.mu.detach()
            sigma_vi = model.sigma.detach()
            pred_mean_vi = (x_point @ mu_vi).item()
            pred_var_vi = ((x_point**2) @ (sigma_vi**2)).item() + 1.0/model.beta
            
            # KL divergence
            mean_diff_sq = (pred_mean_true - pred_mean_vi)**2
            kl_div = 0.5 * (
                np.log(pred_var_vi / pred_var_true) +
                pred_var_true / pred_var_vi +
                mean_diff_sq / pred_var_vi - 1
            )
            kl_divergences[i] = kl_div
        
        all_kl_divergences.append(kl_divergences)
    
    # Determine global min and max for consistent color scaling
    global_kl_min = min([kl.min() for kl in all_kl_divergences])
    global_kl_max = max([kl.max() for kl in all_kl_divergences])
    
    print(f"\nGlobal KL range: [{global_kl_min:.6f}, {global_kl_max:.6f}]")
    
    # Second pass: create plots with consistent color scale
    for idx, (model_name, model) in enumerate(models.items()):
        kl_divergences = all_kl_divergences[idx]
        
        # Plot for this model with consistent color scale
        KL_grid = kl_divergences.reshape(X1_grid.shape)
        im = axes[idx].imshow(KL_grid, extent=[x1_min, x1_max, x2_min, x2_max], 
                            origin='lower', cmap='Reds', alpha=0.7, aspect='auto',
                            vmin=global_kl_min, vmax=global_kl_max)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], shrink=0.8)
        cbar.set_label('KL[True || VI]', fontsize=10)
        
        # Overlay data points
        scatter = axes[idx].scatter(X_features[:, 0], X_features[:, 1], c=y, s=30, 
                                  cmap='viridis', alpha=0.9, edgecolor='white', linewidth=0.8)
        
        axes[idx].set_xlabel('Feature 1', fontsize=11)
        axes[idx].set_ylabel('Feature 2', fontsize=11)
        axes[idx].set_title(f'{model_name} VI\n{model.posterior_type.title()} ({"T" if model.posterior_type == "cold" else "λ"}={model.temperature})', 
                          fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Print statistics
        print(f"\n{model_name} VI KL Statistics:")
        print(f"  Mean: {kl_divergences.mean():.6f}")
        print(f"  Max:  {kl_divergences.max():.6f}")
        print(f"  Std:  {kl_divergences.std():.6f}")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/kl_divergence_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/kl_divergence_heatmaps.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/kl_divergence_heatmaps.png")
    print(f"Saved: {output_dir}/kl_divergence_heatmaps.pdf")
    plt.show()
    
    # Print data summary
    print(f"\n{'='*50}")
    print("TRAINING DATA SUMMARY")
    print(f"{'='*50}")
    print(f"Number of samples: {len(y)}")
    print(f"Number of features: {X_features.shape[1]} (+ bias term)")
    print(f"\nFeature Statistics:")
    print(f"{'Feature':<12} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-" * 55)
    feature_names = ['Feature 1', 'Feature 2']
    for i, name in enumerate(feature_names):
        feat = X_features[:, i]
        print(f"{name:<12} {feat.mean():<10.3f} {feat.std():<10.3f} {feat.min():<10.3f} {feat.max():<10.3f}")
    
    print(f"\nTarget Statistics:")
    print(f"{'Target':<12} {y.mean():<10.3f} {y.std():<10.3f} {y.min():<10.3f} {y.max():<10.3f}")
    
    print(f"\nTrue Model: y = {w_true[0]:.3f} + {w_true[1]:.3f}*x1 + {w_true[2]:.3f}*x2 + noise")
    print(f"{'='*50}")
    
    # Summary comparison of all models
    print(f"\n{'='*70}")
    print("TEMPERATURE EFFECTS SUMMARY")
    print(f"{'='*70}")
    
    for model_name, model in models.items():
        mu_vi = model.mu.detach().numpy()
        sigma_vi = model.sigma.detach().numpy()
        
        # Compute total uncertainty (sum of variances)
        total_uncertainty = np.sum(sigma_vi**2)
        
        # Compute distance from analytical mean
        mean_distance = np.linalg.norm(mu_vi - mu_true.numpy())
        
        print(f"{model_name} VI (T={model.temperature}):")
        print(f"  Total Uncertainty: {total_uncertainty:.6f}")
        print(f"  Distance from True: {mean_distance:.6f}")
        print(f"  Final Loss: {losses_dict[model_name][-1]:.6f}")
    
    print(f"\nTemperature Effects (Both at T=λ=0.8):")
    print(f"  Cold (T=0.8):      Scales BOTH likelihood & prior by 1/0.8=1.25 → concentrated posterior")
    print(f"  Standard (T=1):    Original Bayesian posterior") 
    print(f"  Tempered (λ=0.8):  Scales ONLY likelihood by 1/0.8=1.25 → concentrated posterior")
    print(f"  Key Difference:    Cold affects prior precision; Tempered keeps prior unchanged")
    print(f"{'='*70}")
    
    # Convergence analysis for all models
    print(f"\nCONVERGENCE ANALYSIS")
    print(f"{'='*50}")
    for model_name, losses in losses_dict.items():
        last_10_percent = int(0.1 * len(losses))
        recent_losses = losses[-last_10_percent:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        cv = loss_std / loss_mean * 100
        
        status = "✓" if cv < 0.1 else "⚠" if cv < 1.0 else "✗"
        print(f"{model_name} VI: {status} CV={cv:.4f}% (Final Loss: {losses[-1]:.6f})")
    
    print(f"{'='*50}")

def main():
    """Main execution function"""
    print("Bayesian Linear Regression: Standard vs Cold vs Tempered VI")
    print("="*65)
    
    # Generate synthetic data with correlated features
    X, y, w_true = generate_synthetic_data(n_samples=100, n_features=3, noise_std=0.1)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"True weights: {w_true}")
    
    # Hyperparameters
    alpha = 1.0   # Prior precision
    beta = 100.0  # Likelihood precision (1/noise_variance)
    
    # Compare all posteriors
    mu_true, Sigma_true, models, losses_dict = compare_posteriors(X, y, w_true, alpha=alpha, beta=beta)
    
    # Plot results for all models
    plot_results(X, y, w_true, mu_true, Sigma_true, models, losses_dict)
    
    # Predictive performance comparison
    print(f"\nPREDICTIVE PERFORMANCE COMPARISON:")
    print(f"{'='*50}")
    X_test = torch.tensor(X[:10], dtype=torch.float32)  # Use first 10 samples as test
    y_test = torch.tensor(y[:10], dtype=torch.float32)
    
    # Analytical predictions
    y_pred_analytical = X_test @ mu_true
    mse_analytical = torch.mean((y_test - y_pred_analytical)**2)
    
    print(f"{'Model':<15} {'MSE':<12} {'Temperature':<12}")
    print("-" * 40)
    print(f"{'Analytical':<15} {mse_analytical:.6f} {'N/A':<12}")
    
    # Variational predictions for each model
    for model_name, model in models.items():
        y_pred_vi = X_test @ model.mu
        mse_vi = torch.mean((y_test - y_pred_vi)**2)
        print(f"{model_name + ' VI':<15} {mse_vi:.6f} {model.temperature:<12}")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
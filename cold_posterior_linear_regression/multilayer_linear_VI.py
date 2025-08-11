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
    Single-layer Bayesian Linear Regression with analytical solution
    Used as baseline comparison for multi-layer models
    """
    
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha  # precision of prior
        self.beta = beta    # precision of likelihood (inverse noise variance)
        
    def analytical_posterior(self, X, y):
        """Compute the analytical posterior for single-layer Bayesian linear regression"""
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        # Posterior covariance
        S_n_inv = self.alpha * torch.eye(X.shape[1]) + self.beta * X.T @ X
        S_n = torch.inverse(S_n_inv)
        
        # Posterior mean
        mu_n = self.beta * S_n @ X.T @ y
        
        return mu_n, S_n

class MultiLayerVariationalRegression(nn.Module):
    """
    Multi-Layer Variational Linear Regression (no activation functions)
    
    Each layer: z_l = W_l @ z_{l-1} + b_l
    Final output: y = z_L
    
    Priors: W_l ~ N(0, alpha_l^-1 * I), b_l ~ N(0, alpha_l^-1 * I)
    Variational: q(W_l) = N(mu_W_l, diag(sigma_W_l^2))
                q(b_l) = N(mu_b_l, diag(sigma_b_l^2))
    """
    
    def __init__(self, layer_dims, alpha=1.0, beta=1.0, temperature=1.0, posterior_type="standard"):
        super().__init__()
        self.layer_dims = layer_dims  # [input_dim, hidden_dim1, hidden_dim2, ..., output_dim]
        self.n_layers = len(layer_dims) - 1
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.posterior_type = posterior_type
        
        # Initialize variational parameters for each layer
        # Use nn.ParameterList to store parameters properly
        self.mu_W_layers = nn.ParameterList()
        self.log_sigma_W_layers = nn.ParameterList()
        self.mu_b_layers = nn.ParameterList()
        self.log_sigma_b_layers = nn.ParameterList()
        
        for i in range(self.n_layers):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            
            # Weight parameters
            self.mu_W_layers.append(nn.Parameter(torch.randn(out_dim, in_dim) * 0.1))
            self.log_sigma_W_layers.append(nn.Parameter(torch.randn(out_dim, in_dim) * 0.1))
            
            # Bias parameters  
            self.mu_b_layers.append(nn.Parameter(torch.randn(out_dim) * 0.1))
            self.log_sigma_b_layers.append(nn.Parameter(torch.randn(out_dim) * 0.1))
    
    def get_layer_params(self, layer_idx):
        """Get variational parameters for a specific layer"""
        mu_W = self.mu_W_layers[layer_idx]
        sigma_W = torch.exp(self.log_sigma_W_layers[layer_idx])
        mu_b = self.mu_b_layers[layer_idx]
        sigma_b = torch.exp(self.log_sigma_b_layers[layer_idx])
        return mu_W, sigma_W, mu_b, sigma_b
    
    def sample_layer_weights(self, layer_idx, n_samples=1):
        """Sample weights and biases for a specific layer"""
        mu_W, sigma_W, mu_b, sigma_b = self.get_layer_params(layer_idx)
        
        # Sample weights
        eps_W = torch.randn(n_samples, *mu_W.shape)
        W_samples = mu_W.unsqueeze(0) + sigma_W.unsqueeze(0) * eps_W
        
        # Sample biases
        eps_b = torch.randn(n_samples, *mu_b.shape)
        b_samples = mu_b.unsqueeze(0) + sigma_b.unsqueeze(0) * eps_b
        
        return W_samples, b_samples
    
    def forward_samples(self, X, n_samples=10):
        """
        Forward pass with sampled weights
        Returns: [n_samples, batch_size, output_dim]
        """
        X = torch.tensor(X, dtype=torch.float32)
        batch_size = X.shape[0]
        
        # Start with input
        z = X.unsqueeze(0).repeat(n_samples, 1, 1)  # [n_samples, batch_size, input_dim]
        
        # Forward through each layer
        for layer_idx in range(self.n_layers):
            W_samples, b_samples = self.sample_layer_weights(layer_idx, n_samples)
            
            # Linear transformation: z = z @ W^T + b
            z = torch.bmm(z, W_samples.transpose(1, 2)) + b_samples.unsqueeze(1)
        
        return z  # [n_samples, batch_size, output_dim]
    
    def kl_divergence(self):
        """
        Total KL divergence across all layers
        KL[q(θ) || p(θ)] = Σ_l KL[q(W_l, b_l) || p(W_l, b_l)]
        """
        total_kl = 0.0
        
        for layer_idx in range(self.n_layers):
            mu_W, sigma_W, mu_b, sigma_b = self.get_layer_params(layer_idx)
            
            # Effective alpha for cold posteriors
            if self.posterior_type == "cold":
                alpha_effective = self.alpha / self.temperature
            else:
                alpha_effective = self.alpha
            
            # KL for weights: KL[q(W) || p(W)]
            kl_W = 0.5 * (alpha_effective * (mu_W**2 + sigma_W**2) - 
                         2 * torch.log(sigma_W) - torch.log(torch.tensor(alpha_effective)) - 1)
            
            # KL for biases: KL[q(b) || p(b)]
            kl_b = 0.5 * (alpha_effective * (mu_b**2 + sigma_b**2) - 
                         2 * torch.log(sigma_b) - torch.log(torch.tensor(alpha_effective)) - 1)
            
            total_kl += kl_W.sum() + kl_b.sum()
        
        return total_kl
    
    def elbo(self, X, y, n_samples=10):
        """
        Evidence Lower Bound for multi-layer linear regression
        """
        y = torch.tensor(y, dtype=torch.float32)
        
        # Forward pass with sampled weights
        y_pred_samples = self.forward_samples(X, n_samples)  # [n_samples, batch_size, 1]
        y_pred_samples = y_pred_samples.squeeze(-1)  # [n_samples, batch_size]
        
        # Log likelihood: log p(y|X,θ) = -0.5 * beta * ||y - f(X,θ)||^2 + const
        log_likelihood = -0.5 * self.beta * ((y.unsqueeze(0) - y_pred_samples)**2).sum(dim=1)
        expected_log_likelihood = log_likelihood.mean()
        
        # Apply temperature scaling to likelihood
        if self.posterior_type == "cold":
            scaled_log_likelihood = (1.0 / self.temperature) * expected_log_likelihood
        elif self.posterior_type == "tempered":
            scaled_log_likelihood = (1.0 / self.temperature) * expected_log_likelihood
        else:
            scaled_log_likelihood = expected_log_likelihood
        
        # KL divergence
        kl_div = self.kl_divergence()
        
        # ELBO
        elbo = scaled_log_likelihood - kl_div
        
        return -elbo  # Return negative ELBO for minimization
    
    def predict_mean(self, X, n_samples=100):
        """Predict using the mean of the variational distribution"""
        with torch.no_grad():
            y_pred_samples = self.forward_samples(X, n_samples)
            return y_pred_samples.mean(dim=0).squeeze(-1)
    
    def get_total_parameters(self):
        """Get total number of parameters in the network"""
        total = 0
        for layer_idx in range(self.n_layers):
            in_dim, out_dim = self.layer_dims[layer_idx], self.layer_dims[layer_idx + 1]
            total += out_dim * in_dim + out_dim  # weights + biases
        return total

def generate_synthetic_data(n_samples=100, n_features=3, noise_std=0.1, seed=42):
    """Generate synthetic regression data with correlated features"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # True weights for single-layer model (for comparison)
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
    
    # Targets with noise (using single-layer model)
    y = X @ w_true + noise_std * torch.randn(n_samples)
    
    print(f"Generated data with feature correlation: {correlation}")
    print(f"Actual correlation: {np.corrcoef(features_corr.T)[0,1]:.3f}")
    
    return X.numpy(), y.numpy(), w_true.numpy()

def fit_multilayer_variational_regression(X, y, layer_dims, n_epochs=50000, lr=0.01, 
                                        alpha=1.0, beta=100.0, temperature=1.0, 
                                        posterior_type="standard", model_name="Multi-VI"):
    """Fit the multi-layer variational Bayesian linear regression"""
    
    model = MultiLayerVariationalRegression(layer_dims, alpha=alpha, beta=beta, 
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
        if epoch % 200 == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return model, losses

def compare_multilayer_posteriors(X, y, w_true, layer_dims, alpha=1.0, beta=100.0):
    """Compare analytical single-layer vs multi-layer variational posteriors"""
    
    print(f"\nNetwork Architecture: {' -> '.join(map(str, layer_dims))}")
    
    # Single-layer analytical baseline
    analytical = BayesianLinearRegression(alpha=alpha, beta=beta)
    mu_true, Sigma_true = analytical.analytical_posterior(X, y)
    
    # Multi-layer Standard VI
    ml_model_standard, ml_losses_standard = fit_multilayer_variational_regression(
        X, y, layer_dims, alpha=alpha, beta=beta, temperature=1.0, 
        posterior_type="standard", model_name="Multi-Layer Standard VI"
    )
    
    # Multi-layer Cold posterior VI (T=0.8)
    ml_model_cold, ml_losses_cold = fit_multilayer_variational_regression(
        X, y, layer_dims, alpha=alpha, beta=beta, temperature=0.8, 
        posterior_type="cold", model_name="Multi-Layer Cold VI (T=0.8)"
    )
    
    # Multi-layer Tempered posterior VI (λ=0.8)
    ml_model_tempered, ml_losses_tempered = fit_multilayer_variational_regression(
        X, y, layer_dims, alpha=alpha, beta=beta, temperature=0.8, 
        posterior_type="tempered", model_name="Multi-Layer Tempered VI (λ=0.8)"
    )
    
    models = {
        'Standard': ml_model_standard,
        'Cold': ml_model_cold,
        'Tempered': ml_model_tempered
    }
    
    losses_dict = {
        'Standard': ml_losses_standard,
        'Cold': ml_losses_cold,
        'Tempered': ml_losses_tempered
    }
    
    print("\n" + "="*80)
    print("MULTI-LAYER POSTERIOR COMPARISON - STANDARD vs COLD vs TEMPERED")
    print("="*80)
    print("ARCHITECTURE:")
    for i, model_name in enumerate(['Standard', 'Cold', 'Tempered']):
        model = models[model_name]
        print(f"  {model_name}: {model.get_total_parameters()} total parameters across {model.n_layers} layers")
    
    print("\nDEFINITIONS:")
    print("  Standard:  ELBO = E_q[log p(y|X,θ)] - KL[q(θ)||p(θ)]")
    print("  Cold:      ELBO = (1/T)E_q[log p(y|X,θ)] - KL[q(θ)||p_cold(θ)], p_cold has α/T")
    print("  Tempered:  ELBO = (1/λ)E_q[log p(y|X,θ)] - KL[q(θ)||p(θ)]")
    print("="*80)
    
    # Compare predictive performance
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Single-layer analytical prediction
    y_pred_analytical = X_tensor @ mu_true
    mse_analytical = torch.mean((y_tensor - y_pred_analytical)**2).item()
    
    print(f"PREDICTIVE PERFORMANCE (MSE on training data):")
    print(f"{'Model':<20} {'MSE':<12} {'Parameters':<12} {'Temperature':<12}")
    print("-" * 60)
    print(f"{'Single-layer Analytical':<20} {mse_analytical:<12.6f} {len(mu_true):<12} {'N/A':<12}")
    
    for model_name, model in models.items():
        y_pred = model.predict_mean(X, n_samples=100)
        mse = torch.mean((y_tensor - y_pred)**2).item()
        temp_str = f"T={model.temperature}" if model.posterior_type == "cold" else f"λ={model.temperature}" if model.posterior_type == "tempered" else "1.0"
        print(f"{f'Multi-layer {model_name}':<20} {mse:<12.6f} {model.get_total_parameters():<12} {temp_str:<12}")
    
    print("="*80)
    
    return mu_true, Sigma_true, models, losses_dict
    
    print("="*80)
    
    # Compare first layer parameters with analytical solution (if same input dimension)
    if layer_dims[0] == len(mu_true):
        print(f"\nFIRST LAYER WEIGHT COMPARISON (vs Analytical Single-Layer):")
        print(f"{'Parameter':<12} {'Analytical':<12} {'Standard':<12} {'Cold':<12} {'Tempered':<12}")
        print("-" * 65)
        
        param_names = ['Bias', 'Weight 1', 'Weight 2']
        sigma_true = np.sqrt(np.diag(Sigma_true.numpy()))
        
        for i in range(min(len(mu_true), layer_dims[1])):  # Compare up to first layer output size
            row = f"{param_names[i] if i < len(param_names) else f'W[{i}]':<12} {mu_true[i].item():.3f}±{sigma_true[i]:.3f}   "
            
            for model_name in ['Standard', 'Cold', 'Tempered']:
                model = models[model_name]
                # Get first layer weights corresponding to this input
                mu_W, sigma_W, mu_b, sigma_b = model.get_layer_params(0)
                
                if i == 0:  # Bias term
                    mu_val = mu_b[0].item() if len(mu_b) > 0 else 0.0
                    sigma_val = sigma_b[0].item() if len(sigma_b) > 0 else 0.0
                elif i-1 < mu_W.shape[1]:  # Weight term
                    mu_val = mu_W[0, i-1].item() if mu_W.shape[0] > 0 else 0.0
                    sigma_val = sigma_W[0, i-1].item() if sigma_W.shape[0] > 0 else 0.0
                else:
                    mu_val, sigma_val = 0.0, 0.0
                
                row += f"{mu_val:.3f}±{sigma_val:.3f}   "
            
            print(row)
    
    return mu_true, Sigma_true, models, losses_dict

def plot_multilayer_results(X, y, w_true, mu_true, Sigma_true, models, losses_dict, layer_dims):
    """Plot comparison results for multi-layer models"""
    
    # Create output directory
    output_dir = "figs/multilayer_cold_posterior_regression"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving figures to: {output_dir}")
    
    # Extract features (skip bias column)
    X_features = X[:, 1:]
    
    # 1. Training loss curves comparison
    plt.figure(figsize=(12, 8))
    
    colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
    for model_name, losses in losses_dict.items():
        model = models[model_name]
        
        if model.posterior_type == "cold":
            label = f'{model_name} (Cold T={model.temperature})'
        elif model.posterior_type == "tempered":
            label = f'{model_name} (Tempered λ={model.temperature})'
        else:
            label = f'{model_name} (Standard)'
            
        plt.plot(losses, linewidth=2, color=colors[model_name], 
                label=label, alpha=0.8)
    
    plt.yscale('log')
    plt.title(f'Multi-Layer Training Loss Comparison\nArchitecture: {" → ".join(map(str, layer_dims))}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Negative ELBO (log scale)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/multilayer_training_loss.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/multilayer_training_loss.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/multilayer_training_loss.png")
    plt.show()
    
    # 2. Test data KL analysis with PCA visualization
    print(f"\n{'='*60}")
    print("GENERATING TEST DATA FOR KL ANALYSIS")
    print(f"{'='*60}")
    
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
    
    # Compute KL divergences for all models vs analytical baseline
    all_test_kl = {}
    colors = {'Standard': 'red', 'Cold': 'purple', 'Tempered': 'orange'}
    
    for model_name, model in models.items():
        print(f"Computing test KL divergences for {model_name} VI...")
        
        test_kl_divergences = np.zeros(n_test)
        
        for i in tqdm(range(n_test), desc=f"Test KL for {model_name}"):
            x_point = X_test_tensor[i:i+1]
            
            # True posterior predictive (analytical single-layer)
            pred_mean_true = (x_point @ mu_true).item()
            pred_var_true = (x_point @ Sigma_true @ x_point.T).item() + 1.0/model.beta
            
            # Multi-layer variational posterior predictive
            with torch.no_grad():
                y_pred_samples = model.forward_samples(x_point.numpy(), n_samples=50)
                y_pred_samples = y_pred_samples.squeeze(-1)  # [n_samples, 1]
                
                pred_mean_vi = y_pred_samples.mean().item()
                pred_var_vi = y_pred_samples.var().item() + 1.0/model.beta
            
            # KL divergence between predictive distributions
            mean_diff_sq = (pred_mean_true - pred_mean_vi)**2
            kl_div = 0.5 * (
                np.log(pred_var_vi / pred_var_true) +
                pred_var_true / pred_var_vi +
                mean_diff_sq / pred_var_vi - 1
            )
            test_kl_divergences[i] = max(kl_div, 0)  # Ensure non-negative
        
        all_test_kl[model_name] = test_kl_divergences
        
        # Print test statistics
        print(f"\n{model_name} VI Test KL Statistics:")
        print(f"  Mean: {test_kl_divergences.mean():.6f}")
        print(f"  Max:  {test_kl_divergences.max():.6f}")
        print(f"  Min:  {test_kl_divergences.min():.6f}")
        print(f"  Std:  {test_kl_divergences.std():.6f}")
    
    # Create PC vs KL plots
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
    axes[0].set_ylabel('KL[Analytical || Multi-Layer VI]', fontsize=12)
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
    axes[1].set_ylabel('KL[Analytical || Multi-Layer VI]', fontsize=12)
    axes[1].set_title('Test KL Divergence vs PC2', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Layer KL Divergence vs Principal Components\n(Lower KL = Better Approximation to Analytical)', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/multilayer_kl_vs_pca_components.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/multilayer_kl_vs_pca_components.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/multilayer_kl_vs_pca_components.png")
    plt.show()
    
    # 3. Input data visualization with KL divergence backgrounds
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Create grid for KL divergence computation
    x1_min, x1_max = X_features[:, 0].min() - 0.5, X_features[:, 0].max() + 0.5
    x2_min, x2_max = X_features[:, 1].min() - 0.5, X_features[:, 1].max() + 0.5
    grid_resolution = 30  # Reduced for faster computation
    x1_grid = np.linspace(x1_min, x1_max, grid_resolution)
    x2_grid = np.linspace(x2_min, x2_max, grid_resolution)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    grid_points = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    n_grid_points = len(grid_points)
    X_grid = np.column_stack([np.ones(n_grid_points), grid_points])
    X_grid_tensor = torch.tensor(X_grid, dtype=torch.float32)
    
    print("Computing KL divergences for all models...")
    
    # Compute all KL divergences to determine global min/max
    all_kl_divergences = []
    
    for idx, (model_name, model) in enumerate(models.items()):
        print(f"Computing KL divergences for {model_name} VI...")
        
        kl_divergences = np.zeros(n_grid_points)
        
        for i in tqdm(range(n_grid_points), desc=f"KL for {model_name}"):
            x_point = X_grid_tensor[i:i+1]
            
            # True posterior predictive (analytical)
            pred_mean_true = (x_point @ mu_true).item()
            pred_var_true = (x_point @ Sigma_true @ x_point.T).item() + 1.0/model.beta
            
            # Multi-layer variational posterior predictive
            with torch.no_grad():
                y_pred_samples = model.forward_samples(x_point.numpy(), n_samples=30)
                y_pred_samples = y_pred_samples.squeeze(-1)  # [n_samples, 1]
                
                pred_mean_vi = y_pred_samples.mean().item()
                pred_var_vi = y_pred_samples.var().item() + 1.0/model.beta
            
            # KL divergence
            mean_diff_sq = (pred_mean_true - pred_mean_vi)**2
            kl_div = 0.5 * (
                np.log(pred_var_vi / pred_var_true) +
                pred_var_true / pred_var_vi +
                mean_diff_sq / pred_var_vi - 1
            )
            kl_divergences[i] = max(kl_div, 0)  # Ensure non-negative
        
        all_kl_divergences.append(kl_divergences)
    
    # Determine global min and max for consistent color scaling
    global_kl_min = min([kl.min() for kl in all_kl_divergences])
    global_kl_max = max([kl.max() for kl in all_kl_divergences])
    
    print(f"\nGlobal KL range: [{global_kl_min:.6f}, {global_kl_max:.6f}]")
    
    # Create plots with consistent color scale
    for idx, (model_name, model) in enumerate(models.items()):
        kl_divergences = all_kl_divergences[idx]
        
        # Plot for this model with consistent color scale
        KL_grid = kl_divergences.reshape(X1_grid.shape)
        im = axes[idx].imshow(KL_grid, extent=[x1_min, x1_max, x2_min, x2_max], 
                            origin='lower', cmap='Reds', alpha=0.7, aspect='auto',
                            vmin=global_kl_min, vmax=global_kl_max)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[idx], shrink=0.8)
        cbar.set_label('KL[Analytical || Multi-Layer VI]', fontsize=10)
        
        # Overlay data points
        scatter = axes[idx].scatter(X_features[:, 0], X_features[:, 1], c=y, s=30, 
                                  cmap='viridis', alpha=0.9, edgecolor='white', linewidth=0.8)
        
        axes[idx].set_xlabel('Feature 1', fontsize=11)
        axes[idx].set_ylabel('Feature 2', fontsize=11)
        
        if model.posterior_type == "cold":
            title = f'{model_name} VI\nCold (T={model.temperature})'
        elif model.posterior_type == "tempered":
            title = f'{model_name} VI\nTempered (λ={model.temperature})'
        else:
            title = f'{model_name} VI\nStandard'
            
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Print statistics
        print(f"\n{model_name} VI KL Statistics:")
        print(f"  Mean: {kl_divergences.mean():.6f}")
        print(f"  Max:  {kl_divergences.max():.6f}")
        print(f"  Std:  {kl_divergences.std():.6f}")
    
    plt.suptitle(f'Multi-Layer KL Divergence Heatmaps vs Analytical Single-Layer\nArchitecture: {" → ".join(map(str, layer_dims))}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multilayer_kl_divergence_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/multilayer_kl_divergence_heatmaps.pdf", bbox_inches='tight')
    print(f"Saved: {output_dir}/multilayer_kl_divergence_heatmaps.png")
    plt.show()
    
    print(f"\nAll figures saved to: {output_dir}")
    print(f"Files saved:")
    print(f"  - multilayer_training_loss.png/.pdf")
    print(f"  - multilayer_kl_vs_pca_components.png/.pdf") 
    print(f"  - multilayer_kl_divergence_heatmaps.png/.pdf")
    
    # Print summary
    print(f"\n{'='*70}")
    print("MULTI-LAYER MODEL SUMMARY")
    print(f"{'='*70}")
    print(f"Architecture: {' → '.join(map(str, layer_dims))}")
    print(f"Total layers: {len(layer_dims) - 1}")
    
    for model_name, model in models.items():
        print(f"\n{model_name} Model:")
        print(f"  Parameters: {model.get_total_parameters()}")
        print(f"  Temperature: {model.temperature}")
        print(f"  Type: {model.posterior_type}")
        print(f"  Final Loss: {losses_dict[model_name][-1]:.6f}")
    
    print(f"\nFiles saved to: {output_dir}")
    print(f"  - multilayer_training_loss.png/.pdf")
    print(f"  - multilayer_predictions_uncertainties.png/.pdf") 
    print(f"  - first_layer_parameters.png/.pdf")
    print(f"{'='*70}")

def main():
    """Main execution function"""
    print("Multi-Layer Bayesian Linear Regression: Standard vs Cold vs Tempered VI")
    print("="*75)
    
    # Generate synthetic data
    X, y, w_true = generate_synthetic_data(n_samples=100, n_features=3, noise_std=0.1)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"True weights (single-layer): {w_true}")
    
    # Define multi-layer architecture
    # Input: 3 features (including bias) → 8 → 8 → 8 → 8 → Output: 1
    layer_dims = [3, 8, 8, 8, 8, 1]
    
    # Hyperparameters
    alpha = 1.0   # Prior precision
    beta = 100.0  # Likelihood precision
    
    # Compare multi-layer posteriors
    mu_true, Sigma_true, models, losses_dict = compare_multilayer_posteriors(
        X, y, w_true, layer_dims, alpha=alpha, beta=beta
    )
    
    # Plot only the requested figures
    plot_multilayer_results(X, y, w_true, mu_true, Sigma_true, models, losses_dict, layer_dims)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print("Generated figures:")
    print("  1. Training loss comparison")
    print("  2. KL divergence vs PCA components")  
    print("  3. KL divergence heatmaps over input space")
    print(f"All saved to: figs/cold_posterior_linear_regression/")

if __name__ == "__main__":
    main()
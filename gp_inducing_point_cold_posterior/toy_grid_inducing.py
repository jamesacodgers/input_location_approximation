def plot_training_losses(results):
    """Plot training loss curves for all models to verify convergence."""
    
    # Filter out exact GP and get sparse/grid results
    sparse_results = {k: v for k, v in results.items() 
                     if k != 'exact' and 'sparse' in v.get('inducing_config', '')}
    grid_results = {k: v for k, v in results.items() 
                   if k != 'exact' and 'grid' in v.get('inducing_config', '')}
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Define colors for different temperatures
    temp_colors = {0.01: 'blue', 0.1: 'green', 1.0: 'red', 10.0: 'purple'}
    
    # Plot 1: Sparse Mean-Field
    ax = axes[0, 0]
    for key, result in sparse_results.items():
        if result['variational_family'] == 'mean_field':
            temp = result['temperature']
            color = temp_colors.get(temp, 'black')
            ax.plot(result['losses'], color=color, linewidth=2, 
                   label=f'T={temp}', alpha=0.8)
    
    ax.set_title('1D Inducing + Mean-Field')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (ELBO)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Sparse Cholesky
    ax = axes[0, 1]
    for key, result in sparse_results.items():
        if result['variational_family'] == 'cholesky':
            temp = result['temperature']
            color = temp_colors.get(temp, 'black')
            ax.plot(result['losses'], color=color, linewidth=2, 
                   label=f'T={temp}', alpha=0.8)
    
    ax.set_title('1D Inducing + Cholesky')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (ELBO)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Grid Mean-Field
    ax = axes[1, 0]
    for key, result in grid_results.items():
        if result['variational_family'] == 'mean_field':
            temp = result['temperature']
            color = temp_colors.get(temp, 'black')
            ax.plot(result['losses'], color=color, linewidth=2, 
                   label=f'T={temp}', alpha=0.8)
    
    ax.set_title('2D Inducing + Mean-Field')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (ELBO)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Grid Cholesky
    ax = axes[1, 1]
    for key, result in grid_results.items():
        if result['variational_family'] == 'cholesky':
            temp = result['temperature']
            color = temp_colors.get(temp, 'black')
            ax.plot(result['losses'], color=color, linewidth=2, 
                   label=f'T={temp}', alpha=0.8)
    
    ax.set_title('2D Inducing + Cholesky')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (ELBO)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution, VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood
from itertools import product

class ColdPosteriorSparseGP(ApproximateGP):
    """
    Sparse GP with temperature scaling and configurable variational family.
    """
    
    def __init__(self, inducing_points, temperature=1.0, variational_family='cholesky'):
        """
        Args:
            inducing_points: Tensor of inducing point locations [M, D]
            temperature: Temperature parameter (T=1 standard, T<1 cold)
            variational_family: 'cholesky' (full rank) or 'mean_field'
        """
        # Initialize variational distribution based on family
        if variational_family.lower() == 'mean_field':
            variational_distribution = MeanFieldVariationalDistribution(
                inducing_points.size(0)
            )
        elif variational_family.lower() == 'cholesky':
            variational_distribution = CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
        else:
            raise ValueError("variational_family must be 'cholesky' or 'mean_field'")
        
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=False
        )
        
        super().__init__(variational_strategy)
        
        # GP components
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        
        self.temperature = torch.tensor(temperature)
        self.variational_family = variational_family
    
    def forward(self, x):
        """Forward pass through the GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class ColdPosteriorExactGP(ExactGP):
    """
    Exact GP with temperature scaling for cold posterior effect research.
    """
    
    def __init__(self, train_x, train_y, likelihood, temperature=1.0):
        super().__init__(train_x, train_y, likelihood)
        
        # GP components
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        
        # Temperature parameter
        self.temperature = temperature
    
    def forward(self, x):
        """Forward pass through the GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class ColdPosteriorELBO(VariationalELBO):
    """
    Modified ELBO with temperature scaling for cold posterior effect.
    """
    
    def __init__(self, likelihood, model, num_data, temperature=None):
        super().__init__(likelihood, model, num_data)
        self.model = model
        self.temperature = temperature
        
    def forward(self, variational_dist_f, target, **kwargs):
        """
        Compute temperature-scaled ELBO.
        """
        # Standard ELBO computation
        log_likelihood = self.likelihood.expected_log_prob(target, variational_dist_f).sum()
        kl_divergence = self.model.variational_strategy.kl_divergence().sum()
        
        # Get temperature from model if not provided
        temp = self.temperature if self.temperature is not None else self.model.temperature
        
        # Temperature scaling: likelihood^(1/T), KL unchanged
        scaled_log_likelihood = log_likelihood / temp
        
        # ELBO = E[log p(y|f)] - KL[q(f)||p(f)]
        # For cold posterior: E[log p(y|f)^(1/T)] - KL[q(f)||p(f)]
        elbo = scaled_log_likelihood - kl_divergence
        
        return elbo

def set_kernel_hyperparameters(model, lengthscale, outputscale):
    """Fix kernel hyperparameters to known values."""
    model.covar_module.outputscale = outputscale
    model.covar_module.base_kernel.lengthscale = lengthscale
    # Make them non-trainable
    model.covar_module.raw_outputscale.requires_grad = False
    model.covar_module.base_kernel.raw_lengthscale.requires_grad = False

def set_likelihood_noise(likelihood, noise_var):
    """Fix likelihood noise to known value."""
    likelihood.noise = noise_var
    likelihood.raw_noise.requires_grad = False

def create_synthetic_data(n_train=4, n_test=100, n_test_per_dim=30, noise_std=0.1, seed=42):
    """Generate synthetic 2D regression data with training and testing data on x2=0 line."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Training data - sparse points on the x2=0 line (1D subspace)
    train_x1 = torch.linspace(0.1, 0.9, n_train)  # Avoid boundaries for better uncertainty
    train_x2 = torch.zeros(n_train)  # All training data at x2=0
    train_x = torch.stack([train_x1, train_x2], dim=1)
    
    # Test data - dense points on the x2=0 line for evaluation
    test_x1 = torch.linspace(0, 1, n_test)
    test_x2 = torch.zeros(n_test)  # All test data at x2=0
    test_x = torch.stack([test_x1, test_x2], dim=1)
    
    # 2D grid for visualization - SYMMETRIC around x2=0
    x1_vis = torch.linspace(0, 1, n_test_per_dim)
    # Make visualization symmetric around x2=0
    x2_extent = 0.8  # Total extent from -0.8 to +0.8
    x2_vis = torch.linspace(-x2_extent, x2_extent, n_test_per_dim)
    X1, X2 = torch.meshgrid(x1_vis, x2_vis, indexing='ij')
    vis_x = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    
    # Combine points where we need function values (train + test + visualization)
    all_x = torch.cat([train_x, test_x, vis_x], dim=0)
    
    # True GP hyperparameters (these will be fixed in all models)
    TRUE_LENGTHSCALE = 0.3
    TRUE_OUTPUTSCALE = 1.0
    TRUE_NOISE = noise_std**2
    
    # Create GP kernel with true hyperparameters
    true_kernel = ScaleKernel(RBFKernel())
    true_kernel.outputscale = TRUE_OUTPUTSCALE
    true_kernel.base_kernel.lengthscale = TRUE_LENGTHSCALE
    
    # Generate function values by sampling from GP prior
    with torch.no_grad():
        # Compute kernel matrix for all points
        K = true_kernel(all_x).evaluate()
        
        # Add jitter for numerical stability
        jitter = 1e-4
        K_jittered = K + jitter * torch.eye(K.size(0))
        
        try:
            # Use Cholesky decomposition for stable sampling
            L = torch.linalg.cholesky(K_jittered)
            # Sample from standard normal and transform
            z = torch.randn(K.size(0))
            f_values = L @ z
        except torch.linalg.LinAlgError:
            print("Cholesky decomposition failed, using eigendecomposition fallback...")
            # Fallback: use eigendecomposition
            eigenvals, eigenvecs = torch.linalg.eigh(K_jittered)
            # Ensure all eigenvalues are positive
            eigenvals = torch.clamp(eigenvals, min=1e-6)
            
            # Sample using eigendecomposition: f = V * sqrt(Λ) * z
            z = torch.randn(K.size(0))
            f_values = eigenvecs @ (torch.sqrt(eigenvals) * z)
    
    # Split back into train, test, and visualization
    train_f = f_values[:n_train]
    test_f = f_values[n_train:n_train+n_test]
    vis_f = f_values[n_train+n_test:]
    
    # Add noise to training data (test data remains clean for evaluation)
    train_y = train_f 
    test_y = test_f  # True function values (no noise)
    
    print(f"True GP hyperparameters:")
    print(f"  Lengthscale: {TRUE_LENGTHSCALE}")
    print(f"  Outputscale: {TRUE_OUTPUTSCALE}")
    print(f"  Noise: {TRUE_NOISE}")
    print(f"Training points: {n_train} (sparse for high uncertainty)")
    print(f"Visualization grid: x2 ∈ [{-x2_extent:.1f}, {x2_extent:.1f}] (symmetric)")
    
    return train_x, train_y, test_x, test_y, vis_x, (X1, X2), (TRUE_LENGTHSCALE, TRUE_OUTPUTSCALE, TRUE_NOISE)

def create_inducing_points_sparse(num_inducing=20, x2_offset=0.0, x1_range=(0, 1)):
    """
    Create sparse inducing points along a line with specified x2 offset.
    """
    x1_coords = torch.linspace(x1_range[0], x1_range[1], num_inducing)
    x2_coords = torch.full((num_inducing,), x2_offset)
    inducing_points = torch.stack([x1_coords, x2_coords], dim=1)
    return inducing_points

def create_inducing_points_grid(num_x1=8, num_x2=8, x1_range=(0, 1), x2_extent=0.6):
    """
    Create a dense grid of inducing points over both dimensions.
    Grid is symmetric around x2=0 with one line of points exactly at x2=0.
    
    Args:
        num_x1: Number of points in x1 direction
        num_x2: Number of points in x2 direction (should be odd for symmetry)
        x1_range: Range for x1 coordinates
        x2_extent: Half-width of grid in x2 direction (grid spans [-x2_extent, +x2_extent])
    """
    x1_coords = torch.linspace(x1_range[0], x1_range[1], num_x1)
    
    # Create symmetric x2 coordinates with one point exactly at x2=0
    if num_x2 % 2 == 0:
        print(f"Warning: num_x2={num_x2} is even. Making it odd ({num_x2+1}) for symmetry around x2=0")
        num_x2 = num_x2 + 1
    
    # Create symmetric points around x2=0
    # For odd num_x2, this ensures one point is exactly at x2=0
    x2_coords = torch.linspace(-x2_extent, x2_extent, num_x2)
    
    # Verify that there's a point at exactly x2=0
    center_idx = num_x2 // 2
    x2_coords[center_idx] = 0.0  # Ensure exact zero (handle floating point precision)
    
    # Create meshgrid and flatten
    X1, X2 = torch.meshgrid(x1_coords, x2_coords, indexing='ij')
    inducing_points = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    
    # Verify we have points at x2=0
    points_at_zero = torch.sum(torch.abs(inducing_points[:, 1]) < 1e-6).item()
    print(f"Grid created: {len(inducing_points)} points total, {points_at_zero} points at x2=0")
    print(f"x2 range: [{x2_coords.min():.3f}, {x2_coords.max():.3f}]")
    
    return inducing_points

def train_cold_posterior_gp_with_inducing(train_x, train_y, inducing_points, true_hyperparams, 
                                         temperature=1.0, variational_family='cholesky',
                                         lr=0.01, epochs=500, verbose=True):
    """Train sparse GP with configurable variational family."""
    lengthscale, outputscale, noise_var = true_hyperparams
    
    # Initialize model and likelihood
    model = ColdPosteriorSparseGP(inducing_points, temperature=temperature, 
                                  variational_family=variational_family)
    likelihood = GaussianLikelihood()
    
    # Fix hyperparameters to true values
    set_kernel_hyperparameters(model, lengthscale, outputscale)
    set_likelihood_noise(likelihood, noise_var)
    
    # DEBUG: Verify parameters are fixed
    print(f"  Likelihood noise fixed: {not likelihood.raw_noise.requires_grad}")
    print(f"  Kernel lengthscale fixed: {not model.covar_module.base_kernel.raw_lengthscale.requires_grad}")
    print(f"  Kernel outputscale fixed: {not model.covar_module.raw_outputscale.requires_grad}")
    
    # Count trainable parameters
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable_count}/{total_count}")
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Optimizer - only optimize variational parameters (likelihood is fixed)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    # Loss function (temperature-scaled ELBO)
    mll = ColdPosteriorELBO(likelihood, model, num_data=train_x.size(0))
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(train_x)
        loss = -mll(output, train_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
                  f'Temperature: {model.temperature.item():.4f}, '
                  f'Var Family: {model.variational_family}')
    
    return model, likelihood, losses

def train_exact_gp(train_x, train_y, true_hyperparams, temperature=1.0, lr=0.01, epochs=500, verbose=True):
    """Train exact GP with fixed hyperparameters."""
    lengthscale, outputscale, noise_var = true_hyperparams
    
    # Initialize model and likelihood
    likelihood = GaussianLikelihood()
    model = ColdPosteriorExactGP(train_x, train_y, likelihood, temperature=temperature)
    
    # Fix hyperparameters to true values
    set_kernel_hyperparameters(model, lengthscale, outputscale)
    set_likelihood_noise(likelihood, noise_var)
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Optimizer - only optimize mean (hyperparameters are fixed)
    trainable_params = [p for p in list(model.parameters()) + list(likelihood.parameters()) 
                       if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    # Loss function - use standard ExactMarginalLogLikelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(train_x)
        loss = -mll(output, train_y)
        
        # Apply temperature scaling manually
        loss = loss / model.temperature
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return model, likelihood, losses

def evaluate_model(model, likelihood, test_x):
    """Evaluate trained model on test data."""
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Get predictive distribution
        f_pred = model(test_x)
        y_pred = likelihood(f_pred)
        
        # Extract mean and variance
        pred_mean = y_pred.mean
        pred_var = y_pred.variance
        pred_std = pred_var.sqrt()
        
        # Get lower and upper bounds (95% confidence)
        lower = pred_mean - 1.96 * pred_std
        upper = pred_mean + 1.96 * pred_std
        
    return pred_mean, pred_std, lower, upper

def compute_kl_divergence(mean1, std1, mean2, std2):
    """
    Compute KL divergence between two univariate Gaussians.
    """
    var1, var2 = std1**2, std2**2
    kl = torch.log(std2 / std1) + (var1 + (mean1 - mean2)**2) / (2 * var2) - 0.5
    return kl

def run_variational_family_experiment(train_x, train_y, test_x, test_y, true_hyperparams,
                                     temperatures=[0.1, 1.0], 
                                     variational_families=['mean_field', 'cholesky'],
                                     inducing_configs=None,
                                     epochs=300, verbose=False):
    """
    Run experiment comparing different variational families and inducing point configurations.
    
    Args:
        inducing_configs: List of dicts with keys 'type', 'name', and config parameters
    """
    
    results = {}
    
    # Train exact GP as reference
    print("Training exact GP (T=1.0) as reference...")
    exact_model, exact_likelihood, _ = train_exact_gp(
        train_x, train_y, true_hyperparams, temperature=1.0, epochs=epochs, verbose=verbose
    )
    exact_pred_mean, exact_pred_std, _, _ = evaluate_model(exact_model, exact_likelihood, test_x)
    
    # Compute exact GP metrics
    exact_mse = torch.mean((exact_pred_mean - test_y)**2).item()
    exact_nll = -torch.distributions.Normal(exact_pred_mean, exact_pred_std).log_prob(test_y).mean().item()
    
    print(f"Exact GP - MSE: {exact_mse:.4f}, NLL: {exact_nll:.4f}")
    
    # Store exact GP results
    results['exact'] = {
        'model': exact_model,
        'likelihood': exact_likelihood,
        'pred_mean': exact_pred_mean,
        'pred_std': exact_pred_std,
        'kl_from_exact': torch.zeros_like(exact_pred_mean),
        'mean_kl': 0.0,
        'mse': exact_mse,
        'nll': exact_nll,
        'final_temp': 1.0,
        'variational_family': 'exact',
        'inducing_config': 'exact',
        'num_inducing': len(train_x)
    }
    
    # Run sparse GP experiments
    for inducing_config in inducing_configs:
        # Create inducing points based on configuration
        if inducing_config['type'] == 'sparse':
            inducing_points = create_inducing_points_sparse(
                num_inducing=inducing_config.get('num_inducing', 20),
                x2_offset=inducing_config.get('x2_offset', 0.0),
                x1_range=inducing_config.get('x1_range', (0, 1))
            )
        elif inducing_config['type'] == 'grid':
            inducing_points = create_inducing_points_grid(
                num_x1=inducing_config.get('num_x1', 8),
                num_x2=inducing_config.get('num_x2', 9),  # Default to odd number for symmetry
                x1_range=inducing_config.get('x1_range', (0, 1)),
                x2_extent=inducing_config.get('x2_extent', 0.6)  # Symmetric extent around x2=0
            )
        else:
            raise ValueError(f"Unknown inducing config type: {inducing_config['type']}")
        
        print(f"\n{inducing_config['name']}: {len(inducing_points)} inducing points")
        
        for var_family in variational_families:
            for temp in temperatures:
                key = f"{inducing_config['name']}_{var_family}_T={temp}"
                print(f"Training {key}")
                
                try:
                    model, likelihood, losses = train_cold_posterior_gp_with_inducing(
                        train_x, train_y, inducing_points, true_hyperparams, 
                        temperature=temp, variational_family=var_family,
                        epochs=epochs, verbose=verbose
                    )
                    
                    # Evaluate
                    pred_mean, pred_std, lower, upper = evaluate_model(model, likelihood, test_x)
                    
                    # Compute metrics
                    mse = torch.mean((pred_mean - test_y)**2).item()
                    nll = -torch.distributions.Normal(pred_mean, pred_std).log_prob(test_y).mean().item()
                    
                    # DEBUG: Print some diagnostics
                    print(f"  Pred mean range: [{pred_mean.min().item():.3f}, {pred_mean.max().item():.3f}]")
                    print(f"  Pred std range: [{pred_std.min().item():.3f}, {pred_std.max().item():.3f}]")
                    print(f"  MSE: {mse:.6f}, NLL: {nll:.4f}")
                    
                    # Compute KL divergence from exact GP
                    kl_from_exact = compute_kl_divergence(pred_mean, pred_std, exact_pred_mean, exact_pred_std)
                    mean_kl = torch.mean(kl_from_exact).item()
                    
                    results[key] = {
                        'model': model,
                        'likelihood': likelihood,
                        'pred_mean': pred_mean,
                        'pred_std': pred_std,
                        'lower': lower,
                        'upper': upper,
                        'losses': losses,
                        'mse': mse,
                        'nll': nll,
                        'final_temp': model.temperature.item(),
                        'kl_from_exact': kl_from_exact,
                        'mean_kl': mean_kl,
                        'temperature': temp,
                        'variational_family': var_family,
                        'inducing_config': inducing_config['name'],
                        'inducing_points': inducing_points,
                        'num_inducing': len(inducing_points)
                    }
                    
                    print(f"  MSE: {mse:.4f}, NLL: {nll:.4f}, Mean KL: {mean_kl:.4f}")
                    
                except Exception as e:
                    print(f"  Failed: {str(e)}")
                    continue
    
    return results

def plot_variational_comparison(train_x, train_y, test_x, test_y, vis_x, vis_grid, results):
    """Plot comparison of different variational families and inducing point configurations."""
    
    # Filter results by configuration type
    sparse_results = {k: v for k, v in results.items() 
                     if k != 'exact' and 'sparse' in v.get('inducing_config', '')}
    grid_results = {k: v for k, v in results.items() 
                   if k != 'exact' and 'grid' in v.get('inducing_config', '')}
    
    X1, X2 = vis_grid
    
    # Get exact GP predictions for reference
    exact_result = results['exact']
    exact_vis_mean, _, _, _ = evaluate_model(exact_result['model'], exact_result['likelihood'], vis_x)
    exact_mean_2d = exact_vis_mean.reshape(X1.shape)
    
    # Create figure comparing configurations - REDUCED SIZE
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Exact GP reference
    ax = axes[0, 0]
    im = ax.contourf(X1.numpy(), X2.numpy(), exact_mean_2d.numpy(), 
                    levels=20, cmap='viridis')
    ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
               c='red', s=20, marker='o', label='Train')
    ax.set_title('Exact GP', fontsize=12)
    ax.legend()
    plt.colorbar(im, ax=ax, shrink=0.6)
    
    # Turn off other subplots in first row
    for i in range(1, 4):
        axes[0, i].axis('off')
    
    # Row 2: Sparse inducing points (x2=0)
    sparse_keys = list(sparse_results.keys())
    for i, key in enumerate(sparse_keys[:4]):  # Show up to 4
        if i >= 4:
            break
            
        result = sparse_results[key]
        
        # Get predictions on visualization grid
        vis_pred_mean, _, _, _ = evaluate_model(result['model'], result['likelihood'], vis_x)
        pred_mean_2d = vis_pred_mean.reshape(X1.shape)
        
        ax = axes[1, i]
        vmin, vmax = exact_mean_2d.min().item(), exact_mean_2d.max().item()
        im = ax.contourf(X1.numpy(), X2.numpy(), pred_mean_2d.numpy(), 
                        levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Plot training data and inducing points
        ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
                   c='red', s=20, marker='o', alpha=0.8)
        inducing_points = result['inducing_points']
        ax.scatter(inducing_points[:, 0].numpy(), inducing_points[:, 1].numpy(),
                   c='white', s=15, marker='s', linewidth=0.5, 
                   edgecolors='black', alpha=0.9)
        
        # Extract configuration details for title
        var_family = result['variational_family']
        temp = result['temperature']
        nll = result['nll']
        
        ax.set_title(f'1D Inducing: {var_family}\nT={temp}, NLL={nll:.2f}', fontsize=10)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    # Row 3: Grid inducing points
    grid_keys = list(grid_results.keys())
    for i, key in enumerate(grid_keys[:4]):  # Show up to 4
        if i >= 4:
            break
            
        result = grid_results[key]
        
        # Get predictions on visualization grid
        vis_pred_mean, _, _, _ = evaluate_model(result['model'], result['likelihood'], vis_x)
        pred_mean_2d = vis_pred_mean.reshape(X1.shape)
        
        ax = axes[2, i]
        vmin, vmax = exact_mean_2d.min().item(), exact_mean_2d.max().item()
        im = ax.contourf(X1.numpy(), X2.numpy(), pred_mean_2d.numpy(), 
                        levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Plot training data and inducing points
        ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
                   c='red', s=20, marker='o', alpha=0.8)
        inducing_points = result['inducing_points']
        ax.scatter(inducing_points[:, 0].numpy(), inducing_points[:, 1].numpy(),
                   c='white', s=8, marker='s', linewidth=0.3, 
                   edgecolors='black', alpha=0.7)
        
        # Extract configuration details for title
        var_family = result['variational_family']
        temp = result['temperature']
        nll = result['nll']
        
        ax.set_title(f'2D Inducing: {var_family}\nT={temp}, NLL={nll:.2f}', fontsize=10)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.colorbar(im, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    return fig

def plot_metrics_comparison(results):
    """Plot detailed metrics comparison."""
    
    # Separate results by inducing configuration
    sparse_results = {k: v for k, v in results.items() 
                     if k != 'exact' and 'sparse' in v.get('inducing_config', '')}
    grid_results = {k: v for k, v in results.items() 
                   if k != 'exact' and 'grid' in v.get('inducing_config', '')}
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Extract data for plotting
    def extract_metrics(results_dict):
        var_families = []
        temperatures = []
        nlls = []
        mses = []
        kls = []
        
        for key, result in results_dict.items():
            var_families.append(result['variational_family'])
            temperatures.append(result['temperature'])
            nlls.append(result['nll'])
            mses.append(result['mse'])
            kls.append(result['mean_kl'])
        
        return var_families, temperatures, nlls, mses, kls
    
    # Sparse results
    if sparse_results:
        var_fam_s, temps_s, nlls_s, mses_s, kls_s = extract_metrics(sparse_results)
        
        # Group by variational family
        unique_families = list(set(var_fam_s))
        unique_temps = sorted(list(set(temps_s)))
        
        for i, family in enumerate(unique_families):
            family_mask = np.array(var_fam_s) == family
            family_temps = np.array(temps_s)[family_mask]
            family_nlls = np.array(nlls_s)[family_mask]
            family_mses = np.array(mses_s)[family_mask]
            family_kls = np.array(kls_s)[family_mask]
            
            # Sort by temperature
            sort_idx = np.argsort(family_temps)
            family_temps = family_temps[sort_idx]
            family_nlls = family_nlls[sort_idx]
            family_mses = family_mses[sort_idx]
            family_kls = family_kls[sort_idx]
            
            color = ['blue', 'red'][i % 2]
            
            axes[0, 0].plot(family_temps, family_nlls, 'o-', color=color, 
                           label=f'Sparse {family}', linewidth=2, markersize=8)
            axes[0, 1].plot(family_temps, family_mses, 's-', color=color, 
                           label=f'Sparse {family}', linewidth=2, markersize=8)
            axes[0, 2].plot(family_temps, family_kls, '^-', color=color, 
                           label=f'Sparse {family}', linewidth=2, markersize=8)
    
    # Grid results
    if grid_results:
        var_fam_g, temps_g, nlls_g, mses_g, kls_g = extract_metrics(grid_results)
        
        # Group by variational family
        unique_families = list(set(var_fam_g))
        
        for i, family in enumerate(unique_families):
            family_mask = np.array(var_fam_g) == family
            family_temps = np.array(temps_g)[family_mask]
            family_nlls = np.array(nlls_g)[family_mask]
            family_mses = np.array(mses_g)[family_mask]
            family_kls = np.array(kls_g)[family_mask]
            
            # Sort by temperature
            sort_idx = np.argsort(family_temps)
            family_temps = family_temps[sort_idx]
            family_nlls = family_nlls[sort_idx]
            family_mses = family_mses[sort_idx]
            family_kls = family_kls[sort_idx]
            
            color = ['darkblue', 'darkred'][i % 2]
            
            axes[1, 0].plot(family_temps, family_nlls, 'o-', color=color, 
                           label=f'Grid {family}', linewidth=2, markersize=8)
            axes[1, 1].plot(family_temps, family_mses, 's-', color=color, 
                           label=f'Grid {family}', linewidth=2, markersize=8)
            axes[1, 2].plot(family_temps, family_kls, '^-', color=color, 
                           label=f'Grid {family}', linewidth=2, markersize=8)
    
    # Add exact GP reference lines
    exact_result = results['exact']
    exact_nll = exact_result['nll']
    exact_mse = exact_result['mse']
    
    for i in range(2):
        axes[i, 0].axhline(y=exact_nll, color='black', linestyle='--', 
                          alpha=0.7, linewidth=2, label='Exact GP' if i == 0 else "")
        axes[i, 1].axhline(y=exact_mse, color='black', linestyle='--', 
                          alpha=0.7, linewidth=2, label='Exact GP' if i == 0 else "")
    
    # Configure axes
    for i in range(2):
        axes[i, 0].set_xlabel('Temperature')
        axes[i, 0].set_ylabel('NLL')
        axes[i, 0].set_title(f'{"1D Inducing" if i == 0 else "2D Inducing"}: NLL vs Temperature')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xscale('log')
        
        axes[i, 1].set_xlabel('Temperature')
        axes[i, 1].set_ylabel('MSE')
        axes[i, 1].set_title(f'{"1D Inducing" if i == 0 else "2D Inducing"}: MSE vs Temperature')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xscale('log')
        
        axes[i, 2].set_xlabel('Temperature')
        axes[i, 2].set_ylabel('Mean KL from Exact GP')
        axes[i, 2].set_title(f'{"1D Inducing" if i == 0 else "2D Inducing"}: KL Divergence')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
        axes[i, 2].set_xscale('log')
    
    plt.tight_layout()
    return fig

def plot_1d_slice_comparison(train_x, train_y, test_x, test_y, results):
    """Plot 1D slice comparison along x2=0 line."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Get exact GP prediction
    exact_result = results['exact']
    
    # Plot exact GP in all subplots as reference
    for ax in axes:
        ax.plot(test_x[:, 0].numpy(), exact_result['pred_mean'].numpy(), 
                'k-', linewidth=3, label='Exact GP', alpha=0.8)
        ax.fill_between(test_x[:, 0].numpy(), 
                       (exact_result['pred_mean'] - 1.96 * exact_result['pred_std']).numpy(),
                       (exact_result['pred_mean'] + 1.96 * exact_result['pred_std']).numpy(),
                       color='gray', alpha=0.2)
    
    # Group results by configuration type and variational family
    plot_configs = [
        ('sparse', 'mean_field', 0, '1D + Mean Field'),
        ('sparse', 'cholesky', 1, '1D + Cholesky'), 
        ('grid', 'mean_field', 2, '2D + Mean Field'),
        ('grid', 'cholesky', 3, '2D + Cholesky')
    ]
    
    colors = ['blue', 'red', 'green', 'purple']
    
    for config_type, var_family, ax_idx, title in plot_configs:
        if ax_idx >= len(axes):
            continue
            
        ax = axes[ax_idx]
        
        # Find matching results
        matching_keys = [k for k, v in results.items() 
                        if k != 'exact' and 
                        config_type in v.get('inducing_config', '') and
                        v.get('variational_family', '') == var_family]
        
        # Plot each temperature
        for i, key in enumerate(matching_keys):
            if key not in results:
                continue
                
            result = results[key]
            temp = result['temperature']
            color = colors[i % len(colors)]
            
            # Plot mean prediction
            ax.plot(test_x[:, 0].numpy(), result['pred_mean'].numpy(), 
                   color=color, linewidth=2, label=f'T={temp}', alpha=0.8)
            
            # Plot confidence intervals
            ax.fill_between(test_x[:, 0].numpy(), 
                           result['lower'].numpy(),
                           result['upper'].numpy(),
                           color=color, alpha=0.2)
        
        # Plot training data
        ax.scatter(train_x[:, 0].numpy(), train_y.numpy(), 
                  c='red', s=30, marker='o', alpha=0.8, zorder=5, label='Train')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('Function Value')
        ax.set_title(f'{title}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    return fig

def run_variational_experiment():
    """Run the complete variational family experiment."""
    
    print("Cold Posterior Variational Family Experiment")
    print("=" * 60)
    
    # Generate synthetic data with fewer training points for higher uncertainty
    train_x, train_y, test_x, test_y, vis_x, vis_grid, true_hyperparams = create_synthetic_data(
        n_train=4, n_test=200, n_test_per_dim=25, noise_std=0.3, seed=42  # Increased noise too
    )
    
    print(f"Training data shape: {train_x.shape}")
    print(f"Test data shape: {test_x.shape}")
    
    # Define experimental configurations
    temperatures = [0.01, 0.1, 1.0]
    variational_families = ['mean_field', 'cholesky']
    
    # Define inducing point configurations
    inducing_configs = [
        {
            'type': 'sparse',
            'name': 'sparse_x2=0',
            'num_inducing': 25,
            'x2_offset': 0.0,
            'x1_range': (0, 1)
        },
        {
            'type': 'grid', 
            'name': 'grid_2d',
            'num_x1': 9,
            'num_x2': 9,  # Odd number ensures symmetry around x2=0
            'x1_range': (0, 1),
            'x2_extent': 0.6  # Grid spans from x2=-0.6 to x2=+0.6, symmetric around x2=0
        }
    ]
    
    print(f"\nExperimental setup:")
    print(f"  Temperatures: {temperatures}")
    print(f"  Variational families: {variational_families}")
    print(f"  Inducing configurations: {[c['name'] for c in inducing_configs]}")
    
    # Run experiments
    results = run_variational_family_experiment(
        train_x, train_y, test_x, test_y, true_hyperparams,
        temperatures=temperatures,
        variational_families=variational_families,
        inducing_configs=inducing_configs,
        epochs=400,
        verbose=False
    )
    
    print(f"\nTrained {len(results)} models successfully")
    
    # Create visualizations
    fig1 = plot_variational_comparison(train_x, train_y, test_x, test_y, vis_x, vis_grid, results)
    fig2 = plot_metrics_comparison(results)
    fig3 = plot_1d_slice_comparison(train_x, train_y, test_x, test_y, results)
    fig4 = plot_training_losses(results)  # NEW: Add loss curves
    
    plt.show()
    
    # Print summary table
    print("\nDetailed Results Summary:")
    print("-" * 100)
    print("Model".ljust(25) + "Var Family".ljust(12) + "Temp".ljust(6) + 
          "# Induc".ljust(8) + "MSE".ljust(8) + "NLL".ljust(8) + "Mean KL".ljust(10))
    print("-" * 100)
    
    # Print exact GP first
    exact_result = results['exact']
    print("Exact GP".ljust(25) + "exact".ljust(12) + "1.0".ljust(6) + 
          f"{exact_result['num_inducing']}".ljust(8) +
          f"{exact_result['mse']:.4f}".ljust(8) + 
          f"{exact_result['nll']:.4f}".ljust(8) + "0.0000".ljust(10))
    
    # Print sparse GP results
    for key in sorted([k for k in results.keys() if k != 'exact']):
        result = results[key]
        model_name = result['inducing_config']
        var_family = result['variational_family']
        temp = result['temperature']
        num_induc = result['num_inducing']
        
        print(f"{model_name}".ljust(25) + f"{var_family}".ljust(12) + 
              f"{temp}".ljust(6) + f"{num_induc}".ljust(8) +
              f"{result['mse']:.4f}".ljust(8) + 
              f"{result['nll']:.4f}".ljust(8) + 
              f"{result['mean_kl']:.4f}".ljust(10))
    
    return results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the experiment
    results = run_variational_experiment()
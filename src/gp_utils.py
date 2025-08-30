import torch 
import numpy as np

import matplotlib.pyplot as plt

import gpytorch
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, MeanFieldVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood



def create_inducing_points_2D(num_x1=8, num_x2=8, x1_range=(0, 1), x2_extent=0.6):
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

def plot_losses(results):
    """Plot training losses for each model."""
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for key, result in sparse_results.items():
        ax.plot(result['losses'], label=key, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (ELBO)')
    ax.set_title('Training Losses for Different Models')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_1d_comparison(train_x, train_y, test_x, test_y, results):
    """Plot 1D comparison of different inducing positions and temperatures."""
    
    # Filter out exact GP
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    exact_result = results['exact']
    # Organize results by title and temperature
    titles = sorted(list(set([r['title'] for r in sparse_results.values()])))
    temperatures = sorted(list(set([r['temperature'] for r in sparse_results.values()])))
    
    n_titles = len(titles)
    n_temps = len(temperatures)

    # Create figure with subplots for each temperature
    fig, axes = plt.subplots(1, n_temps, figsize=(5*n_temps, 5))
    if n_titles == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'purple']

    for i, temp in enumerate(temperatures):
        ax = axes[i]
        
        # Plot exact GP first (reference)
        ax.plot(test_x[:, 0].numpy(), exact_result['pred_mean'].numpy(), 
                'k-', linewidth=2, label='Exact GP', alpha=0.8)
        ax.fill_between(test_x[:, 0].numpy(), 
                       (exact_result['pred_mean'] - 1.96 * exact_result['pred_std']).numpy(),
                       (exact_result['pred_mean'] + 1.96 * exact_result['pred_std']).numpy(),
                       color='gray', alpha=0.2)
        # Plot sparse GPs for each x2_offset
        for j, title in enumerate(titles):
            key = f"x2={title}_T={temp}"
            if key not in sparse_results:
                continue
                
            result = sparse_results[key]
            
            color = colors[j % len(colors)]
            alpha = 0.7
            
            # Plot mean prediction
            ax.plot(test_x[:, 0].numpy(), result['pred_mean'].numpy(), 
                   color=color, linewidth=2, label=f'offset={title}', alpha=alpha)
            
            # Plot confidence intervals
            ax.fill_between(test_x[:, 0].numpy(), 
                           result['lower'].numpy(),
                           result['upper'].numpy(),
                           color=color, alpha=0.2)
        
        # Plot training data
        ax.scatter(train_x[:, 0].numpy(), train_y.numpy(), 
                  c='red', s=30, marker='o', alpha=0.8, zorder=5, label='Training data')
        
        # Plot test data (true function values)
        ax.scatter(test_x[:, 0].numpy(), test_y.numpy(), 
                  c='lightcoral', s=10, marker='.', alpha=0.6, zorder=5, label='Test data (true)')
        
        
        
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('Function Value', fontsize=12)
        ax.set_title(f'Temperature = {temp}\n' , fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_2d_posterior_comparison(train_x, train_y, test_x, vis_x, vis_grid, results):
    """Plot 2D posterior comparison of different inducing positions and temperatures."""
    
    # Filter out exact GP
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    exact_result = results['exact']

    # Organize results by title and temperature
    titles = sorted(list(set([r['title'] for r in sparse_results.values()])))
    temperatures = sorted(list(set([r['temperature'] for r in sparse_results.values()])))
    
    n_titles = len(titles)
    n_temps = len(temperatures)
    
    # Create figure - with more subplots now
    fig, axes = plt.subplots(n_temps + 1, n_titles, figsize=(2.5*n_titles, 2.5*(n_temps+1)))
    if titles == 1:
        axes = axes.reshape(-1, 1)
    
    X1, X2 = vis_grid
    
    # Get exact GP predictions on visualization grid
    exact_vis_mean, exact_vis_std, _, _ = evaluate_model(exact_result['model'], exact_result['likelihood'], vis_x)
    exact_mean_2d = exact_vis_mean.reshape(X1.shape)
    
    for i, titles in enumerate(titles):
        # Row 1: Exact GP (reference) - only show in first column
        if i == 0:
            ax = axes[0, i]
            im = ax.contourf(X1.numpy(), X2.numpy(), exact_mean_2d.numpy(), 
                            levels=20, cmap='viridis')
            
            # Plot training data
            ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
                       c='red', s=15, marker='o', label='Train')
            
            # Plot test data points
            ax.scatter(test_x[:, 0].numpy(), test_x[:, 1].numpy(), 
                       c='lightcoral', s=4, marker='.', alpha=0.6, label='Test')
            
            ax.set_title('Exact GP', fontsize=8)
            ax.set_xlabel('x1', fontsize=7)
            ax.set_ylabel('x2', fontsize=7)
            ax.legend(fontsize=6)
            plt.colorbar(im, ax=ax, shrink=0.6)
        else:
            axes[0, i].text(0.5, 0.5, 'Same as\nfirst column', 
                           ha='center', va='center', transform=axes[0, i].transAxes,
                           fontsize=7)
            axes[0, i].set_title(f'x2={titles}', fontsize=8)
        
        # Rows 2+: Different temperatures for each x2_offset
        for j, temp in enumerate(temperatures):
            key = f"x2={titles}_T={temp}"
            if key not in sparse_results:
                continue
                
            result = sparse_results[key]
            
            # Get predictions on visualization grid
            vis_pred_mean, vis_pred_std, _, _ = evaluate_model(result['model'], result['likelihood'], vis_x)
            pred_mean_2d = vis_pred_mean.reshape(X1.shape)
            
            # Plot predictions with consistent color scale
            ax = axes[j+1, i]
            # Use same color limits as exact GP for comparison
            vmin, vmax = exact_mean_2d.min().item(), exact_mean_2d.max().item()
            im = ax.contourf(X1.numpy(), X2.numpy(), pred_mean_2d.numpy(), 
                            levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Plot training data
            ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
                       c='red', s=15, marker='o', alpha=0.8)
            
            # Plot test data points
            ax.scatter(test_x[:, 0].numpy(), test_x[:, 1].numpy(), 
                       c='lightcoral', s=4, marker='.', alpha=0.6)
            
            # Plot fixed inducing points
            inducing_points = result['inducing_points']
            ax.scatter(inducing_points[:, 0].numpy(), inducing_points[:, 1].numpy(),
                       c='white', s=15, marker='s', linewidth=0.5, 
                       edgecolors='black', alpha=0.8)
            
            ax.set_title(f'T={temp}\nNLL: {result["nll"]:.2f}', fontsize=8)
            ax.set_xlabel('x1', fontsize=7)
            ax.set_ylabel('x2', fontsize=7)
            ax.tick_params(axis='both', which='major', labelsize=6)
            
            # Add horizontal line at x2=0 (where training data is)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
            ax.axhline(y=titles, color='white', linestyle=':', alpha=0.9, linewidth=1.5)
            
            plt.colorbar(im, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    return fig



def create_inducing_points_1D(num_inducing=20, x2_offset=0.0, x1_range=(0, 1)):
    """
    Create inducing points with specified x2 offset from training data.
    
    Args:
        num_inducing: Number of inducing points
        x2_offset: Offset in x2 dimension (0 = same as training data)
        x1_range: Range for x1 coordinates
    """
    x1_coords = torch.linspace(x1_range[0], x1_range[1], num_inducing)
    x2_coords = torch.full((num_inducing,), x2_offset)
    inducing_points = torch.stack([x1_coords, x2_coords], dim=1)
    return inducing_points

def train_cold_posterior_gp_with_inducing(train_x, train_y, inducing_points, true_hyperparams, 
                                         temperature=1.0, lr=0.01, epochs=500, verbose=True, variational_family='cholesky'):
    """Train sparse GP with pre-specified inducing points and fixed hyperparameters."""
    lengthscale, outputscale, noise_var = true_hyperparams
    
    # Initialize model and likelihood
    model = ColdPosteriorSparseGP(inducing_points, temperature=temperature, variational_family=variational_family)
    likelihood = GaussianLikelihood()
    
    # Fix hyperparameters to true values
    set_kernel_hyperparameters(model, lengthscale, outputscale)
    set_likelihood_noise(likelihood, noise_var)
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Optimizer - only optimize variational parameters (hyperparameters and temperature are fixed)
    trainable_params = [p for p in list(model.parameters())]
    # Remove temperature from trainable parameters
    trainable_params = [p for p in trainable_params]
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
                  f'Temperature: {model.temperature.item():.4f}')
    
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
        
        Cold posterior modifies the likelihood term by temperature scaling.
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
    
    # Optimizer - only optimize mean (hyperparameters and temperature are fixed)
    trainable_params = [p for p in list(model.parameters()) + list(likelihood.parameters()) 
                       if p.requires_grad]
    # Remove temperature from trainable parameters
    trainable_params = [p for p in trainable_params]
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
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, '
                  f'Temperature: {model.temperature.item():.4f}')
    
    return model, likelihood, losses

def compute_kl_divergence(mean1, std1, mean2, std2):
    """
    Compute KL divergence between two univariate Gaussians.
    KL(N(μ1, σ1²) || N(μ2, σ2²))
    """
    var1, var2 = std1**2, std2**2
    kl = torch.log(std2 / std1) + (var1 + (mean1 - mean2)**2) / (2 * var2) - 0.5
    return kl

def create_synthetic_data(n_train=200, n_test=100, n_test_per_dim=30, noise_std=0.1, seed=42):
    """Generate synthetic 2D regression data with training and testing data on x2=0 line.
    
    Function is a single draw from a GP with known hyperparameters.
    Creates 2D grid for visualization but only tests on 1D line.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Training data - ALL on the x2=0 line (1D subspace)
    # train_x1 = torch.rand(n_train)
    train_x1 = torch.linspace(0,1,n_train)
    train_x2 = torch.zeros(n_train)  # All training data at x2=0
    train_x = torch.stack([train_x1, train_x2], dim=1)
    
    # Test data - on the x2=0 line for evaluation
    test_x1 = torch.linspace(0, 1, n_test)
    test_x2 = torch.zeros(n_test)  # All test data at x2=0
    test_x = torch.stack([test_x1, test_x2], dim=1)
    
    # 2D grid for visualization only (not for evaluation)
    x1_vis = torch.linspace(0, 1, n_test_per_dim)
    x2_vis = torch.linspace(-0.2, 1.2, n_test_per_dim)  # Extended range to see extrapolation
    X1, X2 = torch.meshgrid(x1_vis, x2_vis, indexing='ij')
    vis_x = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    
    # Combine points where we need function values (train + test + visualization)
    all_x = torch.cat([train_x, test_x, vis_x], dim=0)
    
    # True GP hyperparameters (these will be fixed in all models)
    TRUE_LENGTHSCALE = 0.3
    TRUE_OUTPUTSCALE = 1.0
    TRUE_NOISE = noise_std**2
    
    # Create GP kernel with true hyperparameters
    from gpytorch.kernels import ScaleKernel, RBFKernel
    true_kernel = ScaleKernel(RBFKernel())
    true_kernel.outputscale = TRUE_OUTPUTSCALE
    true_kernel.base_kernel.lengthscale = TRUE_LENGTHSCALE
    
    # Generate function values by sampling from GP prior
    with torch.no_grad():
        # Compute kernel matrix for all points
        K = true_kernel(all_x).evaluate()
        

        
        # Use Cholesky decomposition for more stable sampling
        # try:
        L = torch.linalg.cholesky(K + noise_std**2*torch.eye(K.size(0)))
        # Sample from standard normal and transform
        z = torch.randn(K.size(0))
        f_values = L @ z
        # except torch.linalg.LinAlgError:
        #     # Fallback: add more jitter and try eigendecomposition
        #     print("Warning: Using eigendecomposition fallback for sampling")
        #     jitter = 1e-3
        #     K += jitter * torch.eye(K.size(0))
        #     eigenvals, eigenvecs = torch.linalg.eigh(K)
        #     eigenvals = torch.clamp(eigenvals, min=1e-6)  # Ensure positive
            
        #     # Sample using eigendecomposition
        #     z = torch.randn(K.size(0))
        #     f_values = eigenvecs @ (torch.sqrt(eigenvals) * z)
    
    # Split back into train, test, and visualization
    train_f = f_values[:n_train]
    test_f = f_values[n_train:n_train+n_test]
    vis_f = f_values[n_train+n_test:]
    
    # Add noise to training data
    train_y = train_f 
    test_y = test_f  # True function values (no noise)
    
    print(f"True GP hyperparameters:")
    print(f"  Lengthscale: {TRUE_LENGTHSCALE}")
    print(f"  Outputscale: {TRUE_OUTPUTSCALE}")
    print(f"  Noise: {TRUE_NOISE}")
    
    return train_x, train_y, test_x, test_y, vis_x, (X1, X2), (TRUE_LENGTHSCALE, TRUE_OUTPUTSCALE, TRUE_NOISE)
import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO, ExactMarginalLogLikelihood

class ColdPosteriorSparseGP(ApproximateGP):
    """
    Sparse GP with temperature scaling for cold posterior effect research.
    Based on GPyTorch's variational sparse GP implementation.
    """
    
    def __init__(self, inducing_points, temperature=1.0):
        """
        Args:
            inducing_points: Tensor of inducing point locations [M, D]
            temperature: Temperature parameter (T=1 standard, T<1 cold)
        """
        # Initialize variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=False
        )
        
        super().__init__(variational_strategy)
        
        # GP components
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        self.temperature = torch.tensor(temperature)

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
        self.register_parameter('raw_temperature', 
                              torch.nn.Parameter(torch.log(torch.tensor(temperature, dtype=torch.float32))))
        
    @property 
    def temperature(self):
        return torch.exp(self.raw_temperature)
    
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
    
    # Optimizer - only optimize mean and temperature (hyperparameters are fixed)
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

# def create_synthetic_data(n_train=200, n_test_per_dim=30, noise_std=0.1, seed=42):
#     """Generate synthetic 2D regression data with training data on x2=0 line.
    
#     Function is a single draw from a GP with known hyperparameters.
#     """
#     torch.manual_seed(seed)
#     np.random.seed(seed)
    
#     # Training data - ALL on the x2=0 line (1D subspace)
#     train_x1 = torch.linspace(0, 1, n_train)
#     train_x2 = torch.zeros(n_train)  # All training data at x2=0
#     train_x = torch.stack([train_x1, train_x2], dim=1)
    
#     # Test data - regular grid covering both dimensions
#     x1_test = torch.linspace(0, 1, n_test_per_dim)
#     x2_test = torch.linspace(-0.2, 1.2, n_test_per_dim)  # Extended range to see extrapolation
#     X1, X2 = torch.meshgrid(x1_test, x2_test, indexing='ij')
#     test_x = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    
#     # Combine all points where we need function values
#     all_x = torch.cat([train_x, test_x], dim=0)
    
#     # True GP hyperparameters (these will be fixed in all models)
#     TRUE_LENGTHSCALE = 0.3
#     TRUE_OUTPUTSCALE = 1.0
#     TRUE_NOISE = noise_std
    
#     # Create GP kernel with true hyperparameters
#     from gpytorch.kernels import ScaleKernel, RBFKernel
#     true_kernel = ScaleKernel(RBFKernel())
#     true_kernel.outputscale = TRUE_OUTPUTSCALE
#     true_kernel.base_kernel.lengthscale = TRUE_LENGTHSCALE
    
#     # Generate function values by sampling from GP prior
#     with torch.no_grad():
#         # Compute kernel matrix for all points
#         K = true_kernel(all_x).evaluate()
        
#         # Add larger jitter for numerical stability
#         jitter = 1e-4
#         K += jitter * torch.eye(K.size(0))
        
#         # Use Cholesky decomposition for more stable sampling
#         try:
#             L = torch.linalg.cholesky(K)
#             # Sample from standard normal and transform
#             z = torch.randn(K.size(0))
#             f_values = L @ z
#         except torch.linalg.LinAlgError:
#             # Fallback: add more jitter and try eigendecomposition
#             print("Warning: Using eigendecomposition fallback for sampling")
#             jitter = 1e-3
#             K += jitter * torch.eye(K.size(0))
#             eigenvals, eigenvecs = torch.linalg.eigh(K)
#             eigenvals = torch.clamp(eigenvals, min=1e-6)  # Ensure positive
            
#             # Sample using eigendecomposition
#             z = torch.randn(K.size(0))
#             f_values = eigenvecs @ (torch.sqrt(eigenvals) * z)
    
#     # Split back into train and test
#     train_f = f_values[:n_train]
#     test_f = f_values[n_train:]
    
#     # Add noise to training data
#     train_y = train_f + noise_std * torch.randn(n_train)
#     test_y = test_f  # True function values (no noise)
    
#     print(f"True GP hyperparameters:")
#     print(f"  Lengthscale: {TRUE_LENGTHSCALE}")
#     print(f"  Outputscale: {TRUE_OUTPUTSCALE}")
#     print(f"  Noise: {TRUE_NOISE}")
    
#     return train_x, train_y, test_x, test_y, (X1, X2), (TRUE_LENGTHSCALE, TRUE_OUTPUTSCALE, TRUE_NOISE)


def create_synthetic_data(n_train=200, n_test=25**2, n_test_per_dim=30, noise_std=0.1, seed=42):
    """Generate synthetic 2D regression data with training and testing data on x2=0 line.
    
    Function is a single draw from a GP with known hyperparameters.
    Creates 2D grid for visualization but only tests on 1D line.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Training data - ALL on the x2=0 line (1D subspace)
    train_x1 = torch.linspace(0, 1, n_train)
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
    TRUE_NOISE = noise_std
    
    # Create GP kernel with true hyperparameters
    from gpytorch.kernels import ScaleKernel, RBFKernel
    true_kernel = ScaleKernel(RBFKernel())
    true_kernel.outputscale = TRUE_OUTPUTSCALE
    true_kernel.base_kernel.lengthscale = TRUE_LENGTHSCALE
    
    # Generate function values by sampling from GP prior
    with torch.no_grad():
        # Compute kernel matrix for all points
        K = true_kernel(all_x).evaluate()
        
        # Add larger jitter for numerical stability
        jitter = 1e-4
        K += jitter * torch.eye(K.size(0))
        
        # Use Cholesky decomposition for more stable sampling
        try:
            L = torch.linalg.cholesky(K)
            # Sample from standard normal and transform
            z = torch.randn(K.size(0))
            f_values = L @ z
        except torch.linalg.LinAlgError:
            # Fallback: add more jitter and try eigendecomposition
            print("Warning: Using eigendecomposition fallback for sampling")
            jitter = 1e-3
            K += jitter * torch.eye(K.size(0))
            eigenvals, eigenvecs = torch.linalg.eigh(K)
            eigenvals = torch.clamp(eigenvals, min=1e-6)  # Ensure positive
            
            # Sample using eigendecomposition
            z = torch.randn(K.size(0))
            f_values = eigenvecs @ (torch.sqrt(eigenvals) * z)
    
    # Split back into train, test, and visualization
    train_f = f_values[:n_train]
    test_f = f_values[n_train:n_train+n_test]
    vis_f = f_values[n_train+n_test:]
    
    # Add noise to training data
    train_y = train_f + noise_std * torch.randn(n_train)
    test_y = test_f  # True function values (no noise)
    
    print(f"True GP hyperparameters:")
    print(f"  Lengthscale: {TRUE_LENGTHSCALE}")
    print(f"  Outputscale: {TRUE_OUTPUTSCALE}")
    print(f"  Noise: {TRUE_NOISE}")
    return train_x, train_y, test_x, test_y,  (X1, X2), (TRUE_LENGTHSCALE, TRUE_OUTPUTSCALE, TRUE_NOISE)

def create_inducing_points(num_inducing=20, x2_offset=0.0, x1_range=(0, 1)):
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
                                         temperature=1.0, lr=0.01, epochs=500, verbose=True):
    """Train sparse GP with pre-specified inducing points and fixed hyperparameters."""
    lengthscale, outputscale, noise_var = true_hyperparams
    
    # Initialize model and likelihood
    model = ColdPosteriorSparseGP(inducing_points, temperature=temperature)
    likelihood = GaussianLikelihood()
    
    # Fix hyperparameters to true values
    set_kernel_hyperparameters(model, lengthscale, outputscale)
    set_likelihood_noise(likelihood, noise_var)
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Optimizer - only optimize variational parameters (hyperparameters are fixed)
    trainable_params = [p for p in list(model.parameters()) + list(likelihood.parameters()) 
                       if p.requires_grad]
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

def compare_inducing_positions_and_temperatures(train_x, train_y, test_x, test_y, true_hyperparams,
                                               temperatures=[0.1, 1.0], 
                                               x2_offsets=[0.0, 0.5, 1.0],
                                               num_inducing=20, epochs=300):
    """Compare models with different inducing point positions and temperatures."""
    
    results = {}
    
    print("Training exact GP (T=1.0) as reference...")
    exact_model, exact_likelihood, _ = train_exact_gp(
        train_x, train_y, true_hyperparams, temperature=1.0, epochs=epochs, verbose=False
    )
    exact_pred_mean, exact_pred_std, _, _ = evaluate_model(exact_model, exact_likelihood, test_x)
    
    # Compute exact GP metrics
    exact_mse = torch.mean((exact_pred_mean - test_y)**2).item()
    exact_nll = -torch.distributions.Normal(exact_pred_mean, exact_pred_std).log_prob(test_y).mean().item()
    
    # Compute exact GP extrapolation error
    extrap_mask = test_x[:, 1] != 0
    if extrap_mask.sum() > 0:
        exact_extrap_mse = torch.mean((exact_pred_mean[extrap_mask] - test_y[extrap_mask])**2).item()
        exact_extrap_nll = -torch.distributions.Normal(
            exact_pred_mean[extrap_mask], exact_pred_std[extrap_mask]
        ).log_prob(test_y[extrap_mask]).mean().item()
    else:
        exact_extrap_mse = float('nan')
        exact_extrap_nll = float('nan')
    
    print(f"Exact GP - MSE: {exact_mse:.4f}, NLL: {exact_nll:.4f}, "
          f"Extrap MSE: {exact_extrap_mse:.4f}, Extrap NLL: {exact_extrap_nll:.4f}")
    
    print(f"\nTraining data all at x2=0")
    print(f"Testing inducing point x2 offsets: {x2_offsets}")
    print(f"Testing temperatures: {temperatures}")
    
    for x2_offset in x2_offsets:
        for temp in temperatures:
            key = f"x2={x2_offset}_T={temp}"
            print(f"\nTraining {key}")
            
            # Create inducing points at specified x2 offset
            inducing_points = create_inducing_points(num_inducing, x2_offset)
            
            model, likelihood, losses = train_cold_posterior_gp_with_inducing(
                train_x, train_y, inducing_points, true_hyperparams, temperature=temp, 
                epochs=epochs, verbose=False
            )
            
            # Evaluate
            pred_mean, pred_std, lower, upper = evaluate_model(model, likelihood, test_x)
            
            # Compute metrics
            mse = torch.mean((pred_mean - test_y)**2).item()
            nll = -torch.distributions.Normal(pred_mean, pred_std).log_prob(test_y).mean().item()
            
            # Compute KL divergence from exact GP
            kl_from_exact = compute_kl_divergence(pred_mean, pred_std, exact_pred_mean, exact_pred_std)
            mean_kl = torch.mean(kl_from_exact).item()
            
            # Compute extrapolation error (test points where x2 != 0)
            if extrap_mask.sum() > 0:
                extrap_mse = torch.mean((pred_mean[extrap_mask] - test_y[extrap_mask])**2).item()
                extrap_nll = -torch.distributions.Normal(
                    pred_mean[extrap_mask], pred_std[extrap_mask]
                ).log_prob(test_y[extrap_mask]).mean().item()
            else:
                extrap_mse = float('nan')
                extrap_nll = float('nan')
            
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
                'extrap_mse': extrap_mse,
                'extrap_nll': extrap_nll,
                'final_temp': model.temperature.item(),
                'kl_from_exact': kl_from_exact,
                'mean_kl': mean_kl,
                'x2_offset': x2_offset,
                'temperature': temp,
                'inducing_points': inducing_points
            }
            
            print(f"Final temp: {model.temperature.item():.4f}, MSE: {mse:.4f}, NLL: {nll:.4f}, "
                  f"Extrap MSE: {extrap_mse:.4f}, Extrap NLL: {extrap_nll:.4f}, Mean KL: {mean_kl:.4f}")
    
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
        'extrap_mse': exact_extrap_mse,
        'extrap_nll': exact_extrap_nll,
        'final_temp': 1.0
    }
    
    return results


def plot_inducing_position_comparison(train_x, train_y, test_x, test_y, test_grid, results):
    """Plot comparison of different inducing positions and temperatures."""
    
    # Filter out exact GP
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    exact_result = results['exact']
    
    # Organize results by x2_offset and temperature
    x2_offsets = sorted(list(set([r['x2_offset'] for r in sparse_results.values()])))
    temperatures = sorted(list(set([r['temperature'] for r in sparse_results.values()])))
    
    n_offsets = len(x2_offsets)
    n_temps = len(temperatures)
    
    # Create smaller figure - ensure all temperatures fit
    fig, axes = plt.subplots(n_temps + 1, n_offsets, figsize=(3.5*n_offsets, 3*(n_temps+1)))
    if n_offsets == 1:
        axes = axes.reshape(-1, 1)
    
    X1, X2 = test_grid
    exact_mean_2d = exact_result['pred_mean'].reshape(X1.shape)
    
    for i, x2_offset in enumerate(x2_offsets):
        # Row 1: Exact GP (reference) - only show in first column
        if i == 0:
            ax = axes[0, i]
            im = ax.contourf(X1.numpy(), X2.numpy(), exact_mean_2d.numpy(), 
                            levels=20, cmap='viridis')
            
            # Plot training data
            ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
                       c='red', s=25, marker='o', label='Training data')
            
            # Plot test data points
            ax.scatter(test_x[:, 0].numpy(), test_x[:, 1].numpy(), 
                       c='lightcoral', s=8, marker='.', alpha=0.6, label='Test data')
            
            ax.set_title('Exact GP (Reference)', fontsize=10)
            ax.set_xlabel('x1', fontsize=9)
            ax.set_ylabel('x2', fontsize=9)
            ax.legend(fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            axes[0, i].text(0.5, 0.5, 'Same as\nfirst column', 
                           ha='center', va='center', transform=axes[0, i].transAxes,
                           fontsize=9)
            axes[0, i].set_title(f'x2_offset = {x2_offset}', fontsize=10)
        
        # Rows 2-4: Different temperatures for each x2_offset
        for j, temp in enumerate(temperatures):
            key = f"x2={x2_offset}_T={temp}"
            if key not in sparse_results:
                continue
                
            result = sparse_results[key]
            pred_mean_2d = result['pred_mean'].reshape(X1.shape)
            kl_2d = result['kl_from_exact'].reshape(X1.shape)
            
            # Plot predictions with consistent color scale
            ax = axes[j+1, i]
            # Use same color limits as exact GP for comparison
            vmin, vmax = exact_mean_2d.min().item(), exact_mean_2d.max().item()
            im = ax.contourf(X1.numpy(), X2.numpy(), pred_mean_2d.numpy(), 
                            levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
            
            # Plot training data
            ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
                       c='red', s=25, marker='o', alpha=0.8)
            
            # Plot test data points
            ax.scatter(test_x[:, 0].numpy(), test_x[:, 1].numpy(), 
                       c='lightcoral', s=8, marker='.', alpha=0.6)
            
            # Plot fixed inducing points
            inducing_points = result['inducing_points']
            ax.scatter(inducing_points[:, 0].numpy(), inducing_points[:, 1].numpy(),
                       c='white', s=25, marker='s', linewidth=1, 
                       edgecolors='black', alpha=0.8, label='Fixed inducing grid')
            
            ax.set_title(f'T={temp}, x2_offset={x2_offset}\n'
                        f'NLL: {result["nll"]:.3f}, KL: {result["mean_kl"]:.3f}', fontsize=10)
            ax.set_xlabel('x1', fontsize=9)
            ax.set_ylabel('x2', fontsize=9)
            
            # Add horizontal line at x2=0 (where training data is)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.axhline(y=x2_offset, color='white', linestyle=':', alpha=0.9, linewidth=2)
            
            if i == 0:  # Only show legend on first column
                # Update legend to include test data
                ax.legend(['Fixed inducing grid'], loc='upper right', fontsize=7)
            
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    return fig

def plot_extrapolation_analysis(results):
    """Plot analysis focusing on KL divergence vs NLL - the key metrics."""
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    
    # Extract data for plotting
    x2_offsets = []
    temperatures = []
    nll_values = []
    extrap_nll_values = []
    kl_values = []
    
    for key, result in sparse_results.items():
        x2_offsets.append(result['x2_offset'])
        temperatures.append(result['temperature'])
        nll_values.append(result['nll'])
        extrap_nll_values.append(result['extrap_nll'])
        kl_values.append(result['mean_kl'])
    
    # Convert to arrays for easier plotting
    x2_offsets = np.array(x2_offsets)
    temperatures = np.array(temperatures)
    nll_values = np.array(nll_values)
    extrap_nll_values = np.array(extrap_nll_values)
    kl_values = np.array(kl_values)
    
    # Create focused figure on key metrics
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Get exact GP results for reference lines
    exact_result = results['exact']
    exact_nll = exact_result['nll']
    exact_extrap_nll = exact_result['extrap_nll']
    
    unique_temps = np.unique(temperatures)
    unique_offsets = np.unique(x2_offsets)
    
    # Plot 1: NLL vs x2_offset for different temperatures
    ax = axes[0, 0]
    for temp in unique_temps:
        mask = temperatures == temp
        ax.plot(x2_offsets[mask], nll_values[mask], 'o-', 
                label=f'T={temp}', linewidth=2, markersize=8)
    ax.axhline(y=exact_nll, color='black', linestyle='--', alpha=0.7, 
               linewidth=2, label='Exact GP')
    ax.set_xlabel('Inducing Point x2 Offset')
    ax.set_ylabel('Overall NLL')
    ax.set_title('Predictive NLL vs Inducing Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Extrapolation NLL vs x2_offset
    ax = axes[0, 1]
    for temp in unique_temps:
        mask = temperatures == temp
        ax.plot(x2_offsets[mask], extrap_nll_values[mask], 's-', 
                label=f'T={temp}', linewidth=2, markersize=8)
    ax.axhline(y=exact_extrap_nll, color='black', linestyle='--', alpha=0.7, 
               linewidth=2, label='Exact GP')
    ax.set_xlabel('Inducing Point x2 Offset')
    ax.set_ylabel('Extrapolation NLL (x2≠0)')
    ax.set_title('Extrapolation NLL vs Inducing Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: KL divergence vs x2_offset  
    ax = axes[1, 0]
    for temp in unique_temps:
        mask = temperatures == temp
        ax.plot(x2_offsets[mask], kl_values[mask], '^-', 
                label=f'T={temp}', linewidth=2, markersize=8)
    ax.set_xlabel('Inducing Point x2 Offset')
    ax.set_ylabel('Mean KL from Exact GP')
    ax.set_title('KL Divergence vs Inducing Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: KL vs NLL scatter plot (key relationship)
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_offsets)))
    for i, offset in enumerate(unique_offsets):
        mask = x2_offsets == offset
        for temp in unique_temps:
            temp_mask = mask & (temperatures == temp)
            if temp_mask.sum() > 0:
                marker = 'o' if temp == unique_temps[0] else 's'
                ax.scatter(kl_values[temp_mask], nll_values[temp_mask], 
                          c=[colors[i]], s=100, marker=marker, 
                          label=f'x2={offset}, T={temp}', alpha=0.8)
    
    # Add exact GP point
    ax.scatter(0, exact_nll, c='black', s=100, marker='*', 
               label='Exact GP', alpha=0.8)
    
    ax.set_xlabel('Mean KL from Exact GP')
    ax.set_ylabel('Overall NLL')
    ax.set_title('KL Divergence vs Predictive NLL')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_inducing_movement_analysis(results):
    """Plot analysis of how inducing points move during training."""
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get unique offsets and temperatures
    x2_offsets = sorted(list(set([r['x2_offset'] for r in sparse_results.values()])))
    temperatures = sorted(list(set([r['temperature'] for r in sparse_results.values()])))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(temperatures)))
    
    for i, x2_offset in enumerate(x2_offsets):
        ax = axes[i]
        
        for j, temp in enumerate(temperatures):
            key = f"x2={x2_offset}_T={temp}"
            if key not in sparse_results:
                continue
                
            result = sparse_results[key]
            initial_pts = result['initial_inducing_points']
            final_pts = result['final_inducing_points']
            
            # Plot initial positions
            ax.scatter(initial_pts[:, 0].numpy(), initial_pts[:, 1].numpy(),
                      c='lightgray', s=30, marker='o', alpha=0.5)
            
            # Plot final positions
            ax.scatter(final_pts[:, 0].numpy(), final_pts[:, 1].numpy(),
                      c=[colors[j]], s=60, marker='x', linewidth=2,
                      label=f'T={temp}')
            
            # Draw arrows showing movement
            for k in range(len(initial_pts)):
                ax.annotate('', xy=final_pts[k].numpy(), xytext=initial_pts[k].numpy(),
                           arrowprops=dict(arrowstyle='->', color=colors[j], alpha=0.6))
        
        # Add training data line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                   label='Training data (x2=0)')
        ax.axhline(y=x2_offset, color='black', linestyle=':', alpha=0.5, linewidth=1, 
                   label=f'Initial x2={x2_offset}')
        
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title(f'Inducing Point Movement\nInitial x2_offset={x2_offset}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.3, 1.3)
    
    plt.tight_layout()
    return fig
    """Plot training loss curves for different temperatures."""
    plt.figure(figsize=(10, 6))
    
    for temp, result in results.items():
        if temp != 'exact':
            plt.plot(result['losses'], label=f'T = {temp}', linewidth=2)
    
def plot_training_curves(results):
    """Plot training loss curves for different temperatures."""
    plt.figure(figsize=(10, 6))
    
    for temp, result in results.items():
        if temp != 'exact':
            plt.plot(result['losses'], label=f'T = {temp}', linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Negative ELBO')
    plt.title('Training Curves for Different Temperatures')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    return plt.gcf()

def run_cold_posterior_experiment():
    """Run complete cold posterior experiment with sparse GPs."""
    
    print("Cold Posterior Inducing Point Position Experiment (2D)")
    print("=" * 55)
    
    # Generate 2D data with training on x2=0 line
    train_x, train_y, test_x, test_y, test_grid, true_hyperparams = create_synthetic_data(
        n_train=50, n_test_per_dim=25, noise_std=0.1
    )
    
    print(f"Training data shape: {train_x.shape}")
    print(f"Training data x2 range: [{train_x[:, 1].min():.3f}, {train_x[:, 1].max():.3f}]")
    print(f"Test data shape: {test_x.shape}")
    
    # Compare different inducing positions and temperatures
    temperatures = [0.1, 1.0]  # Cold vs normal temperature
    x2_offsets = [0.0, 0.5, 1.0]  # Different positions for inducing points
    
    results = compare_inducing_positions_and_temperatures(
        train_x, train_y, test_x, test_y, true_hyperparams,
        temperatures, x2_offsets
    )
    
    # Create visualizations
    fig1 = plot_inducing_position_comparison(train_x, train_y, test_x, test_y, test_grid, results)
    fig2 = plot_extrapolation_analysis(results)
    
    plt.show()
    
    return results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiment
    results = run_cold_posterior_experiment()
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 95)
    print("Model".ljust(20) + "MSE".ljust(8) + "NLL".ljust(8) + "Extrap MSE".ljust(12) + "Extrap NLL".ljust(12) + "Mean KL".ljust(10) + "Final T")
    print("-" * 95)
    
    # Print exact GP first
    exact_result = results['exact']
    print("Exact GP".ljust(20) + 
          f"{exact_result['mse']:.4f}".ljust(8) + 
          f"{exact_result['nll']:.4f}".ljust(8) + 
          f"{exact_result['extrap_mse']:.4f}".ljust(12) + 
          f"{exact_result['extrap_nll']:.4f}".ljust(12) + 
          "0.0000".ljust(10) + 
          f"{exact_result['final_temp']:.4f}")
    
    # Print sparse GPs organized by inducing position
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    for key in sorted(sparse_results.keys()):
        result = sparse_results[key]
        model_name = f"x2={result['x2_offset']}, T={result['temperature']}"
        print(model_name.ljust(20) + 
              f"{result['mse']:.4f}".ljust(8) + 
              f"{result['nll']:.4f}".ljust(8) + 
              f"{result['extrap_mse']:.4f}".ljust(12) + 
              f"{result['extrap_nll']:.4f}".ljust(12) + 
              f"{result['mean_kl']:.4f}".ljust(10) + 
              f"{result['final_temp']:.4f}")
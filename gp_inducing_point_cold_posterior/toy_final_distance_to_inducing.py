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

def compare_inducing_positions_and_temperatures(train_x, train_y, test_x, test_y, true_hyperparams,
                                               temperatures=[0.1, 1.0], 
                                               x2_offsets=[0.0, 0.5, 1.0],
                                               num_inducing=32, epochs=300):
    """Compare models with different inducing point positions and temperatures."""
    
    results = {}
    
    print("Training exact GP (T=1.0) as reference...")
    exact_model, exact_likelihood, _ = train_exact_gp(
        train_x, train_y, true_hyperparams, temperature=1.0, epochs=epochs, verbose=False
    )
    exact_pred_mean, exact_pred_std, _, _ = evaluate_model(exact_model, exact_likelihood, test_x)
    
    # Compute exact GP metrics (no extrapolation since all data is on x2=0)
    exact_mse = torch.mean((exact_pred_mean - test_y)**2).item()
    exact_nll = -torch.distributions.Normal(exact_pred_mean, exact_pred_std).log_prob(test_y).mean().item()
    
    print(f"Exact GP - MSE: {exact_mse:.4f}, NLL: {exact_nll:.4f}")
    
    print(f"\nTraining and testing data all at x2=0")
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
                'x2_offset': x2_offset,
                'temperature': temp,
                'inducing_points': inducing_points
            }
            
            print(f"Final temp: {model.temperature.item():.4f}, MSE: {mse:.4f}, NLL: {nll:.4f}, "
                  f"Mean KL: {mean_kl:.4f}")
    
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
        'final_temp': 1.0
    }
    
    return results

def plot_2d_posterior_comparison(train_x, train_y, test_x, vis_x, vis_grid, results):
    """Plot 2D posterior comparison of different inducing positions and temperatures."""
    
    # Filter out exact GP
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    exact_result = results['exact']
    
    # Organize results by x2_offset and temperature
    x2_offsets = sorted(list(set([r['x2_offset'] for r in sparse_results.values()])))
    temperatures = sorted(list(set([r['temperature'] for r in sparse_results.values()])))
    
    n_offsets = len(x2_offsets)
    n_temps = len(temperatures)
    
    # Create figure - with more subplots now
    fig, axes = plt.subplots(n_temps + 1, n_offsets, figsize=(2.5*n_offsets, 2.5*(n_temps+1)))
    if n_offsets == 1:
        axes = axes.reshape(-1, 1)
    
    X1, X2 = vis_grid
    
    # Get exact GP predictions on visualization grid
    exact_vis_mean, exact_vis_std, _, _ = evaluate_model(exact_result['model'], exact_result['likelihood'], vis_x)
    exact_mean_2d = exact_vis_mean.reshape(X1.shape)
    
    for i, x2_offset in enumerate(x2_offsets):
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
            axes[0, i].set_title(f'x2={x2_offset}', fontsize=8)
        
        # Rows 2+: Different temperatures for each x2_offset
        for j, temp in enumerate(temperatures):
            key = f"x2={x2_offset}_T={temp}"
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
            ax.axhline(y=x2_offset, color='white', linestyle=':', alpha=0.9, linewidth=1.5)
            
            plt.colorbar(im, ax=ax, shrink=0.6)
    
    plt.tight_layout()
    return fig
def plot_1d_comparison(train_x, train_y, test_x, test_y, results):
    """Plot 1D comparison of different inducing positions and temperatures."""
    
    # Filter out exact GP
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    exact_result = results['exact']
    
    # Organize results by x2_offset and temperature
    x2_offsets = sorted(list(set([r['x2_offset'] for r in sparse_results.values()])))
    temperatures = sorted(list(set([r['temperature'] for r in sparse_results.values()])))
    
    n_offsets = len(x2_offsets)
    n_temps = len(temperatures)
    
    # Create figure with subplots for each x2_offset
    fig, axes = plt.subplots(1, n_offsets, figsize=(5*n_offsets, 5))
    if n_offsets == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, x2_offset in enumerate(x2_offsets):
        ax = axes[i]
        
        # Plot exact GP first (reference)
        ax.plot(test_x[:, 0].numpy(), exact_result['pred_mean'].numpy(), 
                'k-', linewidth=2, label='Exact GP', alpha=0.8)
        ax.fill_between(test_x[:, 0].numpy(), 
                       (exact_result['pred_mean'] - 1.96 * exact_result['pred_std']).numpy(),
                       (exact_result['pred_mean'] + 1.96 * exact_result['pred_std']).numpy(),
                       color='gray', alpha=0.2)
        
        # Plot sparse GPs for each temperature
        for j, temp in enumerate(temperatures):
            key = f"x2={x2_offset}_T={temp}"
            if key not in sparse_results:
                continue
                
            result = sparse_results[key]
            
            color = colors[j % len(colors)]
            alpha = 0.7
            
            # Plot mean prediction
            ax.plot(test_x[:, 0].numpy(), result['pred_mean'].numpy(), 
                   color=color, linewidth=2, label=f'T={temp}', alpha=alpha)
            
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
        
        # Show inducing point positions
        for j, temp in enumerate(temperatures):
            key = f"x2={x2_offset}_T={temp}"
            if key in sparse_results:
                result = sparse_results[key]
                inducing_points = result['inducing_points']
                # Show inducing points on x-axis
                ax.scatter(inducing_points[:, 0].numpy(), 
                          [ax.get_ylim()[0]] * len(inducing_points), 
                          c=colors[j % len(colors)], s=40, marker='|', alpha=0.8,
                          label=f'Inducing (T={temp})' if i == 0 else "")
        
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('Function Value', fontsize=12)
        ax.set_title(f'Inducing Points x2_offset = {x2_offset}\n' + 
                    f'(Training/Test at x2=0)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_1d_analysis_metrics(results):
    """Plot analysis metrics for the 1D case."""
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    
    # Extract data for plotting
    x2_offsets = []
    temperatures = []
    nll_values = []
    kl_values = []
    mse_values = []
    
    for key, result in sparse_results.items():
        x2_offsets.append(result['x2_offset'])
        temperatures.append(result['temperature'])
        nll_values.append(result['nll'])
        kl_values.append(result['mean_kl'])
        mse_values.append(result['mse'])
    
    # Convert to arrays for easier plotting
    x2_offsets = np.array(x2_offsets)
    temperatures = np.array(temperatures)
    nll_values = np.array(nll_values)
    kl_values = np.array(kl_values)
    mse_values = np.array(mse_values)
    
    # Create focused figure on key metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get exact GP results for reference lines
    exact_result = results['exact']
    exact_nll = exact_result['nll']
    exact_mse = exact_result['mse']
    
    unique_temps = np.unique(temperatures)
    unique_offsets = np.unique(x2_offsets)
    
    # Plot 1: NLL vs x2_offset for different temperatures
    ax = axes[0]
    for temp in unique_temps:
        mask = temperatures == temp
        ax.plot(x2_offsets[mask], nll_values[mask], 'o-', 
                label=f'T={temp}', linewidth=2, markersize=8)
    ax.axhline(y=exact_nll, color='black', linestyle='--', alpha=0.7, 
               linewidth=2, label='Exact GP')
    ax.set_xlabel('Inducing Point x2 Offset')
    ax.set_ylabel('NLL')
    ax.set_title('Predictive NLL vs Inducing Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MSE vs x2_offset for different temperatures
    ax = axes[1]
    for temp in unique_temps:
        mask = temperatures == temp
        ax.plot(x2_offsets[mask], mse_values[mask], 's-', 
                label=f'T={temp}', linewidth=2, markersize=8)
    ax.axhline(y=exact_mse, color='black', linestyle='--', alpha=0.7, 
               linewidth=2, label='Exact GP')
    ax.set_xlabel('Inducing Point x2 Offset')
    ax.set_ylabel('MSE')
    ax.set_title('MSE vs Inducing Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: KL divergence vs x2_offset  
    ax = axes[2]
    for temp in unique_temps:
        mask = temperatures == temp
        ax.plot(x2_offsets[mask], kl_values[mask], '^-', 
                label=f'T={temp}', linewidth=2, markersize=8)
    ax.set_xlabel('Inducing Point x2 Offset')
    ax.set_ylabel('Mean KL from Exact GP')
    ax.set_title('KL Divergence vs Inducing Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

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

def run_cold_posterior_experiment(temperatures, x2_offsets):
    """Run complete cold posterior experiment with sparse GPs - 1D testing."""
    
    print("Cold Posterior Inducing Point Position Experiment (1D Testing)")
    print("=" * 60)
    
    # Generate data with training and testing on x2=0 line, plus 2D visualization
    train_x, train_y, test_x, test_y, vis_x, vis_grid, true_hyperparams = create_synthetic_data(
        n_train=5, n_test=200, n_test_per_dim=25, noise_std=0.5  
    )
    
    print(f"Training data shape: {train_x.shape}")
    print(f"Training data x2 range: [{train_x[:, 1].min():.3f}, {train_x[:, 1].max():.3f}]")
    print(f"Test data shape: {test_x.shape}")
    print(f"Test data x2 range: [{test_x[:, 1].min():.3f}, {test_x[:, 1].max():.3f}]")
    print(f"Visualization grid shape: {vis_x.shape}")
    

    
    results = compare_inducing_positions_and_temperatures(
        train_x, train_y, test_x, test_y, true_hyperparams,
        temperatures, x2_offsets
    )
    
    # Create visualizations - both 2D posterior and 1D analysis
    fig1 = plot_2d_posterior_comparison(train_x, train_y, test_x, vis_x, vis_grid, results)
    fig2 = plot_1d_comparison(train_x, train_y, test_x, test_y, results)
    fig3 = plot_1d_analysis_metrics(results)
    fig4 = plot_losses(results)
    
    plt.show()
    
    return results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    # torch.manual_seed(42)
    np.random.seed(42)
    
        # Compare different inducing positions and temperatures - wider range
    # temperatures = [0.01, 0.1, 1.0, 10.0]  # Very cold to very hot
    # x2_offsets = [0.0,  0.1, 0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # From training line to far away
    temperatures = [0.01, 1.0]  # Very cold to very hot
    x2_offsets = [0.0,  0.5, 1.0]  # From training line to far away

    # Run experiment
    results = run_cold_posterior_experiment(temperatures, x2_offsets)
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 75)
    print("Model".ljust(20) + "MSE".ljust(8) + "NLL".ljust(8) + "Mean KL".ljust(10) + "Final T")
    print("-" * 75)
    
    # Print exact GP first
    exact_result = results['exact']
    print("Exact GP".ljust(20) + 
          f"{exact_result['mse']:.4f}".ljust(8) + 
          f"{exact_result['nll']:.4f}".ljust(8) + 
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
              f"{result['mean_kl']:.4f}".ljust(10) + 
              f"{result['final_temp']:.4f}")
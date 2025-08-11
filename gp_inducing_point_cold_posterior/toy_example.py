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
            learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
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

def train_exact_gp(train_x, train_y, temperature=1.0, lr=0.01, epochs=500, verbose=True):
    """Train exact GP with specified temperature."""
    
    # Initialize model and likelihood
    likelihood = GaussianLikelihood()
    model = ColdPosteriorExactGP(train_x, train_y, likelihood, temperature=temperature)
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Optimizer - collect unique parameters
    all_params = list(model.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    
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

def create_synthetic_data(n_train=200, n_test_per_dim=30, noise_std=0.1, seed=42):
    """Generate synthetic 2D regression data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Training data - random points in [0,1]^2
    train_x = torch.rand(n_train, 2)
    
    # Interesting 2D function: combination of sine waves and gaussian bumps
    def true_function(x):
        x1, x2 = x[:, 0], x[:, 1]
        return (torch.sin(4 * np.pi * x1) * torch.cos(4 * np.pi * x2) + 
                torch.exp(-((x1 - 0.3)**2 + (x2 - 0.7)**2) / 0.1) + 
                0.5 * torch.sin(8 * np.pi * x1 * x2))
    
    train_y = true_function(train_x) + noise_std * torch.randn(n_train)
    
    # Test data - regular grid
    x1_test = torch.linspace(0, 1, n_test_per_dim)
    x2_test = torch.linspace(0, 1, n_test_per_dim)
    X1, X2 = torch.meshgrid(x1_test, x2_test, indexing='ij')
    test_x = torch.stack([X1.flatten(), X2.flatten()], dim=1)
    test_y = true_function(test_x)
    
    return train_x, train_y, test_x, test_y, (X1, X2)

def train_cold_posterior_gp(train_x, train_y, temperature=1.0, num_inducing=8, 
                           lr=0.01, epochs=500, verbose=True):
    """Train sparse GP with specified temperature."""
    
    # Initialize inducing points (random locations in input space)
    inducing_points = torch.rand(num_inducing, train_x.size(1)) * (train_x.max() - train_x.min()) + train_x.min()
    
    # Initialize model and likelihood
    model = ColdPosteriorSparseGP(inducing_points, temperature=temperature)
    likelihood = GaussianLikelihood()
    
    # Set to training mode
    model.train()
    likelihood.train()
    
    # Optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=lr)
    
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

def compare_temperatures(train_x, train_y, test_x, test_y, 
                        temperatures=[0.1, 0.5, 1.0, 2.0], 
                        num_inducing=8, epochs=300):
    """Compare models trained with different temperatures."""
    
    results = {}
    
    print("Training exact GP (T=1.0) as reference...")
    exact_model, exact_likelihood, _ = train_exact_gp(
        train_x, train_y, temperature=1.0, epochs=epochs, verbose=False
    )
    exact_pred_mean, exact_pred_std, _, _ = evaluate_model(exact_model, exact_likelihood, test_x)
    
    print("Training sparse GPs with different temperatures...")
    for temp in temperatures:
        print(f"\nTraining sparse GP with T = {temp}")
        
        model, likelihood, losses = train_cold_posterior_gp(
            train_x, train_y, temperature=temp, 
            num_inducing=num_inducing, epochs=epochs, verbose=False
        )
        
        # Evaluate
        pred_mean, pred_std, lower, upper = evaluate_model(model, likelihood, test_x)
        
        # Compute metrics
        mse = torch.mean((pred_mean - test_y)**2).item()
        nll = -torch.distributions.Normal(pred_mean, pred_std).log_prob(test_y).mean().item()
        
        # Compute KL divergence from exact GP
        kl_from_exact = compute_kl_divergence(pred_mean, pred_std, exact_pred_mean, exact_pred_std)
        mean_kl = torch.mean(kl_from_exact).item()
        
        results[temp] = {
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
            'mean_kl': mean_kl
        }
        
        print(f"Final temperature: {model.temperature.item():.4f}")
        print(f"MSE: {mse:.4f}, NLL: {nll:.4f}, Mean KL from Exact: {mean_kl:.4f}")
    
    # Store exact GP results
    results['exact'] = {
        'model': exact_model,
        'likelihood': exact_likelihood,
        'pred_mean': exact_pred_mean,
        'pred_std': exact_pred_std,
        'kl_from_exact': torch.zeros_like(exact_pred_mean),
        'mean_kl': 0.0
    }
    
    return results

def plot_comparison(train_x, train_y, test_x, test_y, test_grid, results):
    """Plot comparison of different temperature models in 2D."""
    
    # Separate exact from sparse results
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    n_temps = len(sparse_results)
    
    # Create figure with exact GP + sparse GPs + KL divergences
    fig, axes = plt.subplots(3, n_temps + 1, figsize=(4*(n_temps+1), 12))
    if n_temps == 0:
        axes = axes.reshape(-1, 1)
    
    X1, X2 = test_grid
    
    # Plot exact GP first
    exact_result = results['exact']
    exact_mean_2d = exact_result['pred_mean'].reshape(X1.shape)
    exact_std_2d = exact_result['pred_std'].reshape(X1.shape)
    
    # Exact GP mean
    ax = axes[0, 0]
    im = ax.contourf(X1.numpy(), X2.numpy(), exact_mean_2d.numpy(), 
                    levels=20, cmap='viridis')
    ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
               c=train_y.numpy(), s=20, cmap='viridis', 
               edgecolors='white', linewidth=0.5)
    ax.set_title('Exact GP Mean')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Exact GP uncertainty
    ax = axes[1, 0]
    im = ax.contourf(X1.numpy(), X2.numpy(), exact_std_2d.numpy(), 
                    levels=20, cmap='plasma')
    ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
               c='white', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_title('Exact GP Uncertainty')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Empty KL plot for exact GP
    axes[2, 0].text(0.5, 0.5, 'Reference\n(KL = 0)', 
                    ha='center', va='center', transform=axes[2, 0].transAxes,
                    fontsize=12)
    axes[2, 0].set_title('KL from Exact GP')
    axes[2, 0].set_xticks([])
    axes[2, 0].set_yticks([])
    
    # Plot sparse GPs
    for i, (temp, result) in enumerate(sparse_results.items()):
        col_idx = i + 1
        
        # Reshape predictions for plotting
        pred_mean_2d = result['pred_mean'].reshape(X1.shape)
        pred_std_2d = result['pred_std'].reshape(X1.shape)
        kl_2d = result['kl_from_exact'].reshape(X1.shape)
        
        # Plot mean predictions
        ax = axes[0, col_idx]
        im = ax.contourf(X1.numpy(), X2.numpy(), pred_mean_2d.numpy(), 
                        levels=20, cmap='viridis')
        ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
                   c=train_y.numpy(), s=20, cmap='viridis', 
                   edgecolors='white', linewidth=0.5)
        
        # Plot inducing points
        inducing_points = result['model'].variational_strategy.inducing_points.detach()
        ax.scatter(inducing_points[:, 0].numpy(), inducing_points[:, 1].numpy(),
                   c='red', s=50, marker='x', linewidth=2)
        
        ax.set_title(f'Sparse GP Mean (T={temp})\nMSE: {result["mse"]:.4f}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Plot uncertainty
        ax = axes[1, col_idx]
        im = ax.contourf(X1.numpy(), X2.numpy(), pred_std_2d.numpy(), 
                        levels=20, cmap='plasma')
        ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
                   c='white', s=20, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.scatter(inducing_points[:, 0].numpy(), inducing_points[:, 1].numpy(),
                   c='red', s=50, marker='x', linewidth=2)
        
        ax.set_title(f'Sparse GP Uncertainty (T={temp})\nNLL: {result["nll"]:.4f}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Plot KL divergence
        ax = axes[2, col_idx]
        im = ax.contourf(X1.numpy(), X2.numpy(), kl_2d.numpy(), 
                        levels=20, cmap='Reds')
        ax.scatter(train_x[:, 0].numpy(), train_x[:, 1].numpy(), 
                   c='blue', s=20, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.scatter(inducing_points[:, 0].numpy(), inducing_points[:, 1].numpy(),
                   c='blue', s=50, marker='x', linewidth=2)
        
        ax.set_title(f'KL from Exact (T={temp})\nMean KL: {result["mean_kl"]:.4f}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    return fig

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
    
    print("Cold Posterior Effect Experiment with Sparse GPs (2D)")
    print("=" * 55)
    
    # Generate 2D data
    train_x, train_y, test_x, test_y, test_grid = create_synthetic_data(
        n_train=150, n_test_per_dim=25, noise_std=0.1
    )
    
    print(f"Training data shape: {train_x.shape}")
    print(f"Test data shape: {test_x.shape}")
    
    # Compare different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0]
    results = compare_temperatures(
        train_x, train_y, test_x, test_y, 
        temperatures=temperatures, epochs=400
    )
    
    # Create visualizations
    fig1 = plot_comparison(train_x, train_y, test_x, test_y, test_grid, results)
    fig2 = plot_training_curves(results)
    
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
    print("-" * 60)
    print("Model".ljust(15) + "MSE".ljust(8) + "NLL".ljust(8) + "Mean KL".ljust(10) + "Final T")
    print("-" * 60)
    
    # Print exact GP first
    print("Exact GP".ljust(15) + "N/A".ljust(8) + "N/A".ljust(8) + "0.0000".ljust(10) + "1.0000")
    
    # Print sparse GPs
    for temp, result in results.items():
        if temp != 'exact':
            print(f"Sparse T={temp}".ljust(15) + 
                  f"{result['mse']:.4f}".ljust(8) + 
                  f"{result['nll']:.4f}".ljust(8) + 
                  f"{result['mean_kl']:.4f}".ljust(10) + 
                  f"{result['final_temp']:.4f}")
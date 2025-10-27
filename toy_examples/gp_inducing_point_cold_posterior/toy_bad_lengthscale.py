import torch
import numpy as np
import matplotlib.pyplot as plt

from src.gp_utils import create_synthetic_data, compute_kl_divergence, train_exact_gp, evaluate_model, create_inducing_points_1D, train_cold_posterior_gp_with_inducing, plot_2d_posterior_comparison, plot_1d_comparison, plot_losses



def compare_inducing_positions_and_temperatures(train_x, train_y, test_x, test_y, true_hyperparams,
                                               temperatures=[0.1, 1.0], 
                                               bad_lengthscale_factors=[0.5,1,2],
                                               num_inducing=64, epochs=500):
    """Compare models with different inducing point positions and temperatures."""
    

    results = {}
    
    print("Training exact GP (T=1.0) as reference...")
    exact_model, exact_likelihood = train_exact_gp(
        train_x, train_y, true_hyperparams, temperature=1.0, epochs=epochs, verbose=False
    )
    exact_pred_mean, exact_pred_std, _, _ = evaluate_model(exact_model, exact_likelihood, test_x)
    
    # Compute exact GP metrics (no extrapolation since all data is on x2=0)
    exact_mse = torch.mean((exact_pred_mean - test_y)**2).item()
    exact_nll = -torch.distributions.Normal(exact_pred_mean, exact_pred_std).log_prob(test_y).mean().item()
    
    print(f"Exact GP - MSE: {exact_mse:.4f}, NLL: {exact_nll:.4f}")
    
    print(f"\nTraining and testing data all at x2=0")
    print(f"Testing inducing point x2 offsets: {bad_lengthscale_factors}")
    print(f"Testing temperatures: {temperatures}")
    
    for length_scale_factor in bad_lengthscale_factors:
        for temp in temperatures:
            key = f"length_scale_factor={length_scale_factor}_T={temp}"
            print(f"\nTraining {key}")
            
            # Create inducing points at specified x2 offset
            inducing_points = create_inducing_points_1D(num_inducing, 0)
            test_hyperparameters = (true_hyperparams[0]*length_scale_factor, true_hyperparams[1], true_hyperparams[2])
            model, likelihood, losses = train_cold_posterior_gp_with_inducing(
                train_x, train_y, inducing_points, test_hyperparameters, temperature=temp, 
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
                'title': length_scale_factor,
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



def plot_1d_analysis_metrics(results):
    """Plot analysis metrics for the 1D case."""
    sparse_results = {k: v for k, v in results.items() if k != 'exact'}
    
    # Extract data for plotting
    lengthscale_factor = []
    temperatures = []
    nll_values = []
    kl_values = []
    mse_values = []
    
    for key, result in sparse_results.items():
        lengthscale_factor.append(result['title'])
        temperatures.append(result['temperature'])
        nll_values.append(result['nll'])
        kl_values.append(result['mean_kl'])
        mse_values.append(result['mse'])
    
    # Convert to arrays for easier plotting
    lengthscale_factor = np.array(lengthscale_factor)
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
    unique_offsets = np.unique(lengthscale_factor)
    
    # Plot 1: NLL vs x2_offset for different temperatures
    ax = axes[0]
    for temp in unique_temps:
        mask = temperatures == temp
        ax.plot(np.log(lengthscale_factor[mask]), nll_values[mask], 'o-', 
                label=f'T={temp}', linewidth=2, markersize=8)
    ax.axhline(y=exact_nll, color='black', linestyle='--', alpha=0.7, 
               linewidth=2, label='Exact GP')
    ax.set_xlabel('length scale factor')
    ax.set_ylabel('NLL')
    ax.set_title('Predictive NLL vs Inducing Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MSE vs x2_offset for different temperatures
    ax = axes[1]
    for temp in unique_temps:
        mask = temperatures == temp
        ax.plot(np.log(lengthscale_factor[mask]), mse_values[mask], 's-', 
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
        ax.plot(np.log(lengthscale_factor[mask]), kl_values[mask], '^-', 
                label=f'T={temp}', linewidth=2, markersize=8)
    ax.set_xlabel('Inducing Point x2 Offset')
    ax.set_ylabel('Mean KL from Exact GP')
    ax.set_title('KL Divergence vs Inducing Position')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_cold_posterior_experiment(temperatures, bad_lengthscale_factors):
    """Run complete cold posterior experiment with sparse GPs - 1D testing."""
    
    print("Cold Posterior Inducing Point Position Experiment (1D Testing)")
    print("=" * 60)
    
    # Generate data with training and testing on x2=0 line, plus 2D visualization
    train_x, train_y, test_x, test_y, vis_x, vis_grid, true_hyperparams = create_synthetic_data(
        n_train=1024, n_test=200, n_test_per_dim=25, noise_std=0.1  
    )
    
    print(f"Training data shape: {train_x.shape}")
    print(f"Training data x2 range: [{train_x[:, 1].min():.3f}, {train_x[:, 1].max():.3f}]")
    print(f"Test data shape: {test_x.shape}")
    print(f"Test data x2 range: [{test_x[:, 1].min():.3f}, {test_x[:, 1].max():.3f}]")
    print(f"Visualization grid shape: {vis_x.shape}")
    

    
    results = compare_inducing_positions_and_temperatures(
        train_x, train_y, test_x, test_y, true_hyperparams,
        temperatures, bad_lengthscale_factors
    )
    
    # Create visualizations - both 2D posterior and 1D analysis
    # fig1 = plot_2d_posterior_comparison(train_x, train_y, test_x, vis_x, vis_grid, results)
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
    temperatures = [1e-8,1.0]  # Very cold to very hot
    bad_lengthscale_factors = [1,10, 100]  

    # Run experiment
    results = run_cold_posterior_experiment(temperatures, bad_lengthscale_factors)
    
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
        model_name = f"x2={result['title']}, T={result['temperature']}"
        print(model_name.ljust(20) + 
              f"{result['mse']:.4f}".ljust(8) + 
              f"{result['nll']:.4f}".ljust(8) + 
              f"{result['mean_kl']:.4f}".ljust(10) + 
              f"{result['final_temp']:.4f}")
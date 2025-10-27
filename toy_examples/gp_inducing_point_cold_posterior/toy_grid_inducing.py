import torch
import numpy as np
import matplotlib.pyplot as plt

from src.gp_utils import plot_1d_comparison, plot_losses, create_synthetic_data, compute_kl_divergence, create_inducing_points_1D, create_inducing_points_2D, train_exact_gp, evaluate_model, train_cold_posterior_gp_with_inducing



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
            inducing_points = create_inducing_points_1D(
                num_inducing=inducing_config.get('num_inducing', 20),
                x2_offset=inducing_config.get('x2_offset', 0.0),
                x1_range=inducing_config.get('x1_range', (0, 1))
            )
        elif inducing_config['type'] == 'grid':
            inducing_points = create_inducing_points_2D(
                num_x1=inducing_config.get('num_x1', 8),
                num_x2=inducing_config.get('num_x2', 21),  # Default to odd number for symmetry
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
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    
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


def run_variational_experiment():
    """Run the complete variational family exper iment."""
    
    print("Cold Posterior Variational Family Experiment")
    print("=" * 60)
    noise_std=1
    # Generate synthetic data with fewer training points for higher uncertainty
    train_x, train_y, test_x, test_y, vis_x, vis_grid, true_hyperparams = create_synthetic_data(
        n_train=16, n_test=1024, n_test_per_dim=25, noise_std=noise_std, seed=42  # Increased noise too
    )
    
    print(f"Training data shape: {train_x.shape}")
    print(f"Test data shape: {test_x.shape}")
    
    # Define experimental configurations
    temperatures = [0.1, 0.3, 0.6, 1.0]
    variational_families = ['mean_field', 'cholesky']
    
    # Define inducing point configurations
    inducing_configs = [
        {
            'type': 'sparse',
            'name': 'sparse_x2=0',
            'num_inducing': 512,
            'x2_offset': 0.0,
            'x1_range': (-1, 2)
        },
        {
            'type': 'grid', 
            'name': 'grid_2d',
            'num_x1': 6,
            'num_x2': 6,  # Odd number ensures symmetry around x2=0
            'x1_range': (0, 1),
            'x2_extent': 1  
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
    # fig3 = plot_1d_comparison(train_x, train_y, test_x, test_y, results)
    fig4 = plot_losses(results)  # NEW: Add loss curves
    
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
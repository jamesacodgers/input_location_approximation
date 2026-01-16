import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)


def compute_mfvi_analytic(X: torch.Tensor, y: torch.Tensor, temperature: float = 1.0, 
                          noise_std: float = 1.0):
    """
    Compute closed-form mean-field VI solution for Cold Posterior Bayesian linear regression.
    Both likelihood and prior are scaled by 1/T.
    """
    XTX = X.T @ X
    XTy = X.T @ y
    
    # Posterior precision (Cold: 1/T * (Likelihood Precision + Prior Precision))
    # Prior precision is I. Likelihood precision matrix is X^T X / sigma^2.
    # Total precision = 1/T * (X^T X / sigma^2 + I)
    precision = (XTX / (noise_std**2) + torch.eye(X.shape[1])) / temperature
    
    # Posterior mean
    # mu = Precision^{-1} @ (1/T * Likelihood Mean Term)
    # Likelihood Mean Term = X^T y / sigma^2
    # mu = (T * (XTX/sigma^2 + I)^{-1}) @ (1/T * X^T y / sigma^2)
    #    = (XTX/sigma^2 + I)^{-1} @ (X^T y / sigma^2)
    # This is independent of T.
    mu = torch.linalg.solve(precision, XTy / (temperature * noise_std**2))
    
    # Posterior variances (diagonal only for mean-field)
    # Sigma_cold = T * (XTX/sigma^2 + I)^{-1}
    # Diagonal approximation:
    # sigma_sq_i = T / ( (X^T X)_{ii}/sigma^2 + 1 )
    #            = T * sigma^2 / ( (X^T X)_{ii} + sigma^2 )
    XTX_diag = torch.diag(XTX)
    sigma_sq = (temperature * noise_std**2) / (XTX_diag + noise_std**2)
    sigma = torch.sqrt(sigma_sq)
    
    return mu, sigma


def compute_analytic_nll_approximations(mu: torch.Tensor, sigma: torch.Tensor, 
                                      true_weights: torch.Tensor, Sigma_exact: torch.Tensor,
                                      input_std: float, noise_std: float):
    """
    Computes 1st and 2nd order Delta Method approximations of the Expected NLL.
    Returns 3 versions based on how the outer product of error is estimated:
    1. True: Uses (true_weights - mu)(true_weights - mu)^T
    2. Approx: Uses posterior covariance S (from VI)
    3. Exact: Uses exact posterior covariance Sigma_exact
    
    Optimized for the specific case where Input Covariance Sigma = input_std^2 * Ones_Matrix.
    """
    D = len(mu)
    var_x = input_std**2
    
    # S is diagonal posterior covariance (sigma is std dev)
    S_diag = sigma**2
    
    # Common Terms
    # ---------------------------------------------------------
    # Tr(S @ Sigma) = var_x * sum(S_diag)
    tr_S_Sigma = var_x * torch.sum(S_diag)
    
    # Tr((S @ Sigma)^2) = var_x^2 * (sum(S_diag))^2
    tr_S_Sigma_sq = (var_x**2) * (torch.sum(S_diag))**2
    
    # K = Tr(S \Sigma) + \sigma^2 (Denominator)
    K = tr_S_Sigma + noise_std**2
    
    # Helper to compute NLL from N (Numerator) and Correction terms
    def compute_nll(N, correction_term):
        # 1st Order: 0.5 * (N/K + log(2*pi*K))
        nll_1 = 0.5 * (N / K + torch.log(2 * torch.tensor(np.pi) * K))
        
        # 2nd Order Correction
        # Correction = (K-2N)/(2K^3) * Var(b) + 1/(K^2) * Cov(a,b)
        # Var(b) = tr_S_Sigma_sq
        # Cov(a,b) = correction_term (Tr(M Sigma S Sigma))
        
        corr = ((K - 2*N) / (2 * K**3)) * tr_S_Sigma_sq + (1 / K**2) * correction_term
        nll_2 = nll_1 - corr
        return nll_1.item(), nll_2.item()

    # 1. True Weights Approximation
    # ---------------------------------------------------------
    # M = (beta - mu)(beta - mu)^T
    beta_diff = true_weights - mu
    
    # Tr(M @ Sigma) = var_x * (sum(beta_diff))^2
    tr_M_Sigma_true = var_x * (torch.sum(beta_diff))**2
    
    # Tr(M @ Sigma @ S @ Sigma) = var_x^2 * (sum(beta_diff)**2) * sum(S_diag)
    tr_M_Sigma_S_Sigma_true = (var_x**2) * (torch.sum(beta_diff)**2) * torch.sum(S_diag)
    
    nll_true = compute_nll(tr_M_Sigma_true + noise_std**2, tr_M_Sigma_S_Sigma_true)

    # 2. Approximate Posterior Approximation
    # ---------------------------------------------------------
    # M = S (VI Posterior Covariance)
    
    # Tr(M @ Sigma) = Tr(S @ Sigma)
    tr_M_Sigma_approx = tr_S_Sigma
    
    # Tr(M @ Sigma @ S @ Sigma) = Tr(S @ Sigma @ S @ Sigma) = Tr((S @ Sigma)^2)
    tr_M_Sigma_S_Sigma_approx = tr_S_Sigma_sq
    
    nll_approx = compute_nll(tr_M_Sigma_approx + noise_std**2, tr_M_Sigma_S_Sigma_approx)

    # 3. Exact Posterior Approximation
    # ---------------------------------------------------------
    # M = Sigma_exact
    
    # Tr(M @ Sigma) = Tr(Sigma_exact @ var_x * 1 * 1.T) = var_x * sum(Sigma_exact)
    tr_M_Sigma_exact = var_x * torch.sum(Sigma_exact)
    
    # Tr(M @ Sigma @ S @ Sigma) = Tr(Sigma_exact @ Sigma @ S @ Sigma)
    # Sigma @ S @ Sigma = var_x^2 * sum(S_diag) * 1 * 1.T
    # So Tr(...) = var_x^2 * sum(S_diag) * sum(Sigma_exact)
    tr_M_Sigma_S_Sigma_exact = (var_x**2) * torch.sum(S_diag) * torch.sum(Sigma_exact)
    
    nll_exact = compute_nll(tr_M_Sigma_exact + noise_std**2, tr_M_Sigma_S_Sigma_exact)
    
    return nll_true, nll_approx, nll_exact


def compute_exact_posterior(X: torch.Tensor, y: torch.Tensor, noise_std: float = 1.0):
    precision_lik = X.T @ X / (noise_std ** 2)
    precision_post = precision_lik + torch.eye(X.shape[1])
    Sigma_post = torch.inverse(precision_post)
    mu_post = Sigma_post @ (X.T @ y / (noise_std ** 2))
    return mu_post, Sigma_post


def test_nll_mfvi(X_test: torch.Tensor, y_test: torch.Tensor, 
                  mu: torch.Tensor, sigma: torch.Tensor,
                  noise_std: float = 1.0, n_samples: int = 500):
    eps = torch.randn(n_samples, len(mu))
    weights = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
    y_pred = torch.matmul(weights, X_test.T)
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
    nll = -log_pred.mean().item()
    return nll


def test_nll_exact(X_test: torch.Tensor, y_test: torch.Tensor, 
                   mu_post: torch.Tensor, Sigma_post: torch.Tensor,
                   noise_std: float = 1.0, n_samples: int = 500):
    dist = torch.distributions.MultivariateNormal(mu_post, Sigma_post)
    weights = dist.sample((n_samples,))
    y_pred = torch.matmul(weights, X_test.T)
    log_probs = Normal(y_pred, noise_std).log_prob(y_test.unsqueeze(0))
    log_pred = torch.logsumexp(log_probs, dim=0) - np.log(n_samples)
    nll = -log_pred.mean().item()
    return nll


def generate_data(n_samples: int = 100, n_dims: int = 10, input_std: float = 1.0, 
                  noise_std: float = 1.0, seed: int = 42):
    torch.manual_seed(seed)
    true_weights = torch.randn(n_dims)
    x1 = torch.randn(n_samples) * input_std
    X = x1.unsqueeze(1).repeat(1, n_dims)
    y = X @ true_weights + torch.randn(n_samples) * noise_std
    return X, y, true_weights


def generate_test_data(true_weights: torch.Tensor, n_samples: int = 100, 
                       input_std: float = 1.0, noise_std: float = 1.0, 
                       seed: int = 123):
    torch.manual_seed(seed)
    n_dims = len(true_weights)
    x1 = torch.randn(n_samples) * input_std
    X_test = x1.unsqueeze(1).repeat(1, n_dims)
    y_test = X_test @ true_weights + torch.randn(n_samples) * noise_std
    return X_test, y_test


# Main experiment
if __name__ == "__main__":
    # Configuration
    # Configuration
    N_DIMS = 1000
    input_stds = [1.0]
    temperatures = np.logspace(-4, 6, 50)
    NOISE_STD = 0.1
    N_SAMPLES = 100
    
    all_results = []
    
    for input_std in input_stds:
        print(f"\n{'='*60}")
        print(f"INPUT STD = {input_std:.1f} (variance = {input_std**2:.1f})")
        print('='*60)
        
        X_train, y_train, true_weights = generate_data(n_samples=5, n_dims=N_DIMS, input_std=input_std, noise_std=NOISE_STD)
        X_test, y_test = generate_test_data(true_weights, n_samples=100_000, input_std=input_std, noise_std=NOISE_STD)
        
        mu_exact, Sigma_exact = compute_exact_posterior(X_train, y_train, noise_std=NOISE_STD)
        exact_nll = test_nll_exact(X_test, y_test, mu_exact, Sigma_exact, noise_std=NOISE_STD)
        
        print(f"Exact posterior test NLL: {exact_nll:.4f}")
        
        results = {
            'input_std': input_std, 
            'temperatures': [], 
            'test_nlls': [], 
            'analytic_true': [], 
            'analytic_approx': [],
            'analytic_exact': [],

            'var_metric_mfvi': [],
            'exact_nll': exact_nll,
            'X_train': X_train,
            'y_train': y_train,
            'true_weights': true_weights,
            'X_test': X_test
        }
        
        for temp in temperatures:
            mu, sigma = compute_mfvi_analytic(X_train, y_train, temperature=temp, noise_std=NOISE_STD)
            
            # Monte Carlo Test NLL
            test_nll = test_nll_mfvi(X_test, y_test, mu, sigma, noise_std=NOISE_STD)
            

            # Marginal Variance Metric: 1^T S 1
            # For MFVI (diagonal S): sum(sigma^2)
            var_metric_mfvi = torch.sum(sigma**2).item()
            
            # Analytic Approximations
            nll_true, nll_approx, nll_exact = compute_analytic_nll_approximations(
                mu, sigma, true_weights, Sigma_exact, input_std, NOISE_STD
            )
            
            results['temperatures'].append(temp)
            results['test_nlls'].append(test_nll)
            results['analytic_true'].append(nll_true)
            results['analytic_approx'].append(nll_approx)
            results['analytic_exact'].append(nll_exact)

            results['var_metric_mfvi'].append(var_metric_mfvi)
            
            # Ratio Metric: Mean of (v^T S v) / (v^T Sigma_exact v) for eigenvectors v of Sigma_exact
            # Compute eigenvectors of Sigma_exact (outside loop for efficiency if possible, but temp loop is short)
            # Actually Sigma_exact is constant for a given input_std, so we can compute it once.
            # But let's compute it here for simplicity or move it out if needed.
            # Wait, Sigma_exact is already computed outside the loop.
            vals_ex, vecs_ex = torch.linalg.eigh(Sigma_exact)
            
            # Numerator: v^T S v = sum_i v_i^2 sigma_i^2 (since S is diagonal with sigma^2)
            # vecs_ex is (D, D), columns are eigenvectors.
            # (vecs_ex**2).T @ (sigma**2) gives a vector of size D where each element is v_k^T S v_k
            numerator = (vecs_ex**2).T @ (sigma**2)
            
            # Denominator: v^T Sigma_exact v = eigenvalue corresponding to v
            denominator = vals_ex
            
            # Ratio
            ratios = numerator / denominator
            mean_ratio = torch.mean(ratios).item()
            
            print(f"T = {temp:.1e}: MC NLL = {test_nll:.4f} | True(2nd) = {nll_true[1]:.4f} | Mean Ratio = {mean_ratio:.4f}")
        
        # Exact Posterior Variance Metric
        # 1^T Sigma_exact 1 = sum(Sigma_exact)
        var_metric_exact = torch.sum(Sigma_exact).item()
        results['var_metric_exact'] = var_metric_exact
        
        all_results.append(results)
    
    # Create plots
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.
        """
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        
        width, height = 2 * n_std * np.sqrt(vals)
        
        # Generate points for the ellipse
        t = np.linspace(0, 2*np.pi, 5000)
        ell_x = (width / 2) * np.cos(t)
        ell_y = (height / 2) * np.sin(t)
        
        # Rotate points
        angle = np.radians(theta)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        ell_coords = np.dot(R, np.array([ell_x, ell_y]))
        
        # Translate points
        ell_coords[0, :] += mean[0]
        ell_coords[1, :] += mean[1]
        
        # Extract style arguments that ax.plot understands
        plot_kwargs = {}
        if 'edgecolor' in kwargs:
            plot_kwargs['color'] = kwargs['edgecolor']
        elif 'color' in kwargs:
            plot_kwargs['color'] = kwargs['color']
            
        if 'linestyle' in kwargs:
            plot_kwargs['linestyle'] = kwargs['linestyle']
            
        if 'linewidth' in kwargs:
            plot_kwargs['linewidth'] = kwargs['linewidth']
            
        if 'alpha' in kwargs:
            plot_kwargs['alpha'] = kwargs['alpha']
            
        if 'label' in kwargs:
            plot_kwargs['label'] = kwargs['label']

        return ax.plot(ell_coords[0, :], ell_coords[1, :], **plot_kwargs)

    n_vars = len(input_stds)
    # Increase figure height to accommodate 2 rows per input_std
    # 2 rows * n_vars
    fig = plt.figure(figsize=(24, 6 * 2 * n_vars))
    
    np.random.seed(42)
    dim1, dim2 = np.random.choice(N_DIMS, 2, replace=False)
    
    for i, res in enumerate(all_results):
        # Define colormap for temperature
        cmap = plt.get_cmap('coolwarm')
        norm = plt.Normalize(vmin=np.log10(min(res['temperatures'])), vmax=np.log10(max(res['temperatures'])))

        # Row 1, Col 1: Training Data (First 2 Dims)
        ax1 = plt.subplot(2 * n_vars, 3, i * 6 + 1)
        X_train = res['X_train']
        ax1.scatter(X_train[:, 0].numpy(), X_train[:, 1].numpy(), alpha=0.5, color='blue')
        ax1.set_xlabel('Dim 0')
        ax1.set_ylabel('Dim 1')
        ax1.set_title('(a)', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Row 1, Col 2: NLL Approximations
        ax2 = plt.subplot(2 * n_vars, 3, i * 6 + 2)
        
        # Plot MC Estimate
        ax2.plot(res['temperatures'], res['test_nlls'], 'o-', color='black', 
                 linewidth=2, markersize=4, label='MC Estimate', alpha=0.5)
        
        # Extract 1st and 2nd order approximations
        true_1st = [x[0] for x in res['analytic_true']]
        true_2nd = [x[1] for x in res['analytic_true']]
        
        approx_1st = [x[0] for x in res['analytic_approx']]
        approx_2nd = [x[1] for x in res['analytic_approx']]
        
        exact_1st = [x[0] for x in res['analytic_exact']]
        exact_2nd = [x[1] for x in res['analytic_exact']]
        
        
        # Plot Analytic Approximations
        # True Weights
        # ax2.plot(res['temperatures'], true_1st, '--', color='blue', alpha=0.4,
        #          linewidth=1, label='True (1st)')
        # ax2.plot(res['temperatures'], true_2nd, '-', color='blue', 
        #          linewidth=2, label='True (2nd)')
        
        # # Approx Posterior
        # ax2.plot(res['temperatures'], approx_1st, '--', color='orange', alpha=0.4,
        #          linewidth=1, label='Approx (1st)')
        # ax2.plot(res['temperatures'], approx_2nd, '-', color='orange', 
        #          linewidth=2, label='Approx (2nd)')
        
        # # Exact Posterior
        # ax2.plot(res['temperatures'], exact_1st, '--', color='red', alpha=0.4,
        #          linewidth=1, label='Exact (1st)')
        # ax2.plot(res['temperatures'], exact_2nd, '-', color='red', 
        #          linewidth=2, label='Exact (2nd)')
                 

        
        ax2.axhline(y=res['exact_nll'], color='green', linestyle='-', 
                    linewidth=1, alpha=0.5, label='Exact Posterior NLL')
        ax2.axvline(x=1.0, color='grey', linestyle='--', alpha=0.5, label='T=1')
        
        ax2.set_xlabel('Temperature', fontsize=10)
        ax2.set_ylabel('Test NLL', fontsize=10)
        ax2.set_xscale('log')
        ax2.set_title('(b)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8, ncol=2)
        
        # Row 1, Col 3: Posterior Predictive Densities (Scalar)
        ax3 = plt.subplot(2 * n_vars, 3, i * 6 + 3)
        
        # Define x-axis range
        # Use max variance to determine range (including noise)
        max_var = max(max(res['var_metric_mfvi']), res['var_metric_exact']) + NOISE_STD**2
        max_std = np.sqrt(max_var)
        x_range = np.linspace(-4 * max_std, 4 * max_std, 1000)
        
        # Plot Exact Posterior Predictive Density
        exact_std = np.sqrt(res['var_metric_exact'] + NOISE_STD**2)
        exact_pdf = (1 / (exact_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / exact_std)**2)
        ax3.plot(x_range, exact_pdf, color='green', linewidth=3, label='Exact Posterior')
        
        # Plot MFVI Densities for different temperatures
        # Use a colormap
        cmap = plt.get_cmap('coolwarm')
        norm = plt.Normalize(vmin=np.log10(min(res['temperatures'])), vmax=np.log10(max(res['temperatures'])))
        
        for t, temp in enumerate(res['temperatures']):
            mfvi_std = np.sqrt(res['var_metric_mfvi'][t] + NOISE_STD**2)
            mfvi_pdf = (1 / (mfvi_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / mfvi_std)**2)
            
            color = cmap(norm(np.log10(temp)))
            # Highlight T=1
            if abs(temp - 1.0) < 1e-6:
                ax3.plot(x_range, mfvi_pdf, color='black', linestyle='--', linewidth=2, label='T=1')
            else:
                # Plot only a subset or all with transparency
                ax3.plot(x_range, mfvi_pdf, color=color, alpha=0.9, linewidth=1)
                
        ax3.set_xlabel('Value')
        ax3.set_ylabel('Density')
        ax3.set_title(f'(c)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for temperature
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3)
        cbar.set_label('Log10(Temperature)')
        ax3.legend(fontsize=8)

        # Row 2, Col 1: Input Space -> Weights Space with Ellipses (Projected)
        ax4 = plt.subplot(2 * n_vars, 3, i * 6 + 4)
        
        # Define Projection Matrix P
        D = N_DIMS
        # u1: Normalized ones vector
        ones_vec = torch.ones(D)
        u1 = ones_vec / torch.norm(ones_vec)
        
        # u2: Normalized orthogonal vector
        # Use a balanced vector with half 1s and half -1s
        v = torch.ones(D)
        v[D//2:] = -1.0
        # Ensure it is orthogonal to ones (sum should be 0)
        if D % 2 != 0:
            pass
            
        u2 = v / torch.norm(v)
        
        # Projection Matrix (2, D)
        P = torch.stack([u1, u2])
        
        # Project True Weights
        true_w = res['true_weights']
        true_w_proj = (P @ true_w).numpy()
        ax4.scatter(true_w_proj[0], true_w_proj[1], marker='*', s=200, color='black', label='True Weights', zorder=10)
        
        # Recompute Exact Posterior for this input_std
        mu_ex, Sigma_ex = compute_exact_posterior(res['X_train'], res['y_train'], noise_std=NOISE_STD)
        
        # Project Exact Posterior
        mu_ex_proj = (P @ mu_ex).numpy()
        Sigma_ex_proj = (P @ Sigma_ex @ P.T).numpy()
        
        # Draw Exact Ellipse (95% CI -> chi2_val = 5.991)
        n_std_95 = np.sqrt(5.991)
        confidence_ellipse(mu_ex_proj, Sigma_ex_proj, ax4, n_std=n_std_95, edgecolor='green', linewidth=2, label='Exact 95%')
        
        # Plot MFVI Ellipses
        for t, temp in enumerate(res['temperatures']):
             # Recompute MFVI
             mu_mfvi, sigma_mfvi = compute_mfvi_analytic(res['X_train'], res['y_train'], temperature=temp, noise_std=NOISE_STD)
             
             # Project MFVI Posterior
             mu_mfvi_proj = (P @ mu_mfvi).numpy()
             Sigma_mfvi = torch.diag(sigma_mfvi**2)
             Sigma_mfvi_proj = (P @ Sigma_mfvi @ P.T).numpy()
             
             color = cmap(norm(np.log10(temp)))
             if abs(temp - 1.0) < 1e-6:
                 confidence_ellipse(mu_mfvi_proj, Sigma_mfvi_proj, ax4, n_std=n_std_95, edgecolor='black', linestyle='--', linewidth=2, label='MFVI T=1')
             else:
                 confidence_ellipse(mu_mfvi_proj, Sigma_mfvi_proj, ax4, n_std=n_std_95, edgecolor=color, alpha=0.9, linewidth=1)

        ax4.set_xlabel('In data span')
        ax4.set_ylabel('Orthogonal to data span')
        ax4.set_title(f'(d)', fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)

        # ---------------------------------------------------------
        # New Plots: Predictive Variance along directions
        # ---------------------------------------------------------
        
        # Prepare directions
        D = N_DIMS
        
        # Eta range: covers typical input range
        # Norm of x = |eta| * sqrt(D)
        # We want |eta| * sqrt(D) ~ 3 * input_std
        # So eta ~ 3 * input_std / sqrt(D)
        # But user said "eta * torch.ones". If eta is O(1), then x is huge.
        # Let's assume user wants to see the effect over a reasonable range.
        # If we use eta in [-3, 3], and D=1000, x is very far from data.
        # But maybe that's the point? To see OOD behavior?
        # Let's use a range that makes sense relative to the data distribution.
        # Data is N(0, input_std^2).
        # Along diagonal, std is input_std.
        # So eta should be comparable to input_std.
        eta_range = np.linspace(-3 * res['input_std'], 3 * res['input_std'], 100)
        
        # 1. Ones Direction
        # x = eta * ones
        # Variance = x^T Sigma x + noise^2
        #          = eta^2 * (ones^T Sigma ones) + noise^2
        
        # Precompute Sigma_exact term
        # ones^T Sigma_exact ones = sum(Sigma_exact)
        var_exact_ones = res['var_metric_exact'] # This is sum(Sigma_exact)
        
        # Row 2, Col 2: Pred Var along Ones
        ax5 = plt.subplot(2 * n_vars, 3, i * 6 + 5)
        
        pred_var_exact_ones = (eta_range**2) * var_exact_ones + NOISE_STD**2
        ax5.plot(eta_range, pred_var_exact_ones, color='green', linewidth=3, label='Exact Posterior')
        
        for t, temp in enumerate(res['temperatures']):
            # For MFVI, S is diagonal.
            # 1^T S 1 = sum(sigma^2) = var_metric_mfvi
            var_mfvi_ones = res['var_metric_mfvi'][t]
            pred_var_mfvi = (eta_range**2) * var_mfvi_ones + NOISE_STD**2
            
            color = cmap(norm(np.log10(temp)))
            if abs(temp - 1.0) < 1e-6:
                ax5.plot(eta_range, pred_var_mfvi, color='black', linestyle='--', linewidth=2, label='T=1')
            else:
                ax5.plot(eta_range, pred_var_mfvi, color=color, alpha=0.9, linewidth=1)
                
        ax5.set_xlabel(r'$\eta$ ')
        ax5.set_ylabel('Predictive Variance')
        ax5.set_title(r'(e)', fontsize=11)
        ax5.grid(True, alpha=0.3)
        
        # 2. Orthogonal Direction
        # Use the same u2 as in ax2, scaled to have norm sqrt(D)
        orth_vec = u2 * np.sqrt(D)
        
        # Precompute quadratic forms
        # x = eta * orth_vec
        # x^T S x = eta^2 * (orth_vec^T S orth_vec)
        
        # Exact
        # Recompute Sigma_exact
        # Need y_train for mu, but we only need Sigma here.
        # But wait, we need y_train for the ellipses earlier.
        # So we MUST have y_train.
        mu_ex, Sigma_ex = compute_exact_posterior(res['X_train'], res['y_train'], noise_std=NOISE_STD) 
        
        term_exact_orth = (orth_vec @ Sigma_ex @ orth_vec).item()
        pred_var_exact_orth = (eta_range**2) * term_exact_orth + NOISE_STD**2
        
        # Row 2, Col 3: Pred Var along Orthogonal
        ax6 = plt.subplot(2 * n_vars, 3, i * 6 + 6)
        
        ax6.plot(eta_range, pred_var_exact_orth, color='green', linewidth=3, label='Exact Posterior')
        
        for t, temp in enumerate(res['temperatures']):
            # MFVI S is diagonal.
            # orth_vec^T S orth_vec = sum(orth_vec_i^2 * sigma_i^2)
            # Recompute sigma
            mu_mfvi, sigma_mfvi = compute_mfvi_analytic(res['X_train'], res['y_train'], temperature=temp, noise_std=NOISE_STD)
            
            term_mfvi_orth = torch.sum((orth_vec**2) * (sigma_mfvi**2)).item()
            pred_var_mfvi = (eta_range**2) * term_mfvi_orth + NOISE_STD**2
            
            color = cmap(norm(np.log10(temp)))
            if abs(temp - 1.0) < 1e-6:
                ax6.plot(eta_range, pred_var_mfvi, color='black', linestyle='--', linewidth=2, label='T=1')
            else:
                ax6.plot(eta_range, pred_var_mfvi, color=color, alpha=0.9, linewidth=1)
                
        ax6.set_xlabel(r'$\zeta$')
        ax6.set_ylabel('Predictive Variance')
        ax6.set_title('(f)', fontsize=11)
        ax6.grid(True, alpha=0.3)
        

    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    plt.savefig("toy_examples/cold_posterior_linear_regression/analytic_cold_comparison.pdf", bbox_inches="tight")
    plt.show()

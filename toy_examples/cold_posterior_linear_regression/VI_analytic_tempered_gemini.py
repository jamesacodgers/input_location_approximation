import torch
import numpy as np
import matplotlib.pyplot as plt

# ICML/LaTeX Formatting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# ICML Formatting Constants
LBL_FS = 18
TTL_FS = 20
TICK_FS = 14
LEG_FS = 14
LW_EXACT = 3
LW_MFVI = 2.5
LW_MAP = 2
LW_PRIOR = 2.5
from torch.distributions import Normal

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)


def compute_mfvi_analytic(x_vec: torch.Tensor, y: torch.Tensor, n_dims: int,
                          temperature: float = 1.0, noise_std: float = 1.0):
    r"""
    Compute closed-form mean-field VI solution for the COLD Bayesian linear regression posterior.
    p_T(w) \propto [p(y|w) p(w)]^(1/T)
    """
    D = n_dims
    noise_var = noise_std**2
    norm_sq = torch.sum(x_vec**2)
    
    # Cold Posterior Mean is same as standard MAP mean (T=1)
    mu_val = torch.dot(x_vec, y) / (D * norm_sq + noise_var)
    mu = torch.full((D,), mu_val)
    
    # Cold MFVI Variance: v_i = T * sigma^2 / (||x_i||^2 + sigma^2)
    sigma_sq_val = temperature * noise_var / (norm_sq + noise_var)
    sigma = torch.full((D,), torch.sqrt(sigma_sq_val))
    
    return mu, sigma


def compute_analytic_nll_approximations(mu: torch.Tensor, sigma: torch.Tensor, 
                                      true_weights: torch.Tensor, sum_Sigma_exact: float,
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
    
    # Tr(M @ Sigma @ S @ Sigma) = var_x^2 * (sum(beta_diff))^2 * sum(S_diag)
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
    tr_M_Sigma_exact = var_x * sum_Sigma_exact
    
    # Tr(M @ Sigma @ S @ Sigma) = Tr(Sigma_exact @ Sigma @ S @ Sigma)
    # Sigma @ S @ Sigma = var_x^2 * sum(S_diag) * 1 * 1.T
    # So Tr(...) = var_x^2 * sum(S_diag) * sum(Sigma_exact)
    tr_M_Sigma_S_Sigma_exact = (var_x**2) * torch.sum(S_diag) * sum_Sigma_exact
    
    nll_exact = compute_nll(tr_M_Sigma_exact + noise_std**2, tr_M_Sigma_S_Sigma_exact)
    
    return nll_true, nll_approx, nll_exact


def compute_exact_posterior(x_vec: torch.Tensor, y: torch.Tensor, n_dims: int, 
                           noise_std: float = 1.0):
    """
    Computes exact posterior mean and sum of all elements of covariance.
    Uses symmetry and Woodbury identity for O(N) computation.
    """
    N = len(x_vec)
    D = n_dims
    noise_var = noise_std ** 2
    
    # All columns of X are identical: X = x_vec @ 1.T
    norm_sq = torch.sum(x_vec**2)
    
    # mu = (x^T y) / (D ||x||^2 + noise_var) * 1
    mu_val = torch.dot(x_vec, y) / (D * norm_sq + noise_var)
    mu_post = torch.full((D,), mu_val)
    
    # sum(Sigma) = 1.T (I - \lambda/(1 + \lambda D) 1 1.T) 1  where \lambda = ||x||^2 / noise_var
    #            = D / (1 + \lambda D) = D * noise_var / (noise_var + D ||x||^2)
    sum_Sigma_post = (D * noise_var) / (noise_var + D * norm_sq)
    
    return mu_post, sum_Sigma_post.item()


def test_nll_analytic(x_vec: torch.Tensor, y_test: torch.Tensor, 
                      mu: torch.Tensor, var_sum: float,
                      noise_std: float = 1.0):
    """
    Analytical Test NLL for linear Gaussian model.
    var_sum is 1^T Sigma 1.
    """
    # Predictive mean M = x_test^T mu
    # Predictive variance V = x_test^T Sigma x_test + sigma_noise^2
    # Since x_test = x_val * 1, M = x_val * (1^T mu), V = x_val^2 * (1^T Sigma 1) + noise_var
    
    mu_sum = torch.sum(mu)
    
    M = x_vec * mu_sum
    V = (x_vec**2) * var_sum + noise_std**2
    
    nll = 0.5 * torch.log(2 * np.pi * V) + 0.5 * (y_test - M)**2 / V
    return nll.mean().item()


def generate_data(n_samples: int = 100, n_dims: int = 10, input_std: float = 1.0, 
                  noise_std: float = 1.0, seed: int = 42):
    torch.manual_seed(seed)
    true_weights = torch.randn(n_dims)
    x1 = torch.randn(n_samples) * input_std
    # X = x1.unsqueeze(1).repeat(1, n_dims)
    # Optimized: only return the representative vector
    y = x1 * true_weights.sum() + torch.randn(n_samples) * noise_std
    return x1, y, true_weights


def generate_test_data(true_weights: torch.Tensor, n_samples: int = 100, 
                       input_std: float = 1.0, noise_std: float = 1.0, 
                       seed: int = 123):
    torch.manual_seed(seed)
    n_dims = len(true_weights)
    x1 = torch.randn(n_samples) * input_std
    y_test = x1 * true_weights.sum() + torch.randn(n_samples) * noise_std
    return x1, y_test


# Main experiment
if __name__ == "__main__":
    # Configuration
    n_dims_list = [2**1, 2**10]
    N_REPEATS = 10000  # Number of repeats for mean/CI
    temperatures = np.logspace(-4, 0, 50)
    NOISE_STD = 0.3
    N_SAMPLES = 8
    
    all_results = []
    
    for n_dims in n_dims_list:
        print(f"\n{'='*60}")
        print(f"N_DIMS = {n_dims}")
        print('='*60)
        
        # Scaling input std such that prior predictive var = 1.0
        # x^* = 1/sqrt(P) * 1. Sum of components = 1/sqrt(P) * P = sqrt(P)
        # Var(y) = (input_std * sum(w))^2 + noise_var
        # Prior Var(sum(w)) = P
        # So (input_std * sqrt(P))^2 = 1 => input_std^2 * P = 1 => input_std = 1/sqrt(P)
        input_std = 1.0 / np.sqrt(n_dims)
        
        # Data structures to store repeats
        repeat_mfvi_nlls = [] # List of arrays (one per repeat)
        repeat_exact_nlls_t1 = []
        repeat_map_nlls = []
        
        # Store the last run's data for predictive density plot
        last_run_X_train = None
        last_run_y_train = None
        last_run_sum_Sigma_exact_1 = None

        for run_idx in range(N_REPEATS):
            seed_train = 42 + run_idx
            seed_test = 123 + run_idx
            X_train, y_train, true_weights = generate_data(n_samples=N_SAMPLES, n_dims=n_dims, input_std=input_std, noise_std=NOISE_STD, seed=seed_train)
            X_test, y_test = generate_test_data(true_weights, n_samples=1_000, input_std=input_std, noise_std=NOISE_STD, seed=seed_test)
            
            mu_exact, sum_Sigma_exact_1 = compute_exact_posterior(X_train, y_train, n_dims, noise_std=NOISE_STD)
            exact_nll_t1 = test_nll_analytic(X_test, y_test, mu_exact, sum_Sigma_exact_1, noise_std=NOISE_STD)
            map_nll = test_nll_analytic(X_test, y_test, mu_exact, 0.0, noise_std=NOISE_STD)
            
            run_mfvi_nlls = []
            for temp in temperatures:
                mu, sigma = compute_mfvi_analytic(X_train, y_train, n_dims, temperature=temp, noise_std=NOISE_STD)
                var_metric_mfvi = torch.sum(sigma).item()
                test_nll = test_nll_analytic(X_test, y_test, mu, var_metric_mfvi, noise_std=NOISE_STD)
                run_mfvi_nlls.append(test_nll)
            
            repeat_mfvi_nlls.append(run_mfvi_nlls)
            repeat_exact_nlls_t1.append(exact_nll_t1)
            repeat_map_nlls.append(map_nll)
            print(f"Run {run_idx+1}/{N_REPEATS} complete. (D={n_dims})")

            # Store data from the last run for the predictive density plot
            if run_idx == N_REPEATS - 1:
                last_run_X_train = X_train
                last_run_y_train = y_train
                last_run_sum_Sigma_exact_1 = sum_Sigma_exact_1

        # Compute statistics
        mfvi_nlls = np.array(repeat_mfvi_nlls)
        mean_mfvi_nll = np.mean(mfvi_nlls, axis=0)
        sem_mfvi_nll = np.std(mfvi_nlls, axis=0) / np.sqrt(N_REPEATS)
        
        mean_exact_nll = np.mean(repeat_exact_nlls_t1)
        sem_exact_nll = np.std(repeat_exact_nlls_t1) / np.sqrt(N_REPEATS)
        
        mean_map_nll = np.mean(repeat_map_nlls)
        sem_map_nll = np.std(repeat_map_nlls) / np.sqrt(N_REPEATS)
        
        # For the 3rd plot, we still need one representative run for variances
        # We'll just use the last run's data for the 3rd plot distributions
        results = {
            'input_std': input_std, 
            'temperatures': temperatures, 
            'mean_mfvi_nll': mean_mfvi_nll,
            'sem_mfvi_nll': sem_mfvi_nll,
            'mean_exact_nll': mean_exact_nll,
            'sem_exact_nll': sem_exact_nll,
            'mean_map_nll': mean_map_nll,
            'sem_map_nll': sem_map_nll,
            'var_metric_mfvi': [torch.sum(compute_mfvi_analytic(last_run_X_train, last_run_y_train, n_dims, temperature=t, noise_std=NOISE_STD)[1]**2).item() for t in temperatures],
            'sum_Sigma_exact_1': last_run_sum_Sigma_exact_1,
            'n_dims': n_dims
        }
        all_results.append(results)
    
    # Create plots
    n_total_rows = len(all_results)
    fig = plt.figure(figsize=(16, 5 * n_total_rows))
    
    np.random.seed(42)
    # n_dims is now specific to each result, and since they are symmetric, 
    # the choice of dimensions doesn't matter for the scatter plot (which now plots representative column).
    
    for i, res in enumerate(all_results):
        # 1. NLL performance
        ax1 = plt.subplot(n_total_rows, 2, i * 2 + 1)
        
        # Theoretical Critical Temperature
        # T_crit = D / (1 + (D-1) * (1 + N/sigma^2)^-1)
        N_val = float(N_SAMPLES)
        D_val = float(res['n_dims'])
        t_crit = (1.0 + (D_val - 1.0) * (1.0 + N_val / (NOISE_STD**2))**(-1))/D_val
        
        # Plot MFVI Curve and CI
        ax1.plot(res['temperatures'], res['mean_mfvi_nll'], '-', color='black', 
                 linewidth=LW_MFVI, label='MFVI NLL', alpha=0.9)
        ax1.fill_between(res['temperatures'], 
                         res['mean_mfvi_nll'] - 1.96 * res['sem_mfvi_nll'],
                         res['mean_mfvi_nll'] + 1.96 * res['sem_mfvi_nll'],
                         color='black', alpha=0.15, label=r'95\% CI (Mean)')
        
        # Exact Post Baseline
        ax1.axhline(y=res['mean_exact_nll'], color='#2ca02c', linestyle='--', 
                    linewidth=LW_EXACT, alpha=0.8, label='Exact Post ($T=1$)')
        ax1.axhspan(res['mean_exact_nll'] - 1.96 * res['sem_exact_nll'],
                    res['mean_exact_nll'] + 1.96 * res['sem_exact_nll'],
                    color='#2ca02c', alpha=0.1)
        
        # MAP Baseline
        ax1.axhline(y=res['mean_map_nll'], color='#d62728', linestyle='--', 
                    linewidth=LW_MAP, alpha=0.8, label='MAP NLL (Noise Only)')
        ax1.axhspan(res['mean_map_nll'] - 1.96 * res['sem_map_nll'],
                    res['mean_map_nll'] + 1.96 * res['sem_map_nll'],
                    color='#d62728', alpha=0.1)
        ax1.axvline(x=1.0, color='grey', linestyle='--', alpha=0.5, label='$T=1$')
        ax1.axvline(x=t_crit, color='blue', linestyle=':', linewidth=2, alpha=0.8, label=r'$T_{\mathrm{crit}}$')
        
        ax1.set_xlabel('Temperature $T$', fontsize=LBL_FS)
        ax1.set_ylabel('Test NLL', fontsize=LBL_FS)
        ax1.set_xscale('log')
        ax1.set_title(f'NLL vs Temperature ($P={res["n_dims"]}, N_{{runs}}={N_REPEATS}$)', fontsize=TTL_FS)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=LEG_FS, ncol=2)
        ax1.tick_params(labelsize=TICK_FS)
        
        # 2. Predictive Densities
        ax3 = plt.subplot(n_total_rows, 2, i * 2 + 2)
        
        # evaluation point x^* = 1/sqrt(D) * 1, so predictive variance is 1/D * (1^T Sigma 1) + noise
        D_val = float(res['n_dims'])
        var_pred_prior = 1.0 # (P * input_std^2) / P * 1 = 1 if input_std = 1/sqrt(P)
        
        # Exact Posterior at T=1
        var_pred_exact_t1 = res['sum_Sigma_exact_1'] / D_val
        var_pred_mfvi = [v / D_val for v in res['var_metric_mfvi']]
        
        max_var = max(max(var_pred_mfvi), var_pred_exact_t1, var_pred_prior) + NOISE_STD**2
        max_std = np.sqrt(max_var)
        x_range = np.linspace(-4 * max_std, 4 * max_std, 1000)
        
        # Plot Prior Predictive Density
        prior_std = np.sqrt(var_pred_prior + NOISE_STD**2)
        prior_pdf = (1 / (prior_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / prior_std)**2)
        ax3.plot(x_range, prior_pdf, color='#1f77b4', linewidth=LW_PRIOR, label='Prior Predictive')

        # Plot Exact Posterior Predictive Density (at T=1)
        exact_std_t1 = np.sqrt(var_pred_exact_t1 + NOISE_STD**2)
        exact_pdf_t1 = (1 / (exact_std_t1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / exact_std_t1)**2)
        ax3.plot(x_range, exact_pdf_t1, color='#2ca02c', linewidth=LW_EXACT, label='Exact Post ($T=1$)')
        
        # Plot MAP Predictive Density (only noise variance)
        map_std = NOISE_STD
        map_pdf = (1 / (map_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / map_std)**2)
        ax3.plot(x_range, map_pdf, color='#d62728', linewidth=LW_MAP, linestyle='-', label='MAP (Noise Only)', zorder=20)
        
        # Plot MFVI Densities for different temperatures
        cmap = plt.get_cmap('coolwarm')
        t_array = np.array(res['temperatures'])
        norm = plt.Normalize(vmin=np.log10(t_array.min()), vmax=np.log10(t_array.max()))
        
        # Plot all temperatures for a dense "fan" effect
        for t_idx, temp in enumerate(t_array):
            v_p = var_pred_mfvi[t_idx]
            std = np.sqrt(v_p + NOISE_STD**2)
            pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / std)**2)
            color = cmap(norm(np.log10(temp)))
            ax3.plot(x_range, pdf, color=color, alpha=0.3, linewidth=1.0, zorder=1)
            
        # Add back MFVI at T=1 as dashed black
        t1_idx = np.argmin(np.abs(t_array - 1.0))
        v_p_t1 = var_pred_mfvi[t1_idx]
        std_t1 = np.sqrt(v_p_t1 + NOISE_STD**2)
        pdf_t1 = (1 / (std_t1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x_range / std_t1)**2)
        ax3.plot(x_range, pdf_t1, color='black', linestyle='--', linewidth=1.5, label='MFVI ($T=1$)', alpha=0.9, zorder=2)
            
        ax3.set_xlabel('Predictive $y^*$ relative to likelihood', fontsize=LBL_FS)
        ax3.set_ylabel('Density', fontsize=LBL_FS)
        ax3.set_title(fr'Post Pred ($x^* = \frac{1}{{\sqrt{{P}}}} \mathbf{{1}}$)', fontsize=TTL_FS)
        
        # Set ticks at mu +/- n*sigma
        tick_vals = np.array([-6, -4, -2, 0, 2, 4, 6])
        tick_locs = tick_vals * NOISE_STD
        tick_labels = []
        for v in tick_vals:
            if v == 0: tick_labels.append(r'$\mu$')
            elif v > 0: tick_labels.append(rf'$\mu+{v}\sigma^2$')
            else: tick_labels.append(rf'$\mu{v}\sigma^2$')
            
        ax3.set_xticks(tick_locs)
        ax3.set_xticklabels(tick_labels)
        ax3.set_xlim([-7 * NOISE_STD, 7 * NOISE_STD])
        
        ax3.legend(fontsize=LEG_FS)
        ax3.grid(True, alpha=0.2)
        ax3.tick_params(labelsize=TICK_FS)
        
        # Add colorbar for temperature
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax3)
        cbar.set_label('Log10(Temperature)')
    
    plt.tight_layout()
    fig.savefig("prediction_analysis_icml.pdf", bbox_inches='tight')
    plt.show()
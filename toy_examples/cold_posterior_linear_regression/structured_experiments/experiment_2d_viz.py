import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import (compute_mfvi_analytic, compute_exact_posterior, 
                   generate_data, generate_test_data)

# Set default dtype to float64 for numerical stability
torch.set_default_dtype(torch.float64)

# ICML/LaTeX Formatting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    """Create a plot of the covariance confidence ellipse."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    # n_std scaling for 2D Gaussian:
    # 1-sigma: sqrt(chi2.ppf(0.393, 2)) = 1.0
    # 2-sigma: sqrt(chi2.ppf(0.865, 2)) = 2.0
    # 95% CI:  sqrt(chi2.ppf(0.950, 2)) = 2.447
    
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    
    t = np.linspace(0, 2*np.pi, 1000)
    ell_x = (width / 2) * np.cos(t)
    ell_y = (height / 2) * np.sin(t)
    
    # Rotate
    angle = np.radians(theta)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    ell_coords = np.dot(R, np.array([ell_x, ell_y]))
    
    # Translate
    ell_coords[0, :] += mean[0]
    ell_coords[1, :] += mean[1]
    
    plot_kwargs = {
        'color': kwargs.get('edgecolor', kwargs.get('color', 'black')),
        'linestyle': kwargs.get('linestyle', '-'),
        'linewidth': kwargs.get('linewidth', 1),
        'alpha': kwargs.get('alpha', 1.0),
        'label': kwargs.get('label', None)
    }
    return ax.plot(ell_coords[0, :], ell_coords[1, :], **plot_kwargs)

def run_2d_experiment():
    N_DIMS = 2
    N_SAMPLES = 10
    INPUT_STD = 1.0
    NOISE_STD = 0.5
    
    # Generate Data
    # Using diagonal_input=False for a standard 2D view
    X, y, true_weights = generate_data(n_samples=N_SAMPLES, n_dims=N_DIMS, 
                                       input_std=INPUT_STD, noise_std=NOISE_STD, 
                                       seed=42, diagonal_input=True)
    
    # Compute Exact Posterior
    mu_ex, Sigma_ex = compute_exact_posterior(X, y, noise_std=NOISE_STD)
    
    # Temperatures to explore
    temperatures = [0.1, 0.5, 1.0, 5.0, 10.0]
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=np.log10(min(temperatures)), vmax=np.log10(max(temperatures)))
    
    # Compute global limits for shared axes
    n_std_95 = np.sqrt(5.991)
    
    # Data limits
    data_min = X.min().item() - 1
    data_max = X.max().item() + 1
    
    # Weight limits (approximation based on true weights and exact posterior)
    # n_std_95 * scaling of Sigma_ex
    w_std = np.sqrt(Sigma_ex.diag().max().item()) * n_std_95
    weight_min = (true_weights.min().item() - w_std) - 1
    weight_max = (true_weights.max().item() + w_std) + 1
    
    global_min = min(data_min, weight_min)
    global_max = max(data_max, weight_max)
    
    # --- ICML Formatting Constants ---
    PLT_SIZE = 4.5
    LBL_FS = 18
    TTL_FS = 20
    TICK_FS = 14
    LEG_FS = 14

    # Plotting Figure 1: Training Data Space
    fig_data, ax_data = plt.subplots(figsize=(PLT_SIZE + 1.2, PLT_SIZE)) # Extra width for colorbar
    
    x_range_data = np.linspace(global_min, global_max, 250)
    y_range_data = np.linspace(global_min, global_max, 250)
    xx_data, yy_data = np.meshgrid(x_range_data, y_range_data)
    grid_coords_data = torch.tensor(np.stack([xx_data.ravel(), yy_data.ravel()], axis=1))
    
    var_ex_data = torch.sum((grid_coords_data @ Sigma_ex) * grid_coords_data, dim=1).reshape(xx_data.shape)
    
    mu_mfvi_t1, sigma_mfvi_t1 = compute_mfvi_analytic(X, y, temperature=1.0, noise_std=NOISE_STD)
    Sigma_mfvi_t1 = torch.diag(sigma_mfvi_t1)
    var_mfvi_data = torch.sum((grid_coords_data @ Sigma_mfvi_t1) * grid_coords_data, dim=1).reshape(xx_data.shape)
    
    var_ratio = (var_mfvi_data + NOISE_STD**2) / (var_ex_data + NOISE_STD**2)
    var_ratio_np = var_ratio.numpy()
    
    delta_cov = (Sigma_ex - Sigma_mfvi_t1).numpy()
    d11, d12, d22 = delta_cov[0, 0], delta_cov[0, 1], delta_cov[1, 1]
    discriminant = (2 * d12)**2 - 4 * d22 * d11
    slopes = []
    if discriminant >= 0:
        m1 = (-2 * d12 + np.sqrt(discriminant)) / (2 * d22)
        m2 = (-2 * d12 - np.sqrt(discriminant)) / (2 * d22)
        slopes = [m1, m2]
    
    from matplotlib.colors import TwoSlopeNorm
    norm_ratio = TwoSlopeNorm(vmin=var_ratio_np.min(), vcenter=1.0, vmax=var_ratio_np.max())
    
    im = ax_data.pcolormesh(xx_data, yy_data, var_ratio_np, shading='auto', cmap='RdBu_r', norm=norm_ratio, alpha=0.3)
    cbar = plt.colorbar(im, ax=ax_data)
    cbar.ax.tick_params(labelsize=TICK_FS)
    cbar.set_label('Variance Ratio', fontsize=LBL_FS)
    
    # Analytic lines label
    analytic_line_label = 'Var Ratio = 1.0'
    x_line = np.array([global_min, global_max])
    for i, m in enumerate(slopes):
        ax_data.plot(x_line, m * x_line, color='green', lw=2, linestyle='--', 
                     label=analytic_line_label if i == 0 else "")
    
    ax_data.scatter(X[:, 0].numpy(), X[:, 1].numpy(), color='blue', alpha=0.8, edgecolors='white', s=50, label='Training Data')
    ax_data.plot([global_min, global_max], [global_min, global_max], 'k--', alpha=0.5, label='$x_1 = x_2$')
    
    ax_data.set_xlabel('$x_1$', fontsize=LBL_FS)
    ax_data.set_ylabel('$x_2$', fontsize=LBL_FS)
    ax_data.tick_params(labelsize=TICK_FS)
    ax_data.legend(fontsize=LEG_FS, loc='upper left')
    ax_data.grid(True, alpha=0.2)
    ax_data.set_xlim(global_min, global_max)
    ax_data.set_ylim(global_min, global_max)
    ax_data.set_aspect('equal')
    
    fig_data.tight_layout()
    fig_data.savefig("2d_data_viz.pdf", bbox_inches='tight')

    # Plotting Figure 2: Parameter Space
    fig_param, ax_post = plt.subplots(figsize=(PLT_SIZE, PLT_SIZE))
    
    # Local limits for parameter space (focus on posterior uncertainty)
    w_std = np.sqrt(max(Sigma_ex[0,0], Sigma_ex[1,1], Sigma_mfvi_t1[0,0], Sigma_mfvi_t1[1,1]))
    w_min = min(mu_ex[0].item(), mu_ex[1].item()) - 3*w_std
    w_max = max(mu_ex[0].item(), mu_ex[1].item()) + 3*w_std

    confidence_ellipse(mu_ex.numpy(), Sigma_ex.numpy(), ax_post, n_std=n_std_95, edgecolor='red', linewidth=3, label='Exact')
    confidence_ellipse(mu_mfvi_t1.numpy(), Sigma_mfvi_t1.numpy(), ax_post, n_std=n_std_95, edgecolor='green', linestyle='--', linewidth=2, label='MFVI')
    
    ax_post.plot([w_min, w_max], [w_min, w_max], 'k--', alpha=0.5, label='$\\theta_1 = \\theta_2$')
    ax_post.set_xlabel('$\\theta_1$', fontsize=LBL_FS)
    ax_post.set_ylabel('$\\theta_2$', fontsize=LBL_FS)
    ax_post.tick_params(labelsize=TICK_FS)
    ax_post.legend(fontsize=LEG_FS, loc='upper right')
    ax_post.grid(True, alpha=0.3)
    ax_post.set_xlim(w_min, w_max)
    ax_post.set_ylim(w_min, w_max)
    ax_post.set_aspect('equal')
    
    fig_param.tight_layout()
    fig_param.savefig("2d_parameter_viz.pdf", bbox_inches='tight')

    # Plotting Figure 3: Precision Space Analysis (Curvature)
    fig3, ax_prec = plt.subplots(figsize=(PLT_SIZE + 1, PLT_SIZE))
    
    def plot_prec_ellipse(cov, ax, **kwargs):
        P = np.linalg.inv(cov)
        vals_P, vecs_P = np.linalg.eigh(P)
        order = vals_P.argsort()[::-1]
        vals_P = vals_P[order]
        vecs_P = vecs_P[:, order]
        theta = np.degrees(np.arctan2(*vecs_P[:, 0][::-1]))
        width, height = 2 * np.sqrt(vals_P)
        t = np.linspace(0, 2*np.pi, 200)
        ell_x = (width / 2) * np.cos(t)
        ell_y = (height / 2) * np.sin(t)
        angle = np.radians(theta)
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        ell_coords = np.dot(R, np.array([ell_x, ell_y]))
        ax.plot(ell_coords[0, :], ell_coords[1, :], **kwargs)
        return max(width, height)

    max_ext = 0
    ext_ex = plot_prec_ellipse(Sigma_ex.numpy(), ax_prec, color='red', linewidth=3, label='Exact Precision', zorder=5)
    max_ext = max(max_ext, ext_ex)
    
    for temp in temperatures:
        _, sigma_vi = compute_mfvi_analytic(X, y, temperature=temp, noise_std=NOISE_STD)
        Sigma_vi = torch.diag(sigma_vi**2)
        color = cmap(norm(np.log10(temp)))
        ls = '--' if temp == 1.0 else '-'
        lw = 2 if temp == 1.0 else 1.2
        label = f'MFVI Precision T={temp}' if temp in [0.1, 1.0, 10.0] else ""
        ext_vi = plot_prec_ellipse(Sigma_vi.numpy(), ax_prec, color=color, linestyle=ls, linewidth=lw, label=label)
        if temp >= 1.0:
            max_ext = max(max_ext, ext_vi)
    
    prec_lim = max_ext / 2 * 1.2 
    x_prec_line = np.array([-prec_lim * 2, prec_lim * 2])

    ax_prec.axhline(y=0, color='purple', linestyle=':', alpha=0.6)
    ax_prec.axvline(x=0, color='purple', linestyle=':', alpha=0.6)
    
    for m in slopes:
        ax_prec.plot(x_prec_line, m * x_prec_line, color='green', lw=2, linestyle='--')
    
    ax_prec.set_xlabel('$\\delta\\theta_1$', fontsize=LBL_FS)
    ax_prec.set_ylabel('$\\delta\\theta_2$', fontsize=LBL_FS)
    ax_prec.tick_params(labelsize=TICK_FS)
    ax_prec.grid(True, alpha=0.3)
    ax_prec.set_xlim(-prec_lim, prec_lim)
    ax_prec.set_ylim(-prec_lim, prec_lim)
    ax_prec.set_aspect('equal')
    
    fig3.tight_layout()
    fig3.savefig("2d_precision_viz.pdf", bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    run_2d_experiment()

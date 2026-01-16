import torch 
import numpy 

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
LW_LINE = 2.5

from src.linear_utils import polynomial_basis, rbf_basis, compute_exact_posterior, compute_mfvi_analytic

N=32
N_TEST=100

NOISE_SIGMA=0.1

N_CENTERS=[16,1024]
RBF_SIGMA=0.3
torch.manual_seed(42) 

# x = torch.linspace(-3,3, N).reshape(-1,1)
x = torch.randn(N).reshape(-1,1)*2
y = torch.sinc(x.squeeze(-1)) + torch.randn(N)*NOISE_SIGMA
x = x/3
x_test = torch.linspace(-4.5,4.5,N_TEST).reshape(-1,1)
f_test = torch.sinc(x_test.squeeze(-1)) 
x_test = x_test/3

TEMPERATURES = [0.001, 0.01, 0.1, 1.0]
res = {}

for n_centers in N_CENTERS:
    rbf_centers = torch.linspace(-2.5,2.5,n_centers).reshape(-1,1)
    rbf_length_scale = torch.tensor(RBF_SIGMA)

    phi = rbf_basis(x, centers=rbf_centers, lengthscale=rbf_length_scale)
    print(phi.shape)

    mu, Sigma = compute_exact_posterior(phi, y, NOISE_SIGMA)
    phi_test = rbf_basis(x_test, centers=rbf_centers,lengthscale=rbf_length_scale)

    mu_preds = mu@phi_test.T
    Sigma_preds = phi_test@Sigma@phi_test.T
    sigma_diag = torch.sqrt(torch.diag(Sigma_preds))

    res[n_centers] = {
        "mu_preds": mu_preds,
        "sigma_diag": sigma_diag,
        "mfvi": {}
    }

    # Store eigenvalues for the Exact Posterior
    evals_exact = torch.linalg.eigvalsh(Sigma)
    res[n_centers]["evals_exact"] = evals_exact.sort(descending=False)[0]

    for temp in TEMPERATURES:
        m, S = compute_mfvi_analytic(phi, y, temp, NOISE_SIGMA)
        m_preds = m@phi_test.T
        S_preds = phi_test@torch.diag(S)@phi_test.T
        s_diag = torch.sqrt(torch.diag(S_preds))
        
        res[n_centers]["mfvi"][temp] = {
            "m_preds": m_preds,
            "s_diag": s_diag,
            "S": S
        }


with torch.no_grad():
    fig, axs = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
    axs = axs.flatten()
    
    # Colors for different models
    exact_color_main = "black" # Use same black for all predictive plots
    p_colors_spectrum = {16: "#1f77b4", 1024: "#d62728"} # Blue and Red for spectrum comparison
    p_ls_spectrum = {16: "-", 1024: "--"} # 
    mfvi_cmap = plt.get_cmap("YlOrRd") # Yellow to Red for temperature

    for i, (n_centers, data) in enumerate(res.items()): 
        if i >= 2: continue # Only first two are predictive plots
        ax = axs[i]
        
        # Exact Posterior
        ax.plot(x_test, data["mu_preds"], label="Exact Post.", color=exact_color_main, linewidth=2, zorder=5)
        ax.fill_between(x_test.squeeze(-1), 
                            data["mu_preds"]+2*(data["sigma_diag"]), 
                            data["mu_preds"]-2*(data["sigma_diag"]), 
                            alpha=0.1, color=exact_color_main, zorder=4)
        
        # MFVI Posterior for different temperatures
        for j, temp in enumerate(TEMPERATURES):
            mfvi_data = data["mfvi"][temp]
            color_val = (torch.tensor(temp).log10() - torch.tensor(TEMPERATURES[0]).log10()) / \
                        (torch.tensor(TEMPERATURES[-1]).log10() - torch.tensor(TEMPERATURES[0]).log10())
            color = mfvi_cmap(color_val.item() * 0.7 + 0.3)
            label = f"MFVI $T=10^{{{int(torch.tensor(temp).log10().item())}}}$ CI" if temp < 1 else "MFVI $T=1$ CI"
            
            ax.plot(x_test, mfvi_data["m_preds"] + 2*mfvi_data["s_diag"], color=color, ls="--", linewidth=1.2, alpha=0.8)
            ax.plot(x_test, mfvi_data["m_preds"] - 2*mfvi_data["s_diag"], color=color, ls="--", linewidth=1.2, alpha=0.8, label=label)

        # Training Data
        ax.scatter(x, y, s=30, c="black", marker='x', label="Data", zorder=10)
        ax.set_xlabel("$x$", fontsize=LBL_FS)
        ax.set_ylabel("$y$", fontsize=LBL_FS)
        ax.set_ylim(-2,2)
        ax.tick_params(labelsize=TICK_FS)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"$P={n_centers}$", fontsize=TTL_FS)
    
    # Exact Posterior Spectrum plot (Fractional Rank)
    ax = axs[2]
    for n_centers, data in res.items():
        evals = 1/data["evals_exact"]
        # Normalize x-axis to [0, 1]
        x_frac = torch.linspace(0, 1, len(evals))
        # Plot eigenvalues (reverted from 1/evals)
        ax.plot(x_frac, evals, color=p_colors_spectrum[n_centers], ls=p_ls_spectrum[n_centers], label=f"Exact Post. ($P={n_centers}$)", linewidth=2)
    
    ax.set_yscale('log')
    # Use linear x-scale to show fraction clearly
    ax.set_xlabel("$p/P$", fontsize=LBL_FS)
    ax.set_ylabel(r"$1/\delta_p$", fontsize=LBL_FS)
    ax.set_title("Reciprocal Eigenvalues", fontsize=TTL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.3, which='both')

    # Gather all unique handles and labels for the legend
    all_handles, all_labels = [], []
    for axis in axs:
        h, l = axis.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

    # Sort labels or group manually for clarity if needed, but current set is manageable
    fig.legend(all_handles, all_labels, loc='lower center', ncol=len(all_labels)//2 + 1, 
               fontsize=LEG_FS-1, bbox_to_anchor=(0.5, 0.02), frameon=False)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig("fixed_basis_function_icml_v3.pdf", bbox_inches='tight')
    plt.show()

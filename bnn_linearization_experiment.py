import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from src.approx_posterior_bnn import MFVIPosterior, SBVIPosterior
from src.layer_priors import LinearLayer
from src.linear_utils import compute_exact_posterior, compute_mfvi_analytic
from src.utils import set_seeds

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

def get_bnn_model(in_features, out_features, width, depth, posterior_type, n_data, device, prior_std=0.33):
    layer_sizes = [in_features] + [width] * depth + [out_features]
    priors = []
    
    for i in range(len(layer_sizes) - 1):
        in_f = layer_sizes[i]
        out_f = layer_sizes[i+1]
        weight_prior = torch.distributions.Normal(
            torch.zeros(out_f, in_f).to(device), 
            prior_std * torch.ones(out_f, in_f).to(device)
        )
        bias_prior = torch.distributions.Normal(
            torch.zeros(out_f).to(device), 
            prior_std * torch.ones(out_f).to(device)
        )
        priors.append(LinearLayer(in_f, out_f, weight_prior, bias_prior))
        
    likelihood = torch.distributions.Normal(0, 0.01) # Consistent with NOISE_SIGMA
    
    if posterior_type == "mfvi":
        model = MFVIPosterior(
            layer_priors=priors, 
            likelihood=likelihood, 
            device=device, 
            num_samples=1, 
            temperature=1, 
            posterior_exponentiation="tempered", 
            n_data=n_data
        )
    elif posterior_type == "sbvi":
        model = SBVIPosterior(
            layer_priors=priors, 
            likelihood=likelihood, 
            device=device, 
            num_samples=1, 
            n_squash_vectors=64, 
            temperature=1, 
            posterior_exponentiation="tempered", 
            n_data=n_data
        )
    else:
        raise ValueError(f"Unknown posterior type: {posterior_type}")
        
    return model

def functional_forward(params, inputs, layer_priors):
    """Computes the forward pass given a list of weights and biases."""
    curr_x = inputs
    for i, layer in enumerate(layer_priors):
        w = params[f"w_{i}"]
        b = params[f"b_{i}"]
        
        # Linear transform
        lin_out = curr_x @ w.T + b.unsqueeze(0)
        
        if i < len(layer_priors) - 1:
            curr_x = torch.nn.functional.relu(lin_out)
        else:
            curr_x = lin_out
    return curr_x

def get_basis_functions(model, x):
    """Computes the Jacobian of the model output w.r.t its weights at the posterior mean."""
    means = {}
    for i, layer in enumerate(model.layers):
        means[f"w_{i}"] = layer.mu_w.detach().clone().requires_grad_(True)
        if layer.mu_b is not None:
            means[f"b_{i}"] = layer.mu_b.detach().clone().requires_grad_(True)

    # Reconstruct the layers as simple objects for functional forward
    layer_priors = [l.layer for l in model.layers]

    # We want grad of output w.r.t params for each input x_i
    # Output of f is (N, 1)
    # Jacobian is (N, P)
    
    all_grads = []
    for i in range(x.shape[0]):
        x_i = x[i:i+1]
        
        # Forward pass inside the loop to build a new graph each time
        out = functional_forward(means, x_i, layer_priors)
        val = out.sum()
        
        # Compute gradient w.r.t each parameter
        # We need to preserve the order to flatten consistently
        grads = torch.autograd.grad(val, means.values())
        flat_grad = torch.cat([g.flatten() for g in grads])
        all_grads.append(flat_grad)
        
    return torch.stack(all_grads)

def train_model(model, x, y, epochs=5000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for epoch in range(epochs):
        loss = model.train_step(x, y, optimizer)
        losses.append(loss.item() if torch.is_tensor(loss) else loss)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss {losses[-1]}")
    model.eval()
    return losses

def main():
    parser = argparse.ArgumentParser(description="BNN Linearization Experiment")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--posterior", type=str, default="mfvi", choices=["mfvi", "sbvi"])
    parser.add_argument("--epochs", type=int, default=5000)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    set_seeds(42)
    device = "cpu" # Default to cpu for ease of use in this context

    # 1. Data Generation (same as fixed_basis_function.py)
    N = 20
    N_TEST = 1000
    NOISE_SIGMA = 0.01
    
    x_train = torch.linspace(-3, 3, N).reshape(-1, 1)
    y_train = torch.sinc(x_train.squeeze(-1)) + torch.rand(N) * NOISE_SIGMA
    y_train = y_train.reshape(-1, 1)
    x_train = x_train / 3.0
    
    x_test = torch.linspace(-4, 4, N_TEST).reshape(-1, 1)
    y_test_clean = torch.sinc(x_test.squeeze(-1)).reshape(-1, 1)
    x_test = x_test / 3.0
    
    # 2. BNN Setup and Training
    model = get_bnn_model(1, 1, args.width, args.depth, args.posterior, N, device)
    print(f"Training BNN with {args.posterior} posterior...")
    losses = train_model(model, x_train, y_train, epochs=args.epochs)

    # Plot Loss Curve
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(8, 5))
    ax_loss.plot(losses, color="red", linewidth=1)
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel("Epoch", fontsize=LBL_FS)
    ax_loss.set_ylabel("Loss", fontsize=LBL_FS)
    ax_loss.set_title(f"BNN Training Loss ({args.posterior.upper()})", fontsize=TTL_FS)
    ax_loss.grid(True, alpha=0.3)
    fig_loss.savefig("bnn_loss_curve.pdf", bbox_inches='tight')
    plt.close(fig_loss)
    print("Saved loss curve to bnn_loss_curve.pdf")
    
    # 3. Linearization
    print("Computing linearized basis functions...")
    # Center of linearization (mean)
    params_mean = {}
    for i, layer in enumerate(model.layers):
        params_mean[f"w_{i}"] = layer.mu_w.detach()
        if layer.mu_b is not None:
            params_mean[f"b_{i}"] = layer.mu_b.detach()
    
    f_0_train = functional_forward(params_mean, x_train, [l.layer for l in model.layers]).detach()
    f_0_test = functional_forward(params_mean, x_test, [l.layer for l in model.layers]).detach()
    
    phi_train = get_basis_functions(model, x_train)
    phi_test = get_basis_functions(model, x_test)
    
    # Target for linear regression on weights delta
    y_centered = y_train - f_0_train
    
    # 4. Linear Model Comparison (Exact vs MFVI)
    print("Computing linearized Exact and MFVI posteriors...")
    # Prior std from model
    prior_std = 0.33
    # compute_exact_posterior expects prior to be standard I? 
    # Actually, compute_exact_posterior in linear_utils.py assumes prior precision is I.
    # So we should scale our basis functions by prior_std, or adjust the precision.
    # Let's adjust phi to account for prior std: phi_scaled = phi * prior_std
    # Then Delta w_scaled ~ N(0, I)
    phi_train_scaled = phi_train * prior_std
    phi_test_scaled = phi_test * prior_std
    
    # compute_exact_posterior(X, y, noise_std)
    mu_lin_exact, Sigma_lin_exact = compute_exact_posterior(phi_train_scaled, y_centered, NOISE_SIGMA)
    m_lin_mfvi, S_lin_mfvi = compute_mfvi_analytic(phi_train_scaled, y_centered, 1.0, NOISE_SIGMA)
    
    # Predictions
    # f_pred = f_0 + phi_scaled @ delta_w
    preds_exact = f_0_test + phi_test_scaled @ mu_lin_exact
    # Variance: phi_scaled @ Sigma @ phi_scaled.T + noise_sq
    var_exact_params = torch.diag(phi_test_scaled @ Sigma_lin_exact @ phi_test_scaled.T)
    std_exact = torch.sqrt(var_exact_params + NOISE_SIGMA**2)
    
    preds_mfvi = f_0_test + phi_test_scaled @ m_lin_mfvi
    var_mfvi_params = torch.diag(phi_test_scaled @ torch.diag(S_lin_mfvi) @ phi_test_scaled.T)
    std_mfvi = torch.sqrt(var_mfvi_params + NOISE_SIGMA**2)
    
    # 4b. NN with Optimized Linear MFVI Parameters
    print("Computing BNN prediction with optimized linearized MFVI parameters...")
    # Map m_lin_mfvi and S_lin_mfvi back to model parameters
    opt_model = get_bnn_model(1, 1, args.width, args.depth, args.posterior, N, device)
    # Copy trained mean parameters as baseline
    opt_model.load_state_dict(model.state_dict())
    
    current_idx = 0
    delta_w_full = m_lin_mfvi * prior_std
    sigma_w_full = S_lin_mfvi * prior_std
    
    for i, layer in enumerate(opt_model.layers):
        # Update weights
        num_w = layer.mu_w.numel()
        w_delta = delta_w_full[current_idx : current_idx + num_w].reshape(layer.mu_w.shape)
        w_sigma = sigma_w_full[current_idx : current_idx + num_w].reshape(layer.mu_w.shape)
        
        layer.mu_w.data = params_mean[f"w_{i}"] + w_delta
        layer._raw_sigma_w.data = torch.log(w_sigma**2 + 1e-12)
        current_idx += num_w
        
        # Update biases if they exist
        if layer.mu_b is not None:
            num_b = layer.mu_b.numel()
            b_delta = delta_w_full[current_idx : current_idx + num_b].reshape(layer.mu_b.shape)
            b_sigma = sigma_w_full[current_idx : current_idx + num_b].reshape(layer.mu_b.shape)
            
            layer.mu_b.data = params_mean[f"b_{i}"] + b_delta
            layer._raw_sigma_b.data = torch.log(b_sigma**2 + 1e-12)
            current_idx += num_b
            
    bnn_opt_preds_mean = opt_model.predict(x_test, n_samples=100).detach()
    bnn_opt_lower, bnn_opt_upper = opt_model.get_CI(x_test, ci=0.95, n_samples=100)

    # 4c. Linear prediction using BNN learned parameters
    print("Computing linearized prediction using BNN learned parameters...")
    # Extract learned diagonal variances
    with torch.no_grad():
        bnn_sigmas = []
        for layer in model.layers:
            bnn_sigmas.append(layer.Sigma_w.flatten())
            if layer.mu_b is not None:
                bnn_sigmas.append(layer.Sigma_b.flatten())
        flat_bnn_sigmas = torch.cat(bnn_sigmas)
    
    # Linear mean is just f_0_test (at the trained mean)
    # Linear variance is phi S_learned phi.T + noise
    var_bnn_learned_linear = torch.diag(phi_test @ torch.diag(flat_bnn_sigmas) @ phi_test.T)
    std_bnn_learned_linear = torch.sqrt(var_bnn_learned_linear + NOISE_SIGMA**2)

    # BNN Predictions
    print("Getting original BNN predictions...")
    bnn_preds_mean = model.predict(x_test, n_samples=100).detach()
    bnn_lower, bnn_upper = model.get_CI(x_test, ci=0.95, n_samples=100)
    
    # 5. Plotting
    print("Plotting results...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Plot Linearized Exact
    ax.plot(x_test, preds_exact.squeeze(), color="#1f77b4", label="Linearized Exact (opt)", linewidth=LW_LINE)
    ax.fill_between(x_test.squeeze(), 
                        (preds_exact.squeeze() - 2*std_exact).squeeze(), 
                        (preds_exact.squeeze() + 2*std_exact).squeeze(), 
                        alpha=0.1, color="#1f77b4")
    
    # # Plot Linearized BNN (learned)
    # ax.plot(x_test, f_0_test.squeeze(), color="green", label="Linearized BNN (learned)", ls=":", linewidth=LW_LINE)
    # ax.fill_between(x_test.squeeze(), 
    #                     (f_0_test.squeeze() - 2*std_bnn_learned_linear).squeeze(), 
    #                     (f_0_test.squeeze() + 2*std_bnn_learned_linear).squeeze(), 
    #                     alpha=0.15, color="green")

    # Plot Linearized MFVI (optimal)
    ax.plot(x_test, preds_mfvi.squeeze(), color="#ff7f0e", label="Linearized MFVI (opt)", ls="--", linewidth=LW_LINE)
    ax.fill_between(x_test.squeeze(), 
                        (preds_mfvi.squeeze() - 2*std_mfvi).squeeze(), 
                        (preds_mfvi.squeeze() + 2*std_mfvi).squeeze(), 
                        alpha=0.1, color="#ff7f0e")

    # Plot BNN with optimized parameters
    # ax.plot(x_test, bnn_opt_preds_mean, color="purple", label="Full BNN (opt linear params)", linewidth=LW_LINE, alpha=0.9)
    # ax.fill_between(x_test.squeeze(), bnn_opt_lower.squeeze(), bnn_opt_upper.squeeze(), alpha=0.1, color="purple")

    # Plot Original BNN
    ax.plot(x_test, bnn_preds_mean, color="black", label="Full BNN (learned)", linewidth=LW_LINE, alpha=0.7)
    ax.fill_between(x_test.squeeze(), bnn_lower.squeeze(), bnn_upper.squeeze(), alpha=0.1, color="black")
    
    # Data points
    ax.scatter(x_train, y_train, s=40, c="black", marker='x', label="Training Data", zorder=10)
    
    ax.set_xlabel("$x$", fontsize=LBL_FS)
    ax.set_ylabel("$y$", fontsize=LBL_FS)
    ax.set_title(f"BNN Linearization Comparison ({args.posterior.upper()})", fontsize=TTL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=LEG_FS-3, loc="upper left", ncol=2)
        
    plt.tight_layout()
    fig.savefig("bnn_linearization_comparison_combined.pdf", bbox_inches='tight')
    plt.show()
    print("Saved combined plot to bnn_linearization_comparison_combined.pdf")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import ivon

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

def functional_forward(params, inputs):
    """Computes the forward pass given a list of weights and biases."""
    curr_x = inputs
    # parameters are in order: w0, b0, w1, b1, ...
    param_list = list(params.values())
    for i in range(0, len(param_list), 2):
        w = param_list[i]
        b = param_list[i+1]
        
        lin_out = curr_x @ w.T + b.unsqueeze(0)
        
        if i < len(param_list) - 2:
            curr_x = torch.nn.functional.relu(lin_out)
        else:
            curr_x = lin_out
    return curr_x

def get_average_basis_functions(model, optimizer, x, n_samples=32):
    """Computes the average Jacobian of the model output w.r.t its weights over q(theta)."""
    total_phi = None
    print(f"Sampling {n_samples} times for average basis functions...")
    
    for s in range(n_samples):
        with optimizer.sampled_params():
            # Extract current sampled parameters from the model
            # functional_forward expects them in order p_0, p_1...
            params_sample = {f"p_{i}": p.detach().clone().requires_grad_(True) for i, p in enumerate(model.parameters())}
            
            sample_phi = []
            for i in range(x.shape[0]):
                x_i = x[i:i+1]
                out = functional_forward(params_sample, x_i)
                val = out.sum()
                grads = torch.autograd.grad(val, params_sample.values())
                flat_grad = torch.cat([g.flatten() for g in grads])
                sample_phi.append(flat_grad.detach())
            
            sample_phi = torch.stack(sample_phi)
            if total_phi is None:
                total_phi = sample_phi
            else:
                total_phi += sample_phi
                
    return total_phi / n_samples

def train_ivon(model, optimizer, x, y, noise_sigma, epochs=5000):
    model.train()
    # Average Negative Log-Likelihood for Gaussian
    # NLL = 1/(2*sigma^2) * MSE + 0.5 * log(2*pi*sigma^2)
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        with optimizer.sampled_params(train=True):
            preds = model(x)
            mse = torch.nn.functional.mse_loss(preds, y)
            loss = mse / (2 * noise_sigma**2)
            loss.backward()
        optimizer.step()
        losses.append(mse.item()) # Still track MSE for the loss curve
        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, Loss (MSE) {mse.item()}")
    model.eval()
    return losses

def main():
    parser = argparse.ArgumentParser(description="BNN Linearization Experiment with IVON")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--hess_init", type=float, default=1.0)
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--prior_std", type=float, default=0.33)
    parser.add_argument("--train_samples", type=int, default=1)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--basis_samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    set_seeds(42)
    device = "cpu"

    # 1. Data Generation
    N = 64
    N_TEST = 256
    NOISE_SIGMA = 0.01
    
    x_train = torch.linspace(-3, 3, N).reshape(-1, 1)
    # Use the specified noise_sigma for consistency
        # Matching fixed_basis_function.py logic
    # x_train = torch.randn(N, 1) * 2
    y_train = torch.sin(x_train.squeeze(-1)).reshape(-1, 1) + torch.randn(N, 1) * NOISE_SIGMA
    # x_train = x_train / 3.0
    
    x_test = torch.linspace(-6, 6, N_TEST).reshape(-1, 1)
    y_test_clean = torch.sin(x_test.squeeze(-1)).reshape(-1, 1)
    # x_test = x_test / 3.0
    
    # 2. Model and IVON Setup
    layers = []
    curr_dim = 1
    for _ in range(args.depth):
        layers.append(nn.Linear(curr_dim, args.width))
        layers.append(nn.Tanh())
        curr_dim = args.width
    layers.append(nn.Linear(curr_dim, 1))
    model = nn.Sequential(*layers).to(device)
    
    # weight_decay in IVON acts as prior precision (1/prior_var)
    wd = 1.0 / (args.prior_std**2)
    # ess should be the number of training examples
    optimizer = ivon.IVON(model.parameters(), lr=args.lr, ess=N/args.temperature, weight_decay=wd, hess_init=args.hess_init, mc_samples=args.train_samples)
    
    print(f"Training with IVON for {args.epochs} epochs...")
    losses = train_ivon(model, optimizer, x_train, y_train, noise_sigma=NOISE_SIGMA, epochs=args.epochs)
    
    # Plot Loss Curve
    fig_loss, ax_loss = plt.subplots(1, 1, figsize=(8, 5))
    ax_loss.plot(losses, color="red", linewidth=1)
    ax_loss.set_yscale('log')
    ax_loss.set_xlabel("Epoch", fontsize=LBL_FS)
    ax_loss.set_ylabel("MSE Loss", fontsize=LBL_FS)
    ax_loss.set_title("Training Loss Curve", fontsize=TTL_FS)
    ax_loss.grid(True, alpha=0.3)
    fig_loss.savefig("ivon_loss_curve.pdf", bbox_inches='tight')
    plt.close(fig_loss)
    print("Saved loss curve to ivon_loss_curve.pdf")

    # 3. Linearization
    print("Computing average linearized basis functions...")
    params_mean = {f"p_{i}": p.detach() for i, p in enumerate(model.parameters())}
    f_0_train = functional_forward(params_mean, x_train).detach()
    f_0_test = functional_forward(params_mean, x_test).detach()
    
    phi_train = get_average_basis_functions(model, optimizer, x_train, n_samples=args.basis_samples)
    phi_test = get_average_basis_functions(model, optimizer, x_test, n_samples=args.basis_samples)
    
    y_centered = y_train - f_0_train
    
    # 4. Linear Model Comparison
    print("Computing Linearized Exact and MFVI...")
    # Map phi to standard scale for analytical solvers
    # IVON posterior is N(w, 1/h)
    # We can use the Hessian as precision.
    # For LLA: Sigma = H^-1
    
    # Extract IVON Hessian
    with torch.no_grad():
        h_flat = torch.cat([optimizer.state[p]["hess"] if p in optimizer.state and "hess" in optimizer.state[p] else torch.zeros_like(p).flatten() for p in model.parameters()])
        # Wait, in the test script I saw group['hess']. 
        # In ivon source, hess is in group if initialized.
        # Actually it's better to get it from group.
        h_flat = torch.cat([pg["hess"] for pg in optimizer.param_groups])
        
    var_ivon_lla = torch.diag(phi_test @ torch.diag(1.0 / h_flat) @ phi_test.T)
    std_ivon_lla = torch.sqrt(var_ivon_lla + NOISE_SIGMA**2)

    # Compute Analytic Benchmarks (using a standard prior scale for comparison)
    prior_std = 0.33 # Prior scale used in BNN script
    phi_train_scaled = phi_train * prior_std
    phi_test_scaled = phi_test * prior_std
    
    mu_lin_exact, Sigma_lin_exact = compute_exact_posterior(phi_train_scaled, y_centered, NOISE_SIGMA)
    m_lin_mfvi, S_lin_mfvi = compute_mfvi_analytic(phi_train_scaled, y_centered, 1.0, NOISE_SIGMA)
    
    preds_exact = f_0_test + phi_test_scaled @ mu_lin_exact
    std_exact = torch.sqrt(torch.diag(phi_test_scaled @ Sigma_lin_exact @ phi_test_scaled.T))
    
    preds_mfvi = f_0_test + phi_test_scaled @ m_lin_mfvi
    std_mfvi = torch.sqrt(torch.diag(phi_test_scaled @ torch.diag(S_lin_mfvi) @ phi_test_scaled.T))

    # 4b. NN with Optimized Linear MFVI Parameters
    print("Computing BNN (Full NN) with optimized linear MFVI params...")
    delta_w_full = (m_lin_mfvi * prior_std)
    
    # We need a new model instance to avoid messing with tuned means
    opt_model = nn.Sequential(*[nn.Linear(1, args.width), nn.ReLU(), nn.Linear(args.width, 1)] if args.depth==1 else layers).to(device)
    # Actually just clone the model state
    import copy
    opt_model = copy.deepcopy(model)
    
    curr_idx = 0
    with torch.no_grad():
        for p in opt_model.parameters():
            num = p.numel()
            p.data += delta_w_full[curr_idx : curr_idx+num].reshape(p.shape)
            curr_idx += num
    
    # To get CI, we need to sample if we want "Variational" behavior. 
    # But MFVI analytical only gives a Gaussian. 
    # We'll just plot the point prediction for now, or sample white noise on weights.
    bnn_opt_preds = opt_model(x_test).detach()

    # 5. Full IVON Predictions (Sampling)
    print("Sampling from IVON posterior...")
    ivon_samples = []
    for _ in range(args.test_samples):
        with optimizer.sampled_params():
            ivon_samples.append(model(x_test).detach())
    ivon_samples = torch.stack(ivon_samples)
    ivon_mean = ivon_samples.mean(0)
    ivon_std = ivon_samples.std(0)
    ivon_lower = ivon_mean - 2 * ivon_std
    ivon_upper = ivon_mean + 2 * ivon_std

    # 6. Plotting
    print("Plotting results...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Plot Linearized Exact
    # ax.plot(x_test.squeeze(), preds_exact.squeeze(), color="#1f77b4", label="Linearized Exact (opt)", linewidth=LW_LINE)
    # ax.fill_between(x_test.squeeze(), (preds_exact.squeeze() - 2*std_exact.squeeze()), (preds_exact.squeeze() + 2*std_exact.squeeze()), alpha=0.1, color="#1f77b4")
    
    # Plot LLA of IVON
    # ax.plot(x_test.squeeze(), f_0_test.squeeze(), color="green", label="LLA of IVON", ls=":", linewidth=LW_LINE)
    # ax.fill_between(x_test.squeeze(), (f_0_test.squeeze() - 2*std_ivon_lla.squeeze()), (f_0_test.squeeze() + 2*std_ivon_lla.squeeze()), alpha=0.15, color="green")

    # Plot Linearized MFVI (optimal)
    # ax.plot(x_test.squeeze(), preds_mfvi.squeeze(), color="#ff7f0e", label="Linearized MFVI (opt)", ls="--", linewidth=LW_LINE)
    # ax.fill_between(x_test.squeeze(), (preds_mfvi.squeeze() - 2*std_mfvi.squeeze()), (preds_mfvi.squeeze() + 2*std_mfvi.squeeze()), alpha=0.1, color="#ff7f0e")

    # Plot Full IVON (learned)
    ax.plot(x_test.squeeze(), ivon_mean.squeeze(), color="black", label="Full BNN (IVON)", linewidth=LW_LINE, alpha=0.7)
    ax.fill_between(x_test.squeeze(), ivon_lower.squeeze(), ivon_upper.squeeze(), alpha=0.1, color="black")

    # Data points
    ax.scatter(x_train, y_train, s=40, c="black", marker='x', label="Training Data", zorder=10)
    
    ax.set_xlabel("$x$", fontsize=LBL_FS)
    ax.set_ylabel("$y$", fontsize=LBL_FS)
    ax.set_title(f"BNN Linearization Comparison (IVON)", fontsize=TTL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=LEG_FS-3, loc="upper left", ncol=2)
        
    plt.tight_layout()
    fig.savefig("bnn_linearization_ivon.pdf", bbox_inches='tight')
    plt.show()
    print("Saved plot to bnn_linearization_ivon.pdf")

if __name__ == "__main__":
    main()

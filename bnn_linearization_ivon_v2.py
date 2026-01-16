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

# ICML Formatting Constants (Scaled for single column)
LBL_FS = 22
TTL_FS = 24
TICK_FS = 18
LEG_FS = 16
LW_LINE = 3.0

def train_ivon(model, optimizer, x, y, noise_sigma, epochs=5000):
    model.train()
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        with optimizer.sampled_params(train=True):
            preds = model(x)
            mse = torch.nn.functional.mse_loss(preds, y)
            loss = mse / (2 * noise_sigma**2)
            loss.backward()
        optimizer.step()
        losses.append(mse.item())
        if epoch % 5000 == 0:
            print(f"Epoch {epoch}, Loss (MSE) {mse.item()}")
    model.eval()
    return losses

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(depth, width, args, x_train, y_train, x_test, y_test_clean, device):
    set_seeds(42)
    # 2. Model and IVON Setup
    layers = []
    curr_dim = 1
    for _ in range(depth):
        layers.append(nn.Linear(curr_dim, width))
        layers.append(nn.Tanh())
        curr_dim = width
    layers.append(nn.Linear(curr_dim, 1))
    model = nn.Sequential(*layers).to(device)
    
    n_params = count_parameters(model)
    print(f"\n--- Running Experiment: Depth {depth}, Width {width} ({n_params} parameters) ---")
    
    # weight_decay in IVON acts as prior precision (1/prior_var)
    wd = 1.0 / (args.prior_std**2)
    # ess should be the number of training examples
    optimizer = ivon.IVON(model.parameters(), lr=args.lr, ess=x_train.shape[0]/args.temperature, weight_decay=wd, hess_init=args.hess_init, mc_samples=args.train_samples)
    
    print(f"Training with IVON for {args.epochs} epochs...")
    losses = train_ivon(model, optimizer, x_train, y_train, noise_sigma=args.noise_sigma, epochs=args.epochs)

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
    
    return {
        "n_params": n_params,
        "ivon_mean": ivon_mean, "ivon_lower": ivon_lower, "ivon_upper": ivon_upper,
        "losses": losses
    }

def main():
    parser = argparse.ArgumentParser(description="BNN Linearization Experiment with IVON")
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--hess_init", type=float, default=1.0)
    parser.add_argument("--noise_sigma", type=float, default=0.01)
    parser.add_argument("--prior_std", type=float, default=0.33)
    parser.add_argument("--train_samples", type=int, default=1)
    parser.add_argument("--test_samples", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    device = "cpu"

    # 1. Data Generation
    N = 64
    N_TEST = 256
    NOISE_SIGMA = args.noise_sigma
    
    set_seeds(42)
    x_train = torch.linspace(-3, 3, N).reshape(-1, 1)
    y_train = torch.sin(x_train.squeeze(-1)).reshape(-1, 1) + torch.randn(N, 1) * NOISE_SIGMA
    
    x_test = torch.linspace(-6, 6, N_TEST).reshape(-1, 1)
    y_test_clean = torch.sin(x_test.squeeze(-1)).reshape(-1, 1)

    architectures = [(1, 16), (4, 32), (8, 64)]
    results = []
    
    for depth, width in architectures:
        res = run_experiment(depth, width, args, x_train, y_train, x_test, y_test_clean, device)
        results.append(res)

    # 6. Plotting
    print("\nPlotting combined results...")
    # Reduced figure size for single-column ICML (approx 3.25in wide, we use slightly more for clarity)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Using Jet colormap for architectures
    colors = plt.cm.inferno(np.linspace(0.01, 0.9, len(results)))
    line_styles = [":", "--", "-"] # Dotted, Dashed, Solid
    
    for i, res in enumerate(results):
        depth, width = architectures[i]
        n_params = res["n_params"]
        color = colors[i]
        ls = line_styles[i % len(line_styles)]
        
        ax.plot(x_test.squeeze(), res["ivon_mean"].squeeze(), color=color, linestyle=ls,
                label=f"Depth={depth}, Width={width} ({n_params}p)", linewidth=LW_LINE)
        ax.fill_between(x_test.squeeze(), res["ivon_lower"].squeeze(), res["ivon_upper"].squeeze(), alpha=0.15, color=color)

    # Data points
    ax.scatter(x_train, y_train, s=60, c="black", marker='x', alpha=1, label="Data", zorder=10)
    
    ax.set_xlabel("$x$", fontsize=LBL_FS)
    ax.set_ylabel("$y$", fontsize=LBL_FS)
    ax.set_title("BNN predictions from IVON", fontsize=TTL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=LEG_FS, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
              ncol=2, frameon=True, framealpha=0.8)
        
    plt.tight_layout()
    fig.savefig("bnn_ivon_arch_comparison.pdf", bbox_inches='tight')
    plt.show()
    print("Saved combined plot to bnn_ivon_arch_comparison.pdf")

if __name__ == "__main__":
    main()

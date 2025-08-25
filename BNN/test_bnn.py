import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational parameters"""
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 2)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * 0.1 - 2)
    
    def forward(self, x, sample=True):
        if sample:
            # Sample weights and biases
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean values
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """Compute KL divergence between posterior and prior"""
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)
        
        # KL for weights
        kl_weight = 0.5 * torch.sum(
            self.weight_mu**2 / self.prior_std**2 + 
            weight_var / self.prior_std**2 - 
            1 - torch.log(weight_var / self.prior_std**2)
        )
        
        # KL for biases
        kl_bias = 0.5 * torch.sum(
            self.bias_mu**2 / self.prior_std**2 + 
            bias_var / self.prior_std**2 - 
            1 - torch.log(bias_var / self.prior_std**2)
        )
        
        return kl_weight + kl_bias

class BNN(nn.Module):
    """Bayesian Neural Network"""
    def __init__(self, input_dim=1, hidden_dim=50, output_dim=1, prior_std=1.0):
        super().__init__()
        self.layer1 = BayesianLinear(input_dim, hidden_dim, prior_std)
        self.layer2 = BayesianLinear(hidden_dim, hidden_dim, prior_std)
        self.layer3 = BayesianLinear(hidden_dim, output_dim, prior_std)
        
        # Noise parameter for likelihood
        self.log_noise = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x, sample=True):
        x = torch.tanh(self.layer1(x, sample))
        x = torch.tanh(self.layer2(x, sample))
        x = self.layer3(x, sample)
        return x
    
    def kl_divergence(self):
        return self.layer1.kl_divergence() + self.layer2.kl_divergence() + self.layer3.kl_divergence()

def generate_toy_data(n_samples=100, noise_std=0.3, seed=42):
    """Generate 1D regression toy data with sinusoid and constant variance"""
    np.random.seed(seed)
    x = np.linspace(-4, 4, n_samples)
    y = np.sin(x) + 0.3 * np.sin(3*x) + np.random.normal(0, noise_std, n_samples)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

def log_likelihood(model, x, y):
    """Compute log likelihood"""
    pred = model(x, sample=False)
    noise_std = torch.exp(0.5 * model.log_noise)
    return -0.5 * torch.sum((y - pred)**2) / noise_std**2 - len(y) * torch.log(noise_std)

def log_prior(model):
    """Compute log prior"""
    log_p = 0
    for param in model.parameters():
        log_p -= 0.5 * torch.sum(param**2)
    return log_p

class SGLDSampler:
    """Stochastic Gradient Langevin Dynamics sampler"""
    def __init__(self, model, lr=0.01, temperature=1.0):
        self.model = model
        self.lr = lr
        self.temperature = temperature
    
    def step(self, x, y):
        self.model.zero_grad()
        
        # Compute negative log posterior
        log_lik = log_likelihood(self.model, x, y)
        log_p = log_prior(self.model)
        loss = -(log_lik + log_p)
        
        loss.backward()
        
        # SGLD update
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.randn_like(param) * np.sqrt(2 * self.lr * self.temperature)
                param.add_(param.grad, alpha=-self.lr)
                param.add_(noise)

def compute_effective_sample_size(samples):
    """Compute effective sample size using autocorrelation"""
    if len(samples) < 50:
        return len(samples)
    
    # Use first parameter as proxy for convergence
    param_trace = torch.stack([s[0].flatten() for s in samples])
    mean_trace = torch.mean(param_trace, dim=1)
    
    # Compute autocorrelation
    n = len(mean_trace)
    autocorr = []
    
    for lag in range(min(n//4, 50)):
        if lag == 0:
            autocorr.append(1.0)
        else:
            corr = torch.corrcoef(torch.stack([
                mean_trace[:-lag],
                mean_trace[lag:]
            ]))[0, 1]
            if torch.isnan(corr):
                autocorr.append(0.0)
            else:
                autocorr.append(corr.item())
    
    # Find integrated autocorrelation time
    tau_int = 1.0
    for i, c in enumerate(autocorr[1:], 1):
        if c <= 0:
            break
        tau_int += 2 * c
    
    eff_samples = n / (2 * tau_int + 1)
    return max(1, int(eff_samples))

def train_mfvi(model, x, y, epochs=1000, lr=0.01):
    """Train with Mean Field Variational Inference"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass (sample from posterior)
        pred = model(x, sample=True)
        
        # Likelihood term
        noise_std = torch.exp(0.5 * model.log_noise)
        likelihood = -0.5 * torch.sum((y - pred)**2) / noise_std**2 - len(y) * model.log_noise
        
        # KL divergence term
        kl_div = model.kl_divergence()
        
        # ELBO loss
        loss = -likelihood + kl_div / len(y)  # Scale KL by dataset size
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 1000 == 0:
            print(f"MFVI Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return losses

def train_sgld_basic(model, x, y, epochs=2000, lr=0.01, burnin=500):
    """Train with basic SGLD - fixed burnin and thinning"""
    sampler = SGLDSampler(model, lr=lr)
    samples = []
    losses = []
    log_posteriors = []
    
    print(f"Basic SGLD: Running {burnin} burnin iterations, then collecting every 10th sample...")
    
    for epoch in range(epochs):
        sampler.step(x, y)
        
        # Monitor convergence
        with torch.no_grad():
            pred = model(x, sample=False)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())
            
            # Track log posterior for diagnostics
            log_lik = log_likelihood(model, x, y)
            log_p = log_prior(model)
            log_post = log_lik + log_p
            log_posteriors.append(log_post.item())
        
        # Collect samples after burnin with thinning
        if epoch >= burnin and epoch % 10 == 0:  # Thin by factor of 10
            samples.append([param.clone().detach() for param in model.parameters()])
        
        if epoch % 1000 == 0:
            if epoch < burnin:
                print(f"SGLD Burnin {epoch}/{burnin}, Loss: {loss.item():.4f}, Log-Post: {log_post.item():.2f}")
            else:
                n_samples = len(samples)
                print(f"SGLD Epoch {epoch}, Loss: {loss.item():.4f}, Samples: {n_samples}")
    
    print(f"Basic SGLD completed: {len(samples)} samples collected after burnin")
    
    # Simple convergence diagnostic - check if log posterior has stabilized
    if len(log_posteriors) > burnin:
        burnin_posts = log_posteriors[burnin:]
        recent_var = np.var(burnin_posts[-200:]) if len(burnin_posts) > 200 else np.var(burnin_posts)
        print(f"Recent log-posterior variance: {recent_var:.3f} (lower = more converged)")
    
    return samples, losses

def train_sgld_advanced(model, x, y, epochs=5000, lr=0.003, burnin=1000, target_samples=400):
    """Advanced SGLD with adaptive sampling and diagnostics"""
    sampler = SGLDSampler(model, lr=lr)
    samples = []
    losses = []
    log_posteriors = []
    
    print(f"Advanced SGLD: Target {target_samples} effective samples")
    print(f"Burnin: {burnin} iterations, then adaptive sampling...")
    
    # Phase 1: Burnin
    for epoch in range(burnin):
        sampler.step(x, y)
        
        with torch.no_grad():
            pred = model(x, sample=False)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())
            
            log_lik = log_likelihood(model, x, y)
            log_p = log_prior(model)
            log_post = log_lik + log_p
            log_posteriors.append(log_post.item())
        
        if epoch % 500 == 0:
            print(f"Burnin {epoch}/{burnin}, Loss: {loss.item():.4f}")
    
    print("Burnin complete. Starting sample collection...")
    
    # Phase 2: Adaptive sampling
    thin_factor = 5  # Start with modest thinning
    samples_since_check = 0
    
    for epoch in range(burnin, epochs):
        sampler.step(x, y)
        
        with torch.no_grad():
            pred = model(x, sample=False)
            loss = F.mse_loss(pred, y)
            losses.append(loss.item())
        
        # Collect samples with current thinning
        if epoch % thin_factor == 0:
            samples.append([param.clone().detach() for param in model.parameters()])
            samples_since_check += 1
        
        # Check effective sample size periodically
        if samples_since_check >= 100 and len(samples) >= 50:
            eff_size = compute_effective_sample_size(samples)
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: {len(samples)} samples, ~{eff_size} effective, thin={thin_factor}")
            
            # Adaptive thinning based on autocorrelation
            if eff_size < len(samples) * 0.1:  # High autocorrelation
                thin_factor = min(thin_factor + 2, 20)
                print(f"High autocorrelation detected. Increasing thinning to {thin_factor}")
            elif eff_size > len(samples) * 0.5:  # Low autocorrelation
                thin_factor = max(thin_factor - 1, 1)
            
            samples_since_check = 0
            
            # Stop early if we have enough effective samples
            if eff_size >= target_samples:
                print(f"Target effective samples reached: {eff_size}")
                break
    
    final_eff_size = compute_effective_sample_size(samples)
    print(f"Final: {len(samples)} total samples, {final_eff_size} effective samples")
    
    return samples, losses

def predict_with_samples(model, x_test, samples):
    """Make predictions using posterior samples"""
    predictions = []
    
    for sample in samples:
        # Set model parameters to sample
        param_idx = 0
        for param in model.parameters():
            param.data = sample[param_idx]
            param_idx += 1
        
        with torch.no_grad():
            pred = model(x_test, sample=False)
            predictions.append(pred.numpy())
    
    return np.array(predictions)

def evaluate_and_plot(x_train, y_train, x_test, y_test, models, method_names):
    """Evaluate models and create plots"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Bayesian Neural Network Comparison', fontsize=16)
    
    x_plot = torch.linspace(-5, 5, 200).unsqueeze(1)
    
    for i, (model, method) in enumerate(zip(models, method_names)):
        ax = axes[i]
        
        if method == 'MFVI':
            # For MFVI, sample multiple times from the learned posterior
            predictions = []
            for _ in range(100):
                with torch.no_grad():
                    pred = model(x_plot, sample=True)
                    predictions.append(pred.numpy())
            predictions = np.array(predictions)
            
        elif method == 'SGLD':
            # Use collected samples
            predictions = predict_with_samples(model, x_plot, sgld_samples)
        
        # Plot uncertainty
        mean_pred = np.mean(predictions, axis=0).flatten()
        std_pred = np.std(predictions, axis=0).flatten()
        
        x_plot_np = x_plot.numpy().flatten()
        
        ax.fill_between(x_plot_np, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                       alpha=0.3, label='95% Confidence')
        ax.fill_between(x_plot_np, mean_pred - std_pred, mean_pred + std_pred, 
                       alpha=0.5, label='68% Confidence')
        ax.plot(x_plot_np, mean_pred, 'r-', label='Mean Prediction', linewidth=2)
        ax.scatter(x_train.numpy(), y_train.numpy(), alpha=0.6, c='blue', label='Training Data')
        ax.scatter(x_test.numpy(), y_test.numpy(), alpha=0.6, c='green', label='Test Data')
        
        ax.set_title(f'{method} Predictions')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compute test RMSE
        with torch.no_grad():
            if method == 'MFVI':
                test_pred = model(x_test, sample=False)
            else:
                # Use mean of samples for test prediction
                test_predictions = predict_with_samples(model, x_test, sgld_samples)
                test_pred = torch.tensor(np.mean(test_predictions, axis=0))
            
            rmse = torch.sqrt(F.mse_loss(test_pred, y_test))
            ax.text(0.05, 0.95, f'Test RMSE: {rmse:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def main():
    print("=== Bayesian Neural Network Training ===\n")
    
    # Generate toy data with fixed seeds for reproducibility
    print("Generating toy 1D regression data...")
    x_train, y_train = generate_toy_data(n_samples=80, noise_std=0.3, seed=42)
    x_test, y_test = generate_toy_data(n_samples=20, noise_std=0.3, seed=123)
    
    print(f"Training data: {len(x_train)} samples")
    print(f"Test data: {len(x_test)} samples")
    
    # Initialize models with same architecture
    models = {
        'MFVI': BNN(hidden_dim=32),
        'SGLD': BNN(hidden_dim=32)
    }
    
    results = {}
    
    # Train with Mean Field Variational Inference (extended training)
    print("\n1. Training with Mean Field Variational Inference...")
    mfvi_model = models['MFVI']
    mfvi_losses = train_mfvi(mfvi_model, x_train, y_train, epochs=20000, lr=0.005)
    results['MFVI'] = {'model': mfvi_model, 'losses': mfvi_losses}
    
    # Train with SGLD (with advanced diagnostics)
    print("\n2. Training with Stochastic Gradient Langevin Dynamics...")
    sgld_model = models['SGLD']
    global sgld_samples
    
    # Choose between basic and advanced SGLD
    use_advanced = True  # Set to False for basic version
    
    if use_advanced:
        sgld_samples, sgld_losses = train_sgld_advanced(
            sgld_model, x_train, y_train, 
            epochs=12000, lr=0.002, burnin=3000, target_samples=500
        )
    else:
        sgld_samples, sgld_losses = train_sgld_basic(
            sgld_model, x_train, y_train, 
            epochs=10000, lr=0.002, burnin=2000
        )
    
    results['SGLD'] = {'model': sgld_model, 'samples': sgld_samples, 'losses': sgld_losses}
    
    # Evaluate and visualize results
    print("\n3. Evaluating and plotting results...")
    
    # Plot training losses
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(mfvi_losses)
    plt.title('MFVI Training Loss (8000 epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('ELBO Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(sgld_losses)
    plt.title('SGLD Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    if use_advanced:
        plt.axvline(x=3000, color='red', linestyle='--', alpha=0.7, label='Burnin End')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot predictions with uncertainty
    evaluate_and_plot(x_train, y_train, x_test, y_test, 
                     [mfvi_model, sgld_model],
                     ['MFVI', 'SGLD'])
    
    # Compute final metrics
    print("\n=== Final Results ===")
    for method, result in results.items():
        model = result['model']
        with torch.no_grad():
            if method == 'MFVI':
                test_pred = model(x_test, sample=False)
            elif method == 'SGLD':
                test_predictions = predict_with_samples(model, x_test, sgld_samples)
                test_pred = torch.tensor(np.mean(test_predictions, axis=0))
            
            rmse = torch.sqrt(F.mse_loss(test_pred, y_test))
            print(f"{method} Test RMSE: {rmse:.4f}")
    
    print(f"\nNumber of SGLD samples collected: {len(sgld_samples)}")
    if use_advanced:
        eff_samples = compute_effective_sample_size(sgld_samples)
        print(f"Effective SGLD samples: {eff_samples}")
    
    print("\n=== Summary ===")
    print("• MFVI: Fast, approximate posterior with parametric form (8000 epochs)")
    sgld_type = "Advanced adaptive" if use_advanced else "Basic fixed"
    sgld_epochs = "12000" if use_advanced else "10000"
    print(f"• SGLD: {sgld_type} sampling with gradient noise ({sgld_epochs} epochs)")
    print("• Both methods trained on identical train/test splits")
    print("\nBoth methods provide uncertainty quantification for robust predictions!")

if __name__ == "__main__":
    main()
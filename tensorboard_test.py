import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1)
        
    def forward(self, x):
        # Sample weights and biases
        weight_sigma = torch.exp(0.5 * self.weight_logvar)
        weight = Normal(self.weight_mu, weight_sigma).rsample()
        
        bias_sigma = torch.exp(0.5 * self.bias_logvar)
        bias = Normal(self.bias_mu, bias_sigma).rsample()
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self):
        weight_kl = -0.5 * torch.sum(1 + self.weight_logvar - self.weight_mu.pow(2) - self.weight_logvar.exp())
        bias_kl = -0.5 * torch.sum(1 + self.bias_logvar - self.bias_mu.pow(2) - self.bias_logvar.exp())
        return weight_kl + bias_kl

class BayesianNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = BayesianLinear(input_dim, hidden_dim)
        self.fc2 = BayesianLinear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
    
    def kl_loss(self):
        return self.fc1.kl_loss() + self.fc2.kl_loss()

def train_epoch(model, dataloader, optimizer, epoch, writer, global_step):
    model.train()
    total_loss = 0
    total_kl = 0
    total_nll = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute losses
        nll_loss = F.cross_entropy(output, target)
        kl_loss = model.kl_loss() / len(dataloader.dataset)
        
        # ELBO loss
        loss = nll_loss + kl_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_kl += kl_loss.item()
        total_nll += nll_loss.item()
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            step = global_step[0]
            writer.add_scalar('Batch/Loss', loss.item(), step)
            writer.add_scalar('Batch/KL_Loss', kl_loss.item(), step)
            writer.add_scalar('Batch/NLL_Loss', nll_loss.item(), step)
            
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Batch/Learning_Rate', current_lr, step)
            
            global_step[0] += 1
    
    avg_loss = total_loss / len(dataloader)
    avg_kl = total_kl / len(dataloader)
    avg_nll = total_nll / len(dataloader)
    
    return avg_loss, avg_kl, avg_nll

def evaluate_model(model, dataloader, num_samples=10):
    model.eval()
    predictions = []
    targets = []
    uncertainties = []
    
    with torch.no_grad():
        for data, target in dataloader:
            # Monte Carlo sampling
            sample_preds = []
            for _ in range(num_samples):
                pred = model(data)
                sample_preds.append(F.softmax(pred, dim=1))
            
            sample_preds = torch.stack(sample_preds)
            mean_pred = sample_preds.mean(dim=0)
            uncertainty = sample_preds.var(dim=0).mean(dim=1)
            
            predictions.append(mean_pred)
            targets.append(target)
            uncertainties.append(uncertainty)
    
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    uncertainties = torch.cat(uncertainties)
    
    accuracy = (predictions.argmax(dim=1) == targets).float().mean()
    avg_uncertainty = uncertainties.mean()
    
    return accuracy, avg_uncertainty, predictions, uncertainties

def log_parameter_histograms(writer, model, epoch):
    """Log parameter histograms to TensorBoard"""
    for name, param in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(f'Parameters/{name}', param.detach().cpu().numpy(), epoch)
    
    # Log weight means and logvars separately for Bayesian layers
    writer.add_histogram('Bayesian/fc1_weight_mu', model.fc1.weight_mu.detach().cpu().numpy(), epoch)
    writer.add_histogram('Bayesian/fc1_weight_logvar', model.fc1.weight_logvar.detach().cpu().numpy(), epoch)
    writer.add_histogram('Bayesian/fc2_weight_mu', model.fc2.weight_mu.detach().cpu().numpy(), epoch)
    writer.add_histogram('Bayesian/fc2_weight_logvar', model.fc2.weight_logvar.detach().cpu().numpy(), epoch)

def log_uncertainty_distribution(writer, uncertainties, epoch):
    """Log uncertainty distribution"""
    uncertainties_np = uncertainties.detach().cpu().numpy()
    writer.add_histogram('Uncertainty/Distribution', uncertainties_np, epoch)
    writer.add_scalar('Uncertainty/Mean', uncertainties_np.mean(), epoch)
    writer.add_scalar('Uncertainty/Std', uncertainties_np.std(), epoch)

# Main training script
if __name__ == "__main__":
    # Configuration
    config = {
        "learning_rate": 0.001,
        "epochs": 100,
        "batch_size": 64,
        "hidden_dim": 128,
        "input_dim": 784,
        "output_dim": 10
    }
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/bnn_experiment_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Log hyperparameters
    writer.add_text('Hyperparameters', str(config))
    
    # Create model and optimizer
    model = BayesianNet(config["input_dim"], config["hidden_dim"], config["output_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Global step counter for batch-level logging
    global_step = [0]
    
    # Log model graph (optional - requires sample input)
    # sample_input = torch.randn(1, config["input_dim"])
    # writer.add_graph(model, sample_input)
    
    print(f"Logging to: {log_dir}")
    print("Start training...")
    print("View with: tensorboard --logdir runs")
    
    # Training loop
    for epoch in range(config["epochs"]):
        # Mock training (replace with actual data loaders)
        # train_loss, kl_loss, nll_loss = train_epoch(model, train_loader, optimizer, epoch, writer, global_step)
        # val_acc, val_uncertainty, predictions, uncertainties = evaluate_model(model, val_loader)
        
        # Mock values for demonstration
        train_loss = 2.5 - 0.02 * epoch + 0.1 * torch.randn(1).item()
        kl_loss = 1.0 - 0.01 * epoch + 0.05 * torch.randn(1).item()
        nll_loss = train_loss - kl_loss
        val_acc = 0.1 + 0.008 * epoch + 0.02 * torch.randn(1).item()
        val_uncertainty = 0.5 - 0.003 * epoch + 0.01 * torch.randn(1).item()
        
        # Create mock uncertainties for histogram logging
        mock_uncertainties = torch.randn(100) * 0.1 + val_uncertainty
        
        # Log epoch-level metrics
        writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
        writer.add_scalar('Epoch/KL_Loss', kl_loss, epoch)
        writer.add_scalar('Epoch/NLL_Loss', nll_loss, epoch)
        writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)
        writer.add_scalar('Epoch/Val_Uncertainty', val_uncertainty, epoch)
        
        # Log parameter histograms every 10 epochs
        if epoch % 10 == 0:
            log_parameter_histograms(writer, model, epoch)
            log_uncertainty_distribution(writer, mock_uncertainties, epoch)
        
        # Log images/plots every 25 epochs (example)
        if epoch % 25 == 0:
            # Create a simple uncertainty vs accuracy plot
            fig = plt.figure(figsize=(8, 6))
            plt.scatter([val_uncertainty], [val_acc], alpha=0.7)
            plt.xlabel('Uncertainty')
            plt.ylabel('Accuracy')
            plt.title(f'Uncertainty vs Accuracy (Epoch {epoch})')
            writer.add_figure('Plots/Uncertainty_vs_Accuracy', fig, epoch)
            plt.close()
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, KL={kl_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Log final model parameters
    log_parameter_histograms(writer, model, config["epochs"])
    
    writer.close()
    print(f"Training complete. Logs saved to: {log_dir}")
    print("View results with: tensorboard --logdir runs")
    
# Note: You'll need to add this import at the top if you want to use the plotting feature
# import matplotlib.pyplot as plt
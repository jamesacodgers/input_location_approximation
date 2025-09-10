import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal

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
        
        # Priors
        self.prior_mu = 0.0
        self.prior_sigma = 1.0
        
    def forward(self, x):
        # Sample weights and biases
        weight_sigma = torch.exp(0.5 * self.weight_logvar)
        weight = Normal(self.weight_mu, weight_sigma).rsample()
        
        bias_sigma = torch.exp(0.5 * self.bias_logvar)
        bias = Normal(self.bias_mu, bias_sigma).rsample()
        
        return F.linear(x, weight, bias)
    
    def kl_loss(self):
        # KL divergence between posterior and prior
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

def train_epoch(model, dataloader, optimizer, epoch, num_batches):
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
        kl_loss = model.kl_loss() / len(dataloader.dataset)  # Scale KL by dataset size
        
        # ELBO loss
        loss = nll_loss + kl_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_kl += kl_loss.item()
        total_nll += nll_loss.item()
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_kl": kl_loss.item(),
                "batch_nll": nll_loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })
    
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
            # Monte Carlo sampling for uncertainty
            sample_preds = []
            for _ in range(num_samples):
                pred = model(data)
                sample_preds.append(F.softmax(pred, dim=1))
            
            sample_preds = torch.stack(sample_preds)
            mean_pred = sample_preds.mean(dim=0)
            uncertainty = sample_preds.var(dim=0).mean(dim=1)  # Predictive variance
            
            predictions.append(mean_pred)
            targets.append(target)
            uncertainties.append(uncertainty)
    
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    uncertainties = torch.cat(uncertainties)
    
    accuracy = (predictions.argmax(dim=1) == targets).float().mean()
    avg_uncertainty = uncertainties.mean()
    
    return accuracy, avg_uncertainty, predictions, uncertainties

# Main training script
if __name__ == "__main__":
    # Initialize wandb
    wandb.init(
        project="bnn-research",
        config={
            "learning_rate": 0.001,
            "epochs": 100,
            "batch_size": 64,
            "hidden_dim": 128,
            "input_dim": 784,
            "output_dim": 10
        }
    )
    
    config = wandb.config
    
    # Create model and optimizer
    model = BayesianNet(config.input_dim, config.hidden_dim, config.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Watch model (logs gradients and parameters)
    wandb.watch(model, log_freq=100)
    
    # Training loop
    for epoch in range(config.epochs):
        # train_loss, kl_loss, nll_loss = train_epoch(model, train_loader, optimizer, epoch, len(train_loader))
        # val_acc, val_uncertainty, predictions, uncertainties = evaluate_model(model, val_loader)
        
        # Mock values for example
        train_loss = 2.5 - 0.02 * epoch + 0.1 * torch.randn(1).item()
        kl_loss = 1.0 - 0.01 * epoch + 0.05 * torch.randn(1).item()
        nll_loss = train_loss - kl_loss
        val_acc = 0.1 + 0.008 * epoch + 0.02 * torch.randn(1).item()
        val_uncertainty = 0.5 - 0.003 * epoch + 0.01 * torch.randn(1).item()
        
        # Log epoch metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "kl_loss": kl_loss,
            "nll_loss": nll_loss,
            "val_accuracy": val_acc,
            "val_uncertainty": val_uncertainty,
        })
        
        # Log histograms every 10 epochs
        if epoch % 10 == 0:
            for name, param in model.named_parameters():
                wandb.log({f"param_hist/{name}": wandb.Histogram(param.detach().cpu().numpy())})
        
        print(f"Epoch {epoch}: Loss={train_loss:.4f}, KL={kl_loss:.4f}, Val Acc={val_acc:.4f}")
    
    wandb.finish()
from matplotlib import pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.approx_bnn import MAPPosterior, MFVIPosterior
from src.layer_priors import LinearLayer
import tqdm
from src.utils import save_results_to_csv, plot_predictions

from torch.utils.data import DataLoader
from src.synthetic_data import generate_ood_synthetic_data, generate_synthetic_data, generate_clean_synthetic_function



# Simple config - you can replace this with Hydra later

def set_seeds(seed):
    """Set seeds for reproducibility across all random number generators.
    
    Args:
        seed (int): The seed value to use for all random number generators
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # PyTorch CUDA (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Uncomment for additional for full reproducibility, but slower code
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def get_likelihood(cfg):
    if cfg.dataset.type == "regression":
        l =torch.distributions.Normal(0, cfg.dataset.noise_std)
    else: 
        raise NotImplementedError("Only regression is implemented in this example.")
    return l

def get_bnn_layer_priors(cfg, input_dim, output_dim, device):
    """Simple model - replace with your BNN later."""
    prior = []
    if cfg.model.type == "mlp":
        layer_sizes = [input_dim] + cfg.model.hidden_layers + [output_dim]
        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            weight_prior = torch.distributions.Normal(torch.zeros(out_features, in_features).to(device), cfg.model.prior_variance*torch.ones(out_features, in_features).to(device))
            bias_prior = torch.distributions.Normal(torch.zeros(out_features).to(device), cfg.model.prior_variance*torch.ones(out_features).to(device))
            if i < len(layer_sizes) - 2:
                activation = torch.nn.ReLU()
            else:
                activation = nn.Identity()
            layer = LinearLayer(in_features, out_features, weight_prior, bias_prior, activation)
            prior.append(layer)
    else:
        raise NotImplementedError("Only MLP model is implemented in this example.")
    return prior

def get_approx_posterior_model(cfg, layer_priors, likelihood):
    """Get the approximate posterior model."""
    if cfg.posterior.type == "map":
        return MAPPosterior(layer_priors=layer_priors, likelihood=likelihood, total_data_points=cfg.dataset.n_train, batch_size=cfg.optimization.batch_size,device="cuda" if torch.cuda.is_available() else "cpu", temperature=cfg.posterior.temperature, posterior_exponentiation=cfg.posterior.posterior_exponentiation)
    elif cfg.posterior.type == "mfvi":
        return MFVIPosterior(layer_priors=layer_priors, likelihood=likelihood, total_data_points=cfg.dataset.n_train, batch_size=cfg.optimization.batch_size,device="cuda" if torch.cuda.is_available() else "cpu", num_samples=cfg.posterior.num_samples, temperature=cfg.posterior.temperature, posterior_exponentiation=cfg.posterior.posterior_exponentiation)
    else:
        raise NotImplementedError("Only MLP model is implemented in this example.")
    
def get_optimizer(cfg, model, train_dataset, test_dataset):
    """Get optimizer and data loaders."""
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=cfg.optimization.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.optimization.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=cfg.optimization.learning_rate)

    lr_scheduler = None

    return optimizer, lr_scheduler, train_loader, test_loader

def get_data(cfg):
    """Simple data loading - replace with your data loader later."""

    x_train, y_train = generate_synthetic_data(n_samples=cfg.dataset.n_train, n_features=cfg.dataset.n_features, n_empty_features=cfg.dataset.n_empty_features, noise_std=cfg.dataset.noise_std, frequency=cfg.dataset.frequency)
    x_test, y_test = generate_synthetic_data(n_samples=cfg.dataset.n_test, n_features=cfg.dataset.n_features, n_empty_features=cfg.dataset.n_empty_features, noise_std=cfg.dataset.noise_std, frequency=cfg.dataset.frequency)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    return train_dataset, test_dataset, x_train.shape[1], y_train.shape[1] 

def get_ood_data_loaders(cfg):
    dataloaders = []
    for input_std in cfg.dataset.ood_input_variance:
        ood_x, ood_y = generate_ood_synthetic_data(n_samples=1000, n_features=cfg.dataset.n_features, n_empty_features=cfg.dataset.n_empty_features, noise_std=cfg.dataset.noise_std, input_std=input_std, frequency=cfg.dataset.frequency)

        ood_dataset = torch.utils.data.TensorDataset(ood_x, ood_y)
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=cfg.optimization.batch_size, shuffle=False)
        dataloaders.append(ood_loader)
    return dataloaders



def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'optimizer_state_dict': optimizer.state_dict()
    }, filepath)
    
    # Save to wandb
    wandb.save(filepath)

def fit_approx_posterior(cfg, model: MAPPosterior, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader, val_dataloader: torch.utils.data.DataLoader, ood_dataloaders: torch.utils.data.DataLoader, lr_scheduler: torch.optim.lr_scheduler.LRScheduler):
    model.train()

    for epoch in range(cfg.optimization.epochs):
        print(epoch)
        for x,y in train_dataloader:
            x = x.to(model.device)
            y = y.to(model.device)
            train_loss = model.train_step(x,y, optimizer)
            
        if epoch % 100 == 0 : 
            model.eval()
            val_ll = torch.zeros(1)
            results_dict = {}
            for x,y in val_dataloader:
                x = x.to(model.device)
                y = y.to(model.device)
                preds = model.predict(x)
                val_ll += (model.get_mean_log_likelihood_contribution(preds,y)*x.shape[0]).item()
            results_dict["val_ll"] = val_ll.item()/len(val_dataloader.dataset)
            for ood_variance,ood_dataloader in zip(cfg.dataset.ood_input_variance, ood_dataloaders):
                ood_ll = torch.zeros(1)
                for x,y in ood_dataloader: 
                    x = x.to(model.device)
                    y = y.to(model.device)
                    preds = model.predict(x)
                    ood_ll += (model.get_mean_log_likelihood_contribution(preds,y)*x.shape[0]).item()
                results_dict[f"ood_ll_var_{ood_variance}"] = ood_ll.item()/len(ood_dataloader.dataset)
            wandb.log(results_dict)
            model.train()
    return model

def test_model(cfg,model, train_dataloader, val_dataloader, ood_dataloaders):
    val_ll = torch.zeros(1)
    ood_ll = torch.zeros(1)
    results_dict = {}
    for x,y in val_dataloader:
        x = x.to(model.device)
        y = y.to(model.device)
        preds = model.predict(x)
        val_ll += (model.get_mean_log_likelihood_contribution(preds,y)*x.shape[0]).item()
    results_dict["val_ll"] = val_ll.item()/len(val_dataloader.dataset)
    print(f"Validation log likelihood: {val_ll.item()/len(val_dataloader.dataset)}")
    for ood_variance,ood_dataloader in zip(cfg.dataset.ood_input_variance, ood_dataloaders):
        ood_ll = torch.zeros(1)
        for x,y in ood_dataloader:
            x = x.to(model.device)
            y = y.to(model.device)
            preds = model.predict(x)
            ood_ll += (model.get_mean_log_likelihood_contribution(preds,y)*x.shape[0]).item()
        results_dict[f"ood_ll_var_{ood_variance}"] = ood_ll.item()/len(ood_dataloader.dataset)
        print(f"OOD log likelihood with variance {ood_variance}: {ood_ll.item()/len(ood_dataloader.dataset)}")
    wandb.log(results_dict)
    save_results_to_csv(cfg, results_dict)
    
    plot_predictions(cfg, model, train_dataloader, val_dataloader, f"iid_predictions_temp_{cfg.posterior.temperature}")
    plot_predictions(cfg, model, train_dataloader, ood_dataloader, f"ood_predictions_temp_{cfg.posterior.temperature}")

@hydra.main(version_base="1.1", config_path="configs", config_name="synthetic_regression")
def main(cfg: OmegaConf):
    set_seeds(cfg.seed)

    print("Config:")
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project="bnn-research",
        config=OmegaConf.to_container(cfg),
        mode="online"  # Change to "offline" if no internet
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model, data, optimizer
    train_dataset, iid_val_dataset, input_dim, output_dim = get_data(cfg)
    ood_dataloaders = get_ood_data_loaders(cfg)
    
    layer_priors = get_bnn_layer_priors(cfg, input_dim, output_dim, device)
    likelihood = get_likelihood(cfg)

    model = get_approx_posterior_model(cfg, layer_priors, likelihood).to(device)


    optimizer, lr_scheduler, train_dataloader, val_dataloader = get_optimizer(cfg, model, train_dataset, iid_val_dataset)

    fit_approx_posterior(cfg, model, optimizer, train_dataloader, val_dataloader, ood_dataloaders, lr_scheduler)

    test_model(cfg, model, train_dataloader, val_dataloader, ood_dataloaders)

if __name__ == "__main__":
    main()
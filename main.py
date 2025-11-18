from matplotlib import pyplot as plt
import numpy as np
import random

import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from src.approx_posterior_bnn import EnsemblePosterior, MAPPosterior, MFVIPosterior, SBVIPosterior, WeightedMFVIPosterior
from src.layer_priors import FourierLayer, LinearLayer
import tqdm
from src.utils import set_seeds, test_model

from torch.utils.data import DataLoader
from src.synthetic_data import generate_ood_synthetic_data, generate_sin_data, generate_clean_sin_function, generate_spiked_linear_data



# Simple config - you can replace this with Hydra later



def get_likelihood(cfg):
    if cfg.dataset.type == "regression":
        l =torch.distributions.Normal(0, cfg.dataset.noise_std)
    else: 
        raise NotImplementedError("Only regression is implemented in this example.")
    return l



def get_bnn_layer_priors(cfg, input_dim, output_dim, device):
    """Simple model - replace with your BNN later."""
    prior = []
    if cfg.model.ff != "None":
        weight_prior = torch.distributions.Normal(torch.zeros(input_dim, cfg.model.ff).to(device), cfg.model.prior_std*torch.ones(input_dim, cfg.model.ff).to(device))
        if cfg.model.ff_amplitudes: # biases are the amplitudes for the ff layer, if we don't learn them set this bias to none
            bias_prior = torch.distributions.Normal(torch.zeros(cfg.model.ff).to(device), cfg.model.prior_std*torch.ones(cfg.model.ff).to(device))
        else:
            bias_prior = None
        fourier_layer = FourierLayer(weight_prior=weight_prior, bias_prior=bias_prior, in_features=input_dim, out_features=cfg.model.ff)
        prior.append(fourier_layer)
        input_dim = cfg.model.ff*2

    if cfg.model.type == "mlp":
        layer_sizes = [input_dim] + cfg.model.hidden_layers + [output_dim]
        for i in range(len(layer_sizes) - 1):
            in_features = layer_sizes[i]
            out_features = layer_sizes[i + 1]
            weight_prior = torch.distributions.Normal(torch.zeros(out_features, in_features).to(device), cfg.model.prior_std*torch.ones(out_features, in_features).to(device))
            bias_prior = torch.distributions.Normal(torch.zeros(out_features).to(device), cfg.model.prior_std*torch.ones(out_features).to(device))
            layer = LinearLayer(in_features, out_features, weight_prior, bias_prior)
            prior.append(layer)
    else:
        raise NotImplementedError("Only MLP model is implemented in this example.")
    return prior

def get_approx_posterior_model(cfg, layer_priors, likelihood):
    """Get the approximate posterior model."""
    if cfg.posterior.type == "map":
        return MAPPosterior(layer_priors=layer_priors, likelihood=likelihood,device="cuda" if torch.cuda.is_available() else "cpu")
    elif cfg.posterior.type == "mfvi":
        return MFVIPosterior(layer_priors=layer_priors, likelihood=likelihood, device="cuda" if torch.cuda.is_available() else "cpu", num_samples=cfg.posterior.num_samples, temperature=cfg.posterior.temperature, posterior_exponentiation=cfg.posterior.posterior_exponentiation)
    elif cfg.posterior.type == "wmfvi":
        weighting_function = lambda x: torch.exp(torch.distributions.Normal(0,1).log_prob(x))
        return WeightedMFVIPosterior(layer_priors=layer_priors, likelihood=likelihood, device="cuda" if torch.cuda.is_available() else "cpu", num_samples=cfg.posterior.num_samples, temperature=cfg.posterior.temperature, posterior_exponentiation=cfg.posterior.posterior_exponentiation, weighting_function=weighting_function)
    elif cfg.posterior.type == "ensemble":
        return EnsemblePosterior(layer_priors=layer_priors, likelihood=likelihood, device="cuda" if torch.cuda.is_available() else "cpu", num_samples=cfg.posterior.num_samples)
    elif cfg.posterior.type == "sbvi":
        return SBVIPosterior(layer_priors=layer_priors, likelihood=likelihood, device="cuda" if torch.cuda.is_available() else "cpu", num_samples=cfg.posterior.num_samples, n_squash_vectors=cfg.posterior.num_squash_vectors, temperature=cfg.posterior.temperature, posterior_exponentiation=cfg.posterior.posterior_exponentiation, n_data=cfg.dataset.n_train)
    else:
        raise NotImplementedError()
    
def get_optimizer(cfg, model, train_dataset):
    """Get optimizer and data loaders."""
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=cfg.optimization.batch_size, shuffle=True)
    if cfg.optimization.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimization.learning_rate)
    elif cfg.optimization.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.optimization.learning_rate)
    else:
        raise NotImplementedError("only SGD and Adam available")
    lr_scheduler = None

    return optimizer, lr_scheduler, train_loader

def get_data(cfg):
    """Simple data loading - replace with your data loader later."""
    if cfg.dataset.label == "sin":
        x_train, y_train = generate_sin_data(n_samples=cfg.dataset.n_train, n_features=cfg.dataset.n_features, n_empty_features=cfg.dataset.n_empty_features, noise_std=cfg.dataset.noise_std, frequency=cfg.dataset.frequency, input_std=cfg.dataset.input_std)
    if cfg.dataset.label == "spiked_linear":
        x_train, y_train = generate_spiked_linear_data(n_samples=cfg.dataset.n_train, rank=cfg.dataset.rank, n_features=cfg.dataset.n_features, noise_std=cfg.dataset.noise_std, beta_std=cfg.model.prior_std)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

    return (
            train_dataset, 
            x_train.shape[1], 
            y_train.shape[1] 
    )

def get_ood_data_loaders(cfg):
    dataloaders = []
    if cfg.dataset.label == "sin":
        for input_std in cfg.dataset.ood_input_variance:
            ood_x, ood_y = generate_ood_synthetic_data(n_samples=1000, n_features=cfg.dataset.n_features, n_empty_features=cfg.dataset.n_empty_features, noise_std=cfg.dataset.noise_std, input_std=input_std, frequency=cfg.dataset.frequency)

            ood_dataset = torch.utils.data.TensorDataset(ood_x, ood_y)
            ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=cfg.optimization.batch_size, shuffle=False)
            dataloaders.append(ood_loader)
    if cfg.dataset.label == "spiked_linear":
        assert len(cfg.dataset.ood_input_variance)==1 
        for input_std in cfg.dataset.ood_input_variance:
            ood_x, ood_y = generate_spiked_linear_data(n_samples=cfg.dataset.n_train, rank=cfg.dataset.rank, n_features=cfg.dataset.n_features, noise_std=cfg.dataset.noise_std, beta_std=cfg.model.prior_std)

            ood_dataset = torch.utils.data.TensorDataset(ood_x, ood_y)
            ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=cfg.optimization.batch_size, shuffle=False)
            dataloaders.append(ood_loader)
    return dataloaders





def fit_approx_posterior(cfg, model: MAPPosterior, optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader, ood_dataloaders: torch.utils.data.DataLoader, lr_scheduler: torch.optim.lr_scheduler.LRScheduler):
    model.train()
    monitor=1
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    for epoch in range(cfg.optimization.epochs):
        if epoch % 100 == 0:
            print(epoch)
        for x,y in train_dataloader:
            x = x.to(model.device)
            y = x.to(model.device)
            train_loss = model.train_step(x,y, optimizer)
            results_dict = {}
            results_dict["train_loss"] = train_loss
            
        # if epoch % 10_000 == 0 : 
        if epoch % monitor == monitor-1: 
            model.eval()
            test_model(cfg, model, train_dataloader, ood_dataloaders, epoch, label=epoch)
            model.train()
            # model.temperature = model.temperature*2
            # lr_scheduler.step()
        wandb.log(results_dict)
    return model



@hydra.main(version_base="1.1", config_path="configs", config_name="config")
def main(cfg: OmegaConf):
    torch.set_default_dtype(torch.float64)
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
    train_dataset, input_dim, output_dim = get_data(cfg)
    ood_dataloaders = get_ood_data_loaders(cfg)
    

    layer_priors = get_bnn_layer_priors(cfg, input_dim, output_dim, device)
    likelihood = get_likelihood(cfg)

    model = get_approx_posterior_model(cfg, layer_priors, likelihood).to(device)


    optimizer, lr_scheduler, train_dataloader = get_optimizer(cfg, model, train_dataset)

    fit_approx_posterior(cfg, model, optimizer, train_dataloader, ood_dataloaders, lr_scheduler)

    test_model(cfg, model, train_dataloader, ood_dataloaders, cfg.optimization.epochs)

if __name__ == "__main__":
    main()
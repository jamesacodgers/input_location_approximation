
import csv
import matplotlib.pyplot as plt
import torch 
import wandb
import os
import random

import numpy as np

from src.synthetic_data import generate_clean_sin_function, generate_clean_spiked_linear_function

def test_model(cfg,model, train_dataloader, ood_dataloaders, epoch, label="final"):
    ood_ll = torch.zeros(1)
    results_dict = {"epoch":epoch}
    for ood_variance,ood_dataloader in zip(cfg.dataset.ood_input_variance, ood_dataloaders):
        ood_ll = torch.zeros(1)
        for x,y in ood_dataloader:
            x = x.to(model.device)
            y = y.to(model.device)
            preds = model.sample_functions(x)
            ood_ll += (model.get_mean_log_likelihood(preds,y)*x.shape[0]).item()
        results_dict[f"ood_ll_var_{ood_variance}"] = ood_ll.item()/len(ood_dataloader.dataset)
        print(f"OOD log likelihood with variance {ood_variance}: {ood_ll.item()/len(ood_dataloader.dataset)}")
    wandb.log(results_dict)
    save_results_to_csv(cfg, results_dict)
    
    if cfg.posterior.get("temperature") is not None:
        title = f"{label}_predictions_temp_{cfg.posterior.temperature}"
    else: 
        title = f"{label}_{cfg.posterior.type}"

    plot_predictions(cfg, model, train_dataloader, title+"_iid", margin=1)
    plot_predictions(cfg, model, train_dataloader, title+"_ood", margin=10)
    if cfg.dataset.label=="sin":
        plot_model_ft(cfg, model, x_min=-10, x_max=10, max_frequency=2**13, name=title+"_ft", n_samples=10)

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


def save_results_to_csv(cfg, results_dict):
    """Save results in current Hydra output directory."""
    if cfg.posterior.get("temperature") is not None:
        results_dict["temperature"] = cfg.posterior.temperature
        results_dict["posterior_type"] = cfg.posterior.type  
    results_dict["seed"] = cfg.seed
    results_dict["n_train"] = cfg.dataset.n_train
    
    # Only add frequency if it exists in config
    if "frequency" in cfg.dataset:
        results_dict["frequency"] = cfg.dataset.frequency
    
    # Check if file exists
    file_exists = os.path.exists('results.csv')
    
    # Save to current working directory (Hydra's output dir)
    mode = 'a' if file_exists else 'w'
    with open('results.csv', mode, newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_dict)

def plot_predictions(cfg, model, train_dataloader, name: str, margin=1):
    min_x = train_dataloader.dataset.tensors[0].min()-margin
    max_x = train_dataloader.dataset.tensors[0].max()+margin
    if cfg.dataset.label =="sin":
        x_lin,f = generate_clean_sin_function(n_samples=200, n_features=cfg.dataset.n_features, n_empty_features=cfg.dataset.n_empty_features, min_x=min_x, max_x=max_x, frequency=cfg.dataset.frequency)
    elif cfg.dataset.label=="spiked_linear":
        x_lin, f = generate_clean_spiked_linear_function(n_samples=200, n_features=cfg.dataset.n_features, min_x=min_x, max_x=max_x, rank=cfg.dataset.rank, beta_std=cfg.model.prior_std)
    else:
        raise NotImplementedError("whoops")
    x_lin = x_lin.to(model.device)

    preds = model.predict(x_lin)
    sampled_preds = model.sample_functions(x_lin, n_samples=10)
    lower, upper = model.get_CI(x_lin, ci=0.95)
    with torch.no_grad():
        fig,ax = plt.subplots()
        for x,y in train_dataloader: 
            ax.scatter(x[:,0].cpu(),y.cpu(), c="orange", label="train_data")
        ax.plot(x_lin[:,0].cpu(),f.cpu(), c="green", label="true_function")
        ax.plot(x_lin[:,0].cpu(), preds[:,0].cpu(), c="red", label="predictions")
        ax.fill_between(x_lin[:,0].cpu(), lower[:,0].cpu(), upper[:,0].cpu(), color="red", alpha=0.3, label="95% CI")
        for pred in sampled_preds:
            ax.plot(x_lin[:,0].cpu(), pred[:,0].cpu(), c="red", alpha=0.5)
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Predictions vs Data")
        plt.savefig(name+".pdf")
        wandb.log({name: wandb.Image(fig)})
        # plt.show()

# def plot_model_ft(cfg, model, x_min, x_max, max_frequency = 64, name="model_fourier_transform"):
#     x = torch.linspace(x_min, x_max, steps=max_frequency).unsqueeze(1).to(model.device)
#     with torch.no_grad():
#         preds = model.predict(x)
#         sampled_preds = model.sample_functions(x, n_samples=10)
#         ft = torch.fft.fftshift(torch.fft.fft(preds.squeeze()))
#         ft_magnitude = torch.abs(ft)
#         ft_freq = torch.fft.fftshift(torch.fft.fftfreq(len(x), d=(x[1]-x[0]).item()))
#         fig, ax = plt.subplots(1,1,figsize=(8,10))
#         ax.plot(ft_freq.cpu(), ft_magnitude.cpu(), c="blue", label="Fourier Transform Magnitude")
#         ax.vlines(cfg.dataset.frequency, 0, ft_magnitude.max().cpu(), colors='red', linestyles='dashed', label="True Frequency")
#         plt.savefig(name+".pdf")
#         wandb.log({name: wandb.Image(fig)})

def plot_model_ft(cfg, model, x_min, x_max, max_frequency = 1024, name="model_fourier_transform", n_samples=10):
    x = torch.linspace(x_min, x_max, steps=max_frequency).unsqueeze(1).to(model.device)
    with torch.no_grad():
        preds = model.predict(x)
        sampled_preds = model.sample_functions(x, n_samples=n_samples)
        ft = torch.fft.fftshift(torch.fft.fft(preds.squeeze()))
        ft_magnitude = torch.abs(ft)
        ft_freq = torch.fft.fftshift(torch.fft.fftfreq(len(x), d=(x[1]-x[0]).item()))
        
        # Combine positive and negative frequencies
        # Find the center (DC component)
        center_idx = len(ft_freq) // 2
        
        # Get positive frequencies (right half, excluding DC)
        pos_freq = ft_freq[center_idx+1:]
        pos_magnitude = ft_magnitude[center_idx+1:]
        
        # Get negative frequencies (left half, excluding DC)
        neg_freq = ft_freq[:center_idx]
        neg_magnitude = ft_magnitude[:center_idx]
        
        # Handle size mismatch for even-length arrays
        min_len = min(len(pos_magnitude), len(neg_magnitude))
        pos_magnitude_trimmed = pos_magnitude[:min_len]
        neg_magnitude_trimmed = neg_magnitude[-min_len:]  # Take the last min_len elements
        
        # Combine by adding corresponding positive and negative frequency magnitudes
        # Note: negative frequencies correspond to positive frequencies in reverse order
        combined_magnitude = pos_magnitude_trimmed + torch.flip(neg_magnitude_trimmed, [0])
        
        # Include DC component
        dc_component = ft_magnitude[center_idx]
        combined_freq = torch.cat([torch.tensor([0.0]).to(pos_freq.device), pos_freq[:min_len]])
        combined_magnitude = torch.cat([dc_component.unsqueeze(0), combined_magnitude])
        
        fig, ax = plt.subplots(1,1,figsize=(8,10))
        ax.plot(torch.log(combined_freq).cpu(), (combined_magnitude.cpu()), c="blue", label="Combined Fourier Transform Magnitude")
        for sample in sampled_preds:
            sample_ft = torch.fft.fftshift(torch.fft.fft(sample.squeeze()))
            sample_ft_magnitude = torch.abs(sample_ft)
            
            # Combine positive and negative frequencies for the sample
            pos_sample_magnitude = sample_ft_magnitude[center_idx+1:]
            neg_sample_magnitude = sample_ft_magnitude[:center_idx]
            
            # Handle size mismatch for even-length arrays
            min_len = min(len(pos_sample_magnitude), len(neg_sample_magnitude))
            pos_sample_magnitude_trimmed = pos_sample_magnitude[:min_len]
            neg_sample_magnitude_trimmed = neg_sample_magnitude[-min_len:]  # Take the last min_len elements
            
            combined_sample_magnitude = pos_sample_magnitude_trimmed + torch.flip(neg_sample_magnitude_trimmed, [0])
            combined_sample_magnitude = torch.cat([sample_ft_magnitude[center_idx].unsqueeze(0), combined_sample_magnitude])
            
            ax.plot(torch.log(combined_freq).cpu(),(combined_sample_magnitude.cpu()), c="blue", alpha=0.3)
            # ax.plot(torch.log(combined_freq).cpu(), torch.log(combined_sample_magnitude.cpu()), c="blue", alpha=0.3)
        ax.vlines(np.log(cfg.dataset.frequency), 0, combined_magnitude.max().cpu(), colors='red', linestyles='dashed', label="True Frequency")
        plt.xlabel("Log Frequency")
        plt.ylabel("Magnitude")
        plt.legend()
        plt.savefig(name+".pdf")
        wandb.log({name: wandb.Image(fig)})
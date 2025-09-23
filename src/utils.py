
import csv
import matplotlib.pyplot as plt
import torch 
import wandb

from src.synthetic_data import generate_clean_synthetic_function
def save_results_to_csv(cfg, results_dict):
    """Save results in current Hydra output directory."""

    results_dict["temperature"] = cfg.posterior.temperature
    results_dict["posterior_type"] = cfg.posterior.type
    
    # Save to current working directory (Hydra's output dir)
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results_dict.keys())
        writer.writeheader()
        writer.writerow(results_dict)

def plot_predictions(cfg, model, train_dataloader, val_dataloader, name: str):
    min_x = val_dataloader.dataset.tensors[0].min()
    max_x = val_dataloader.dataset.tensors[0].max()
    x_lin,f = generate_clean_synthetic_function(n_samples=200, n_features=cfg.dataset.n_features, n_empty_features=cfg.dataset.n_empty_features, min_x=min_x, max_x=max_x)
    x_lin = x_lin.to(model.device)

    preds = model.predict(x_lin)
    lower, upper = model.get_CI(x_lin, ci=0.95)
    with torch.no_grad():
        fig,ax = plt.subplots()
        for x,y in train_dataloader: 
            ax.scatter(x[:,0].cpu(),y.cpu(), c="orange", label="train_data")
        for x,y in val_dataloader: 
            ax.scatter(x[:,0].cpu(),y.squeeze().cpu(), c="blue", label="val_data")
        ax.plot(x_lin[:,0].cpu(),f.cpu(), c="green", label="true_function")
        ax.plot(x_lin[:,0].cpu(), preds[:,0].cpu(), c="red", label="predictions")
        ax.fill_between(x_lin[:,0].cpu(), lower[:,0].cpu(), upper[:,0].cpu(), color="red", alpha=0.3, label="95% CI")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Predictions vs Data")
        plt.savefig(name+".png")
        wandb.log({name: wandb.Image(fig)})
        # plt.show()
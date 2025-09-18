
import csv
import matplotlib.pyplot as plt
import torch 
import wandb

from src.synthetic_data import generate_clean_synthetic_function
def save_results_to_csv(cfg, final_val_ll):
    """Save results in current Hydra output directory."""
    result = {
        'temperature': cfg.posterior.temperature,
        'posterior_type': cfg.posterior.type,
        'final_val_ll': final_val_ll,
    }
    
    # Save to current working directory (Hydra's output dir)
    with open('results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)

def plot_predictions(cfg, model, train_dataloader, val_dataloader):
    x_lin,f = generate_clean_synthetic_function(n_samples=200, n_features=cfg.dataset.n_features, n_empty_features=cfg.dataset.n_empty_features)
    preds = model.predict(x_lin)
    lower, upper = model.get_CI(x_lin, ci=0.95)
    with torch.no_grad():
        fig,ax = plt.subplots()
        for x,y in train_dataloader: 
            ax.scatter(x[:,0],y, c="orange", label="train_data")
        for x,y in val_dataloader: 
            ax.scatter(x[:,0],y, c="blue", label="val_data")
        ax.plot(x_lin[:,0],f, c="green", label="true_function")
        ax.plot(x_lin[:,0], preds[:,0], c="red", label="predictions")
        ax.fill_between(x_lin[:,0], lower[:,0], upper[:,0], color="red", alpha=0.3, label="95% CI")
        ax.legend()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Predictions vs Data")
        plt.savefig(f"predictions_temp_{cfg.posterior.temperature}.png")
        wandb.log({f"predictions temp = {cfg.posterior.temperature}": wandb.Image(fig)})
        # plt.show()
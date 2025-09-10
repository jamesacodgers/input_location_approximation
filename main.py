import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Simple config - you can replace this with Hydra later
config = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 10,
    "model_type": "simple_bnn",
    "dataset": "mnist",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def get_model():
    """Simple model - replace with your BNN later."""
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

def get_data():
    """Simple data loading - replace with your data loader later."""
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, test_loader

def train_one_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        # Log individual batch losses occasionally
        if batch_idx % 100 == 0:
            wandb.log({
                "batch_loss": loss.item(),
                "batch_idx": batch_idx
            })
        
        # Accumulate for epoch average
        total_loss += loss.item() * data.size(0)
        total_samples += data.size(0)
    
    return total_loss / total_samples

def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data = data.view(-1, 784).to(device)
            target = target.to(device)
            
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_samples += data.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    
    return avg_loss, accuracy

def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,  # Save config too
    }, filepath)
    
    # Save to wandb
    wandb.save(filepath)

def main():
    """Main training function."""
    
    # Initialize wandb
    wandb.init(
        project="bnn-research",
        config=config,
        mode="online"  # Change to "offline" if no internet
    )
    
    # Setup
    device = torch.device(config["device"])
    print(f"Using device: {device}")
    
    # Create model, data, optimizer
    model = get_model().to(device)
    train_loader, test_loader = get_data()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print(f"Starting training for {config['epochs']} epochs...")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config["epochs"]):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss, val_accuracy = validate(model, test_loader, device)
        
        # Log to wandb
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": config["learning_rate"]
        })
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Save checkpoint if best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, "best_model.pth")
            print(f"New best model saved at epoch {epoch}")
    
    # Finish wandb run
    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    main()
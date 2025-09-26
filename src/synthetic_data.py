import torch 
import matplotlib.pyplot as plt

def get_random_linear_input_data(n_samples: int, n_features: int, n_empty_features: int, std: float):
    """
    Generate inputs which lie on a lower dimensional linear subspace within a higher-dimensional space.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of informative features.
        n_empty_features: Number of non-informative (zero) features.
    Returns:
        Tensor of shape (n_samples, n_features + n_empty_features)
    """
    x_full = torch.randn(n_samples, n_features) * std
    x_empty = torch.zeros(n_samples, n_empty_features)

    return torch.cat([x_full, x_empty], dim=1)

def get_linspace_linear_input_data(n_samples: int, n_features: int, n_empty_features: int, min_x: float = -3.5, max_x: float=3.5):
    """
    Generate inputs which lie on a lower dimensional linear subspace within a higher-dimensional space.

    Args:
        n_samples: Number of samples to generate.
        n_features: Number of informative features.
        n_empty_features: Number of non-informative (zero) features.
    Returns:
        Tensor of shape (n_samples, n_features + n_empty_features)
    """
    x_full = torch.linspace(min_x, max_x, n_samples).unsqueeze(1).expand(-1, n_features)
    x_empty = torch.zeros(n_samples, n_empty_features)

    return torch.cat([x_full, x_empty], dim=1)


def synthetic_function(x: torch.Tensor, frequency: float) -> torch.Tensor:
    """
    A synthetic function that takes a 2D input and produces a scalar output.
    The function is defined as: f(x) = cos(2pi|x|_1)

    Args:
        x: Input tensor of shape (n_samples, 2)
    Returns:
        Tensor of shape (n_samples,) representing the function values.
    """
    x_norm = torch.sum(x, dim=1, keepdim=True)

    return torch.cos(2 * torch.pi * frequency * x_norm)

def apply_gaussian_noise(f, noise_std: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to the function values.

    Args:
        f: Function values tensor of shape (n_samples,)
        noise_std: Standard deviation of the Gaussian noise.
    Returns:
        Noisy function values tensor of shape (n_samples,)
    """
    noise = torch.randn_like(f) * noise_std
    return f + noise

def generate_synthetic_data(n_samples: int = 100, n_features: int = 1, n_empty_features: int = 9, noise_std: float = 0.1, input_std: float = 1.0, frequency: float = 1.0):
    x = get_random_linear_input_data(n_samples, n_features, n_empty_features, input_std)
    f = synthetic_function(x, frequency)
    y = apply_gaussian_noise(f, noise_std)
    return x, y

def generate_clean_synthetic_function(n_samples: int = 100, n_features: int = 1, n_empty_features: int = 9, min_x = -3, max_x = 3, frequency: float = 1.0):
    x = get_linspace_linear_input_data(n_samples, n_features, n_empty_features, min_x, max_x)
    f = synthetic_function(x, frequency)
    return x, f

def generate_ood_synthetic_data(n_samples: int, n_features: int, n_empty_features: int, noise_std: float, input_std: float, frequency: float):
    x,y = generate_synthetic_data(n_samples, n_features, n_empty_features, noise_std, input_std, frequency)
    return x, y



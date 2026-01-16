import torch
from src.synthetic_data import generate_gap_sin_data

def verify_gap_sin():
    n_samples = 1000
    gap_min = 1.0
    gap_max = 3.5
    n_features = 1
    n_empty_features = 0
    
    x, y = generate_gap_sin_data(n_samples=n_samples, n_features=n_features, n_empty_features=n_empty_features, gap_min=gap_min, gap_max=gap_max)
    
    # Check shape
    assert x.shape == (n_samples, n_features + n_empty_features)
    assert y.shape == (n_samples, 1), f"Expected y.shape to be {(n_samples, 1)}, got {y.shape}"
    
    # Check values
    x_vals = x[:, :n_features]
    
    # Check that no values are in (-gap_min, gap_min)
    in_gap = (x_vals > -gap_min) & (x_vals < gap_min)
    assert not in_gap.any(), "Found values in the gap!"
    
    # Check that all values are in [-gap_max, -gap_min] U [gap_min, gap_max]
    valid_range = ((x_vals >= -gap_max) & (x_vals <= -gap_min)) | ((x_vals >= gap_min) & (x_vals <= gap_max))
    assert valid_range.all(), "Found values outside the valid range!"
    
    print("Verification passed!")
    print(f"Generated {n_samples} samples.")
    print(f"Min value: {x_vals.min().item()}")
    print(f"Max value: {x_vals.max().item()}")
    print(f"Gap: (-{gap_min}, {gap_min})")

if __name__ == "__main__":
    verify_gap_sin()

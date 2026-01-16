import torch
import torch.nn as nn
import ivon

model = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))
optimizer = ivon.IVON(model.parameters(), lr=0.1, ess=20)

# IVON expects multiple steps to initialize some moving averages
for _ in range(5):
    optimizer.zero_grad()
    x = torch.randn(8, 1)
    y = torch.randn(8, 1)
    with optimizer.sampled_params(train=True):
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
    optimizer.step()

# Correct way to access Hessian in ivon is through the param groups 
for group in optimizer.param_groups:
    print(f"Group keys: {group.keys()}")
    if 'hess' in group:
        print(f"Hessian shape: {group['hess'].shape}")

import torch

device = torch.device("cpu")
print("Using device:", device)

try:
    x = torch.randn(1).to(device)
    print("Tensor on device:", x)
except Exception as e:
    print("Error moving tensor to device:", e)
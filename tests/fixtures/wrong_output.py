# Runs fine but returns zeros: expect status=fail with a mismatch message.
import torch

def custom_kernel(data):
    x, weights, config = data
    return torch.zeros_like(x)

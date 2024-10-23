# utils.py

import torch
from constants import USE_GPU

def get_device():
    """Determine the device to use for torch operations."""
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device
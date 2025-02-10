import torch
# set up GPU for PyTorch


def set_gpu():
    """Set up GPU for PyTorch."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device

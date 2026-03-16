import torch

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

DEVICE = get_device()

CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5

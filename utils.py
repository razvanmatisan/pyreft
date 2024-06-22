import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda:0"
    try:
        if torch.backends.mps.is_available():
            return "mps"
    except AttributeError:
        pass
    return "cpu"

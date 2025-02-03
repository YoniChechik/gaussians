import os
import random

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def seed_everything(seed=0):
    """This function sets all sources of random numbers generators to a constant seed.
    For full reproducibility (including torch cuda elements), check https://pytorch.org/docs/stable/notes/randomness.html"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)

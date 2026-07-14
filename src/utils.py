from .config import *
import random
import numpy as np
import torch

def set_seed(seed=42):
    
    # Python standard
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    print(f"\nRandom Seed set to: {seed}")
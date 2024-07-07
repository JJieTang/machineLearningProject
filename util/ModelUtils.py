import torch 
import torch.nn as nn
import numpy as np
from src.model import NCA

class ModelUtils:
    def get_living_mask(x):
        alpha = x[:, 90:91, ...]
        max_pool = torch.nn.MaxPool3d(kernel_size=21, stride=1, padding=21//2)
        alpha = max_pool(alpha)
        return alpha > 0.1


    def non_liquid_mask(x):
        alpha = x[:, 90:91, ...]
        return alpha < 0.99
    
    


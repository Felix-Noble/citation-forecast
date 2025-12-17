from typing import override
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor

@dataclass
class ModelConfig:
    n_layers: int

class R_RNN(nn.Module):
    def __init__(self, 
                 config: ModelConfig,
                 device: torch.device,
                 dtype: torch.dtype
                 ):
        super().__init__()
    
    @override
    def forward(self, x: Tensor):
        pass



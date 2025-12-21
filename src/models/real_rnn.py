from typing import override
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch import Tensor

class R_RNN(nn.Module):
    MODEL_NAME = 'r_rnn'
    def __init__(self, 
                 config,
                 device: torch.device,
                 dtype: torch.dtype
                 ):
        super().__init__()

    @override
    def forward(self, x: Tensor):
        return x        

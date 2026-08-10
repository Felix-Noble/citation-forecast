import torch.nn as nn
from utils import component
from torch import Tensor, log, mean, abs
import torch.nn.functional as F
from typing import Protocol

class Output(Protocol):
    prediction: Tensor
    sigma: Tensor # variance to smooth loss by

class Batch(Protocol):
    y: Tensor

@component
class MAE_Var(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
    
    def forward(self,output: Output, batch: Batch) -> Tensor:
        mae = abs(batch.y - output.prediction)
        loss = (mae / output.sigma) + log(output.sigma)
        loss_scalar = mean(loss)
        return loss_scalar

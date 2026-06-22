from ._registry import loss_registry
import torch.nn as nn
from torch import Tensor, log, mean
import torch.nn.functional as F
from typing import Protocol

class Output(Protocol):
    prediction: Tensor
    sigma: Tensor # variance to smooth loss by

class Batch(Protocol):
    y: Tensor

@loss_registry
class MAE_Var(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mae_loss = nn.L1Loss()
    
    def forward(self,output: Output, batch: Batch) -> Tensor:
        mae = self.mae_loss(output.prediction, batch.y)
        loss = (mae / output.sigma) + log(output.sigma)
        loss_scalar = mean(loss)
        return loss_scalar

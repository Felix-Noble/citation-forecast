from typing import Protocol, override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, log, mean

from utils import component


class Output(Protocol):
    prediction: Tensor
    sigma: Tensor  # variance to smooth loss by


class Batch(Protocol):
    y: Tensor


@component
class MAE(nn.L1Loss):
    def __init__(self, config):
        super().__init__()
        self.fn = nn.L1Loss()

    @override
    def forward(
        self,
        output: Output,
        batch: Batch,
    ) -> Tensor:
        return self.fn(output.prediction, batch.y)

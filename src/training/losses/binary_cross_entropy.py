from typing import Protocol, override

import torch
from torch import Tensor
from torch.nn.functional import binary_cross_entropy
from torch.nn import BCELoss

from utils import component

from .loss_protocol import LossFn


class BCEBatch(Protocol):
    y: Tensor
    weight: Tensor


class BCEOutput(Protocol):
    probs: Tensor


@component
class BinaryCrossEntropyLoss(BCELoss, LossFn[BCEBatch, BCEOutput]):
    def __init__(self, config):
        super().__init__()
        self.torch_bce_loss = binary_cross_entropy

    @override
    def __call__(self, output: BCEOutput, batch: BCEBatch) -> Tensor:
        return self.torch_bce_loss(
            input=output.probs.squeeze(-1),
            target=batch.y.squeeze(-1),
            weight=batch.weight.squeeze(-1) if not torch.any(torch.isnan(batch.weight)) else None,
        )

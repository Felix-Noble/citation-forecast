from typing import Protocol, override

import torch
from torch import Tensor
from torch.nn import BCELoss

from ._registry import loss_registry
from .loss_protocol import LossFn


class BCEBatch(Protocol):
    y: Tensor


class BCEOutput(Protocol):
    probs: Tensor


@loss_registry("BinaryCrossEntropyLoss")
class BinaryCrossEntropyLoss(BCELoss, LossFn[BCEBatch, BCEOutput]):
    def __init__(self, config):
        super().__init__()
        self.torch_bce_loss = BCELoss()

    @override
    def __call__(self, output: BCEOutput, batch: BCEBatch) -> Tensor:
        return self.torch_bce_loss(
            input=output.probs.squeeze(-1),
            target=batch.y.squeeze(-1),
        )

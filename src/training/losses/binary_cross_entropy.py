from typing import override
import torch
from torch import Tensor
from torch.nn import BCELoss
from ._registry import loss_registry

@loss_registry('BinaryCrossEntropyLoss')
class BinaryCrossEntropyLoss(BCELoss):
    def __init__(self):
        super().__init__()
        self.torch_bce_loss = BCELoss()

    @override
    def __call__(
            self, 
            probs: Tensor,
            target: Tensor,
            **kwargs
            ) -> Tensor:
        return self.torch_bce_loss(input=probs.squeeze(-1), target=target.squeeze(-1))

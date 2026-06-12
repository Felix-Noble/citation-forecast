from ._registry import loss_registry
from typing import override
from torch import Tensor
import torch.nn as nn

@loss_registry('CrossEntropyLoss')
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
        self.torch_ce_loss = nn.CrossEntropyLoss()

    @override
    def __call__(
            self, 
            probs: Tensor,
            target: Tensor,
            **kwargs
            ) -> Tensor:
        print('probs', probs.shape) 
        print('target', target.shape) 
        return self.torch_ce_loss(input=probs, target=target.squeeze(-1))

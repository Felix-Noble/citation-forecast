from ._registry import loss_registry
from typing import override
from torch import Tensor
import torch.nn as nn

@loss_registry('CrossEntropyLoss')
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, config):
        super().__init__()
        self.torch_ce_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=config.model.pad_token_id)
        self.n_classes = config.model.n_out

    @override
    def __call__(
            self, 
            logits: Tensor,
            target: Tensor,
            **kwargs
            ) -> Tensor:
        loss = self.torch_ce_loss(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss

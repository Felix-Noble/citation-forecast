from typing import Protocol, override

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._registry import loss_registry


class CEBatch(Protocol):
    y: Tensor
    weight: Tensor


class CEOutput(Protocol):
    logits: Tensor


@loss_registry("CrossEntropyLoss")
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, config):
        super().__init__()
        self.n_classes = config.model.model.n_out
        self._CELoss = F.cross_entropy
        self.ignore_index = config.data.train.dataset.pad_token_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.data.train.dataset.weights is not None:
            self.weights = config.data.train.dataset.weights.to(self.device)
        else:
            self.weights = None

    @override
    def __call__(self, output: CEOutput, batch: CEBatch) -> Tensor:
        loss = self._CELoss(
            output.logits.view(-1, output.logits.size(-1)),
            batch.y.view(-1),
            # weight=self.weights,
            ignore_index=self.ignore_index,
        )
        return loss

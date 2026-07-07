from typing import Protocol, override

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, log, mean

from ._registry import loss_registry
from .wasserstein_funcs import wasserstein_loss


class Output(Protocol):
    probs: Tensor


class Batch(Protocol):
    target_indicies: Tensor


@loss_registry("WassersteinLoss")
class WassersteinLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fn = wasserstein_loss
        self.n_classes = config.model.model.n_out

    @override
    def forward(
        self,
        output: Output,
        batch: Batch,
    ) -> Tensor:
        target = F.one_hot(batch.y, num_classes=self.n_classes)
        return self.fn(output.probs, target)

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor
from torch.optim import Optimizer


@runtime_checkable
class OptimizerSpec(Protocol):
    """IDE-checkable optimizer configuration (F4)."""

    def build(self, *, params: Iterable[Tensor]) -> Optimizer: ...


@dataclass(kw_only=True)
class AdamWSpec(OptimizerSpec):
    """Typed AdamW configuration."""

    lr: float
    weight_decay: float

    def build(self, *, params: Iterable[Tensor]) -> torch.optim.AdamW:
        return torch.optim.AdamW(params=params, lr=self.lr, weight_decay=self.weight_decay)

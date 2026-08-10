from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


@runtime_checkable
class LRSchedulerSpec(Protocol):
    """IDE-checkable LR scheduler configuration (F4/J2)."""

    def build(self, *, optimizer: Optimizer) -> LRScheduler: ...


@dataclass(kw_only=True)
class WarmupCosineSpec(LRSchedulerSpec):
    """Reproduces the historical ``build_lr_scheduler`` semantics exactly.

    Linear warmup over ``milestones[0]`` epochs followed by
    ``CosineAnnealingWarmRestarts`` for the remaining epochs, composed via
    ``SequentialLR``.  Milestones < 1 are filtered out, matching the legacy
    behaviour.
    """

    milestones: tuple[int, ...]
    warmup_start_factor: float
    eta_min: float
    epochs: int

    def build(self, *, optimizer: Optimizer) -> LRScheduler:
        milestones = list(self.milestones)
        assert sum(milestones) != self.epochs, (
            "Sum of milestones must not equal total epochs"
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.warmup_start_factor,
            end_factor=1.0,
            total_iters=milestones[0],
            last_epoch=-1,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.epochs - milestones[0],
            T_mult=1,
            eta_min=self.eta_min,
            last_epoch=-1,
        )

        warmup_milestone = milestones[0]
        if warmup_milestone >= 1:
            schedulers = [warmup_scheduler, cosine_scheduler]
            milestones_local = [warmup_milestone]
        else:
            # Legacy behaviour: skip the warmup phase entirely.
            schedulers = [cosine_scheduler]
            milestones_local = []

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers,
            milestones_local,
            last_epoch=-1,
        )


@dataclass(kw_only=True)
class WarmupCosineAnnealingSpec(LRSchedulerSpec):
    """Linear warmup followed by cosine annealing decay to ``eta_min``.

    Differs from :class:`WarmupCosineSpec` in the second phase: it uses
    :class:`~torch.optim.lr_scheduler.CosineAnnealingLR` to decay the LR
    smoothly toward ``eta_min`` over the remaining epochs without warm restarts.
    """

    milestones: tuple[int, ...]
    warmup_start_factor: float
    eta_min: float
    epochs: int

    def build(self, *, optimizer: Optimizer) -> LRScheduler:
        milestones = list(self.milestones)
        assert sum(milestones) != self.epochs, (
            "Sum of milestones must not equal total epochs"
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=self.warmup_start_factor,
            end_factor=1.0,
            total_iters=milestones[0],
            last_epoch=-1,
        )

        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs - milestones[0],
            eta_min=self.eta_min,
            last_epoch=-1,
        )

        warmup_milestone = milestones[0]
        if warmup_milestone >= 1:
            schedulers = [warmup_scheduler, cosine_scheduler]
            milestones_local = [warmup_milestone]
        else:
            # Legacy behaviour: skip the warmup phase entirely.
            schedulers = [cosine_scheduler]
            milestones_local = []

        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers,
            milestones_local,
            last_epoch=-1,
        )

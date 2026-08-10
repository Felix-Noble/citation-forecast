from __future__ import annotations

from dataclasses import dataclass

from torch import nn
from torch.utils.data import DataLoader

from training.checkpointing import CheckpointProcessor
from training.strategies import Strategy
from training.tracking.base import MetricTracker


@dataclass(kw_only=True)
class Experiment[T_Batch]:
    """Fully material experiment configuration.

    Everything needed to reproduce a training/evaluation run lives here.
    Runtime concerns (device, dtype, compile flags, subsample) are injected by
    the caller and do not appear in this dataclass.
    """

    experiment_name: str
    model: nn.Module
    strategy: Strategy[T_Batch]
    tracker: MetricTracker
    train_loader: DataLoader[T_Batch]
    val_loader: DataLoader[T_Batch]
    checkpoints: CheckpointProcessor
    epochs: int
    eval_interval: int
    checkpoint_interval: int

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, InstanceOf
from torch import Tensor
from torch.cuda import StreamContext
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from training.optimizers.specs import OptimizerSpec
from training.schedulers import LRSchedulerSpec
from training.tracking.base import MetricTracker

T_Batch = TypeVar("T_Batch")


@runtime_checkable
class Strategy(Protocol[T_Batch]):
    """Lightning-compatible strategy contract (F1/F4).

    The engine drives training/validation by calling these hooks.  Strategies
    own optimizer/scheduler construction, device placement, the per-batch step,
    and any setup such as ``torch.set_float32_matmul_precision`` (J10).
    """

    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]: ...

    def start_epoch(self, *, epoch: int) -> None: ...

    def move_to_device(self, batch: T_Batch) -> T_Batch: ...

    def training_step(self, batch: T_Batch) -> float: ...

    def validation_step(self, batch: T_Batch) -> float: ...

    def scheduler_step(self) -> None: ...

    def load_optimizer_state(
        self,
        *,
        optimizer: Optimizer,
        scheduler: LRScheduler,
    ) -> None: ...


class StrategyConfig(BaseModel):
    """Keyword-only configuration shared by all strategies.

    Runtime concerns such as ``device`` and ``stream`` are injected by the
    caller; experiment-relevant values (optimizer/schedulers specs,
    ``mat_mul_precision``) are declared here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: InstanceOf[nn.Module]
    loss_fn: InstanceOf[nn.Module]
    tracker: InstanceOf[MetricTracker]
    optimizer_spec: OptimizerSpec
    scheduler_spec: LRSchedulerSpec
    stream: InstanceOf[StreamContext] | InstanceOf[nullcontext[None]]
    device: torch.device  # type: ignore[reportMissingTypeArgument]
    examples_per_epoch: int
    accumulation_steps: int = 1
    mat_mul_precision: str = "high"


class BaseStrategy(ABC, Generic[T_Batch]):
    """Shared strategy machinery: counters, device placement, accumulation,
    optimizer/scheduler construction, and resume helpers.
    """

    config = StrategyConfig

    def __init__(self, config: StrategyConfig) -> None:
        torch.set_float32_matmul_precision(config.mat_mul_precision)

        self.config = config
        self.model = config.model
        self.loss_fn = config.loss_fn
        self.tracker = config.tracker
        self.device = config.device
        self.stream = config.stream
        self.accumulation_steps = config.accumulation_steps
        self.examples_per_epoch = config.examples_per_epoch

        self.optimizer, self.scheduler = self.configure_optimizers()

        self.epoch_i: int | None = None
        self._batch_i: int | None = None
        self._batch_steps_i: int | None = None
        self._total_steps_i: int | None = None

        self.stream_sync = (
            torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        )

    def configure_optimizers(self) -> tuple[Optimizer, LRScheduler]:
        """Build optimizer + scheduler from the injected specs (F4/J2)."""
        optimizer = self.config.optimizer_spec.build(params=self.model.parameters())
        scheduler = self.config.scheduler_spec.build(optimizer=optimizer)
        return optimizer, scheduler

    @property
    def batch_i(self) -> int:
        assert self._batch_i is not None
        return self._batch_i

    def start_epoch(self, *, epoch: int) -> None:
        """Reset per-epoch counters."""
        self.epoch_i = epoch
        self._batch_i = 0
        self._batch_steps_i = 0
        self._total_steps_i = 0

    def _to_device(self, value: Any, device: torch.device) -> Any:
        """Move a tensor (or nested object implementing ``.to``) to ``device``."""
        if hasattr(value, "to") and callable(value.to):
            return value.to(device, non_blocking=True)
        return value

    @abstractmethod
    def move_to_device(self, batch: T_Batch) -> T_Batch: ...

    def scheduler_step(self) -> None:
        """Advance the learning-rate scheduler once per epoch."""
        self.scheduler.step()

    def load_optimizer_state(
        self,
        *,
        optimizer: Optimizer | Mapping[str, Any],
        scheduler: LRScheduler | Mapping[str, Any],
    ) -> None:
        """Restore optimizer/scheduler state during resume (J3).

        Accepts either live objects (for tests/interop) or raw state dicts
        (from :class:`Checkpoint`).
        """
        if isinstance(optimizer, Mapping):
            self.optimizer.load_state_dict(optimizer)
        else:
            self.optimizer.load_state_dict(optimizer.state_dict())

        if isinstance(scheduler, Mapping):
            self.scheduler.load_state_dict(scheduler)
        else:
            self.scheduler.load_state_dict(scheduler.state_dict())

from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import NamedTuple, Protocol

import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, ConfigDict, InstanceOf, PositiveInt
from torch import Tensor
from torch.cuda import StreamContext
from torch.optim.lr_scheduler import LRScheduler

from training.tracking import MetricTracker


class TrainerConfig(BaseModel):
    model: InstanceOf[nn.Module]
    optimizer: InstanceOf[optim.Optimizer]
    loss_fn: InstanceOf[nn.Module]
    scheduler: InstanceOf[LRScheduler]
    tracker: InstanceOf[MetricTracker]
    stream: InstanceOf[StreamContext] | InstanceOf[nullcontext[None]]
    device: torch.device  # type: ignore
    accumulation_steps: PositiveInt
    examples_per_epoch: PositiveInt
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    mask: Tensor


class TrainerProtocol[T_Config, T_Batch](Protocol):
    config: type[T_Config]

    def start_epoch(self, epoch: int) -> None:
        "Start epoch, reset counters"
        ...

    def move_to_device(self, batch: T_Batch) -> T_Batch:
        "Move batch to device"
        ...

    def _step(self, batch: T_Batch) -> float:
        "One training step, forward, backward pass"
        ...

    def step(self, batch: T_Batch) -> float:
        "Call child specific _step function, increment counters"
        ...


class BaseTrainer(ABC):
    config = TrainerConfig

    def __init__(self, config: TrainerConfig):
        self.model = config.model
        self.optimizer = config.optimizer
        self.loss_fn = config.loss_fn
        self.scheduler = config.scheduler
        self.tracker = config.tracker
        self.stream = config.stream
        self.device = config.device
        self.accumulation_steps = config.accumulation_steps
        self.examples_per_epoch = config.examples_per_epoch
        self.epoch_i: int | None = None
        self._batch_i: int | None = None
        self._batch_steps_i: int | None = None
        self._total_steps_i: int | None = None
        self.stream_sync = (
            torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        )

    @property
    def batch_i(self) -> int:
        assert self._batch_i is not None
        return self._batch_i

    @batch_i.setter
    def batch_i(self, value: int) -> None:
        self._batch_i = value

    @property
    def batch_steps_i(self) -> int:
        assert self._batch_steps_i is not None
        return self._batch_steps_i

    @batch_steps_i.setter
    def batch_steps_i(self, value: int) -> None:
        self._batch_steps_i = value

    @property
    def total_steps_i(self) -> int:
        assert self._total_steps_i is not None
        return self._total_steps_i

    @total_steps_i.setter
    def total_steps_i(self, value: int) -> None:
        self._total_steps_i = value

    def start_epoch(self, epoch: int) -> None:
        self.epoch_i = epoch
        self._batch_i = 0
        self._batch_steps_i = 0
        self._total_steps_i = 0

    @abstractmethod
    def move_to_device(self, batch: Batch) -> Batch:
        "Move batch to device"
        pass

    @abstractmethod
    def _step(self, batch: Batch) -> float:
        "One training step, forward, backward pass"
        pass

    def step(self, batch: Batch) -> float:
        loss = self._step(batch)
        # self.tracker.process_loss(loss)
        self.batch_i += 1
        self.batch_steps_i += batch.x.shape[0]
        self.total_steps_i += batch.x.shape[0]
        return loss

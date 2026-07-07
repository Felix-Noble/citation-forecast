from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import NamedTuple, Protocol

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, InstanceOf, PositiveInt
from torch import Tensor
from torch.cuda import StreamContext

from training.tracking import MetricTracker


class EvaluatorConfig(BaseModel):
    model: InstanceOf[nn.Module]
    prefix: str
    loss_fn: InstanceOf[nn.Module]
    tracker: InstanceOf[MetricTracker]
    stream: InstanceOf[StreamContext] | InstanceOf[nullcontext[None]]
    device: torch.device  # type: ignore
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Batch(NamedTuple):
    id: Tensor
    x: Tensor
    y: Tensor
    mask: Tensor


class EvaluatorProtocol[T_Config, T_Batch](Protocol):
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


class BaseEvaluator(ABC):
    config = EvaluatorConfig

    def __init__(self, config: EvaluatorConfig):
        self.model = config.model
        self.prefix = config.prefix
        self.loss_fn = config.loss_fn
        self.tracker = config.tracker
        self.stream = config.stream
        self.device = config.device
        self.stream_sync = (
            torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
        )

    @abstractmethod
    def move_to_device(self, batch: Batch) -> Batch:
        "Move batch to device"
        pass

    @abstractmethod
    def _step(self, batch: Batch) -> float:
        "One training step, forward, backward pass"
        pass

    def step(self, batch: Batch) -> float:
        self.model.eval()
        loss = self._step(batch)
        return loss

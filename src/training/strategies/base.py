from typing import Protocol, TypeVar, runtime_checkable

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

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

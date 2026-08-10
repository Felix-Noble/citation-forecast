from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

import torch
from torch import Tensor


@runtime_checkable
class MetricTracker(Protocol):
    """Tracker contract (F5/J4).

    Implementations use dict-backed CPU stores, declare their ``store_names``,
    and own all metric logging including batch-level ``train_loss-batch`` (F3).
    No loose ``config`` object is passed in; every setting is an explicit
    constructor kwarg.
    """

    store_names: ClassVar[tuple[str, ...]]

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        export: bool = False,
        export_loc: Path | None = None,
    ) -> None: ...

    def process_values(
        self,
        values: tuple[Tensor, ...],
        store_names: tuple[str, ...],
    ) -> None: ...

    def log_batch_metric(
        self,
        name: str,
        value: float,
        *,
        step: int,
    ) -> None: ...

    def calc_metrics(self, *, prefix: str, step: int) -> None: ...

    def report(
        self,
        progress_bar: object,
        *,
        epoch: int | None = None,
    ) -> dict[str, float]: ...

    def clear(self) -> None: ...

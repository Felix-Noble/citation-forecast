from pathlib import Path
from typing import Any, NamedTuple, Protocol, runtime_checkable

import torch
from torch import Tensor


class Checkpoint(NamedTuple):
    """Dict-style checkpoint (E2).  No backwards compatibility with old
    model-only checkpoints (I3)."""

    model: dict[str, Tensor]
    optimizer: dict[str, Any] | None
    scheduler: dict[str, Any] | None
    epoch: int


class CheckpointRef(NamedTuple):
    run_id: str
    epoch: int


@runtime_checkable
class CheckpointProcessor(Protocol):
    """Persistence strategy for :class:`Checkpoint` dicts."""

    def save(self, *, state: Checkpoint, run_id: str) -> str: ...

    def load(self, *, ref: CheckpointRef, map_location: torch.device) -> Checkpoint: ...


class ExperimentFileStore(Protocol):
    """Mixin-like protocol for processors that also handle experiment-file
    artifacts (B4/E5/J1)."""

    def save_experiment_file(self, *, path: Path) -> None: ...

    def experiment_file_exists(self, *, run_id: str) -> bool: ...

    def download_experiment_file(self, *, run_id: str, dest: Path) -> Path: ...

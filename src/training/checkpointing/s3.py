from dataclasses import dataclass
from pathlib import Path
from typing import Never

from .base import Checkpoint, CheckpointProcessor, CheckpointRef


@dataclass(kw_only=True)
class S3CheckpointProcessor(CheckpointProcessor):
    """S3 checkpoint processor — protocol stub only (E4).

    A real implementation needs boto3 credentials/endpoint handling, which is
    deferred to plan 1.0.1 (F-6).
    """

    bucket: str
    prefix: str = "checkpoints"

    def save(self, *, state: Checkpoint, run_id: str) -> str:
        raise NotImplementedError("S3CheckpointProcessor.save is a stub")

    def load(self, *, ref: CheckpointRef, map_location: torch.device) -> Checkpoint:
        raise NotImplementedError("S3CheckpointProcessor.load is a stub")

    def save_experiment_file(self, *, path: Path) -> Never:
        raise NotImplementedError("S3CheckpointProcessor.save_experiment_file is a stub")

    def experiment_file_exists(self, *, run_id: str) -> bool:
        raise NotImplementedError("S3CheckpointProcessor.experiment_file_exists is a stub")

    def download_experiment_file(self, *, run_id: str, dest: Path) -> Path:
        raise NotImplementedError("S3CheckpointProcessor.download_experiment_file is a stub")

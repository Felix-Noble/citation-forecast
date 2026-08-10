import os
import tempfile
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

import mlflow
import torch
from mlflow.tracking import MlflowClient

from utils.logging import setup_logger

from .base import Checkpoint, CheckpointProcessor, CheckpointRef, ExperimentFileStore

logger = getLogger(__name__)
_ = setup_logger(logger)


@dataclass(kw_only=True)
class MlflowCheckpointProcessor(CheckpointProcessor, ExperimentFileStore):
    """MLflow-backed checkpoint processor.

    Saves/loads dict checkpoints via MLflow artifacts.  Also owns experiment-file
    artifact I/O (B4/E5/J1) and fails fast with ``experiment_file_exists`` using
    ``client.list_artifacts`` (J1).
    """

    artifact_loc: Path
    tracking_uri: str
    experiment_name: str
    include_optimizer: bool = True
    include_scheduler: bool = True

    _client: MlflowClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)
        self._client = MlflowClient()

    def _local_path(self, run_id: str, epoch: int) -> Path:
        return self.artifact_loc / run_id / f"epoch-{epoch}.pt"

    def _artifact_path(self, epoch: int) -> str:
        return f"epoch-{epoch}.pt"

    def _experiment_file_artifact_path(self) -> str:
        return "experiment.py"

    def save(self, *, state: Checkpoint, run_id: str) -> str:
        slim_state = {
            "model": state.model,
            "epoch": state.epoch,
        }
        if self.include_optimizer:
            slim_state["optimizer"] = state.optimizer
        if self.include_scheduler:
            slim_state["scheduler"] = state.scheduler

        path = self._local_path(run_id, state.epoch)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f"epoch-{state.epoch}-",
            suffix=".pt.tmp",
        )
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                torch.save(slim_state, tmp_file)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        mlflow.log_artifact(str(path), artifact_path="", run_id=run_id)
        logger.info(f"Checkpoint saved to MLflow run {run_id}: {path.name}")
        return str(path)

    def load(self, *, ref: CheckpointRef, map_location: torch.device) -> Checkpoint:
        path = self._local_path(ref.run_id, ref.epoch)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_artifacts(
                ref.run_id,
                self._artifact_path(ref.epoch),
                str(path.parent),
            )

        raw = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(raw, dict):
            raise TypeError(f"Checkpoint at {path} is not a dict")
        return Checkpoint(
            model=raw["model"],
            optimizer=raw.get("optimizer"),
            scheduler=raw.get("scheduler"),
            epoch=raw["epoch"],
        )

    def save_experiment_file(self, *, path: Path) -> None:
        mlflow.log_artifact(str(path), artifact_path="", run_id=None)

    def experiment_file_exists(self, *, run_id: str) -> bool:
        artifacts = self._client.list_artifacts(run_id, path="")
        return any(
            a.path == self._experiment_file_artifact_path() for a in artifacts
        )

    def download_experiment_file(self, *, run_id: str, dest: Path) -> Path:
        dest.mkdir(parents=True, exist_ok=True)
        self._client.download_artifacts(
            run_id,
            self._experiment_file_artifact_path(),
            str(dest),
        )
        return dest / self._experiment_file_artifact_path()

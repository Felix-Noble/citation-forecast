import os
import tempfile
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

import torch

from utils.logging import setup_logger

from .base import Checkpoint, CheckpointProcessor, CheckpointRef

logger = getLogger(__name__)
_ = setup_logger(logger)


@dataclass(kw_only=True)
class LocalCheckpointProcessor(CheckpointProcessor):
    """Atomic local-filesystem checkpoint processor.

    Writes to a temporary file in the target directory and renames it into
    place so readers never see a partial checkpoint.
    """

    artifact_loc: Path

    def _path(self, run_id: str, epoch: int) -> Path:
        return self.artifact_loc / run_id / f"epoch-{epoch}.pt"

    def save(self, *, state: Checkpoint, run_id: str) -> str:
        path = self._path(run_id, state.epoch)
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=path.parent,
            prefix=f"epoch-{state.epoch}-",
            suffix=".pt.tmp",
        )
        try:
            with os.fdopen(tmp_fd, "wb") as tmp_file:
                torch.save(state._asdict(), tmp_file)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(f"Checkpoint saved to {path}")
        return str(path)

    def load(self, *, ref: CheckpointRef, map_location: torch.device) -> Checkpoint:
        path = self._path(ref.run_id, ref.epoch)
        raw = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(raw, dict):
            raise TypeError(f"Checkpoint at {path} is not a dict")
        return Checkpoint(
            model=raw["model"],
            optimizer=raw.get("optimizer"),
            scheduler=raw.get("scheduler"),
            epoch=raw["epoch"],
        )

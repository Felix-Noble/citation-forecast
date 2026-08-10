from dataclasses import dataclass
from pathlib import Path

from .base import DataSource


@dataclass(frozen=True, kw_only=True)
class LocalStagedSource:
    """Local filesystem source whose resolved path is ``path / name``.

    This preserves the old ``env.STAGED_LOC / config.loc`` layout while making
    the source self-contained.
    """

    path: Path
    name: str

    def resolve(self) -> Path:
        return self.path / self.name


DataSource.register(LocalStagedSource)

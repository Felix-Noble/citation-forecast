from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class DataSource(Protocol):
    """Thin path resolver for a dataset backing store.

    The dataset receives a ``DataSource`` and reads/caches from
    :meth:`resolve`.  Sources own their full path (J5); there is no shared
    env-root indirection.
    """

    @property
    def name(self) -> str: ...

    def resolve(self) -> Path: ...

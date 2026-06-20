from pathlib import Path

from utils import import_all_local

import_all_local(pkg_path=Path(__file__).parent, package="training.optimizers")

__all__ = ["optim_registry"]

from ._registry import optim_registry  # noqa E402

from pathlib import Path

from utils import import_all_local

from .entropy import norm_entropy_loss

import_all_local(pkg_path=Path(__file__).parent, package="training.losses")
__all__ = ["loss_registry", "norm_entropy_loss"]

from ._registry import loss_registry  # noqa E402

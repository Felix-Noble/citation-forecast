from .entropy import norm_entropy_loss
from src.utils import import_all_local
from pathlib import Path

import_all_local(
        pkg_path=Path(__file__).parent,
        package='src.training.losses'
                 )
__all__ = [
        'loss_registry',
        'norm_entropy_loss'
        ]

from ._registry import loss_registry #noqa E402

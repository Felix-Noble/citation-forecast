from src.utils import import_all_local
from pathlib import Path

import_all_local(
        pkg_path=Path(__file__).parent,
        package='src.training.optimizers'
                 )

__all__ = [
        'optim_registry'
        ]

from ._registry import optim_registry #noqa E402

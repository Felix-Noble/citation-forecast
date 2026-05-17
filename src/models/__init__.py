from src.utils import import_all_local
from pathlib import Path

import_all_local(
        pkg_path=Path(__file__).parent,
        package='src.models'
                 )
__all__ = [ 'model_registry' ]

from ._registry import model_registry #noqa E402

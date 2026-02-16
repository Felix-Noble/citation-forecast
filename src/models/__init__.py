import pkgutil
import importlib
from pathlib import Path

_pkg_path = Path(__file__).parent
for _, module_name, _ in pkgutil.iter_modules([_pkg_path]):
    if module_name not in ('_registry', '__init__'):
        importlib.import_module('.' + module_name, package='src.models')

__all__ = [ 'model_registry' ]

from ._registry import model_registry #noqa E402

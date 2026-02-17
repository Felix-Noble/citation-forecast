import pkgutil
import importlib
from pathlib import Path

def import_all_local(
        pkg_path: Path,
        package: str,
        ignore: tuple[str, ...] = ('_registry', '__init__')
                     ) -> None:

    for _, module_name, _ in pkgutil.iter_modules([pkg_path]):
        if module_name not in ignore: 
            importlib.import_module('.' + module_name, package=package)

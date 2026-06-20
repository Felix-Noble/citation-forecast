import os
from pathlib import Path


def get_root_dir(
    file: str = __file__,
    markers: tuple[str, ...] = ("pyproject.toml",),
) -> Path:
    pwd = Path(file).resolve()
    if pwd.is_file():
        pwd = pwd.parent

    for parent in [pwd] + list(pwd.parents):
        if any((parent / marker).exists() for marker in markers):
            return parent

    return pwd

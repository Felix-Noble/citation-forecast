"""Manual CLI that regenerates package ``__init__.py`` import blocks.

Usage::

    python -m utils.build_helper

Scans the component packages declared in ``COMPONENT_PACKAGES`` for classes
decorated with ``@component`` and rewrites the auto-generated block between
``# >>> build_helper:auto (do not edit)`` and ``# <<< build_helper:auto``
markers in each package's ``__init__.py``.

Run this after adding, removing, or renaming a component class; the working tree
should be empty of unintended diffs afterwards.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

COMPONENT_PACKAGES: Final[tuple[str, ...]] = (
    "models",
    "data/datasets",
    "training/losses",
    "training/optimizers",
)

AUTO_START: Final[str] = "# >>> build_helper:auto (do not edit)"
AUTO_END: Final[str] = "# <<< build_helper:auto"


def _iter_component_classes(pkg_path: Path) -> list[tuple[str, str]]:
    """Return ``(module_name, class_name)`` for ``@component`` classes."""
    components: list[tuple[str, str]] = []
    for file in sorted(pkg_path.glob("*.py")):
        if file.name.startswith("_"):
            continue
        module_name = file.stem
        source = file.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for decorator in node.decorator_list:
                name = _component_decorator_name(decorator)
                if name is not None:
                    components.append((module_name, node.name))
                    break
    return components


def _component_decorator_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name) and decorator.id == "component":
        return "component"
    if isinstance(decorator, ast.Call):
        func = decorator.func
        if isinstance(func, ast.Name) and func.id == "component":
            return "component"
    return None


def _build_block(components: list[tuple[str, str]]) -> str:
    imports = "\n".join(
        f"from .{module} import {cls}" for module, cls in components
    )
    all_entries = ",\n    ".join(f'"{cls}"' for _, cls in components)
    return (
        f"{AUTO_START}\n"
        f"{imports}\n\n"
        f"__all__ = [\n"
        f"    {all_entries},\n"
        f"]\n"
        f"{AUTO_END}"
    )


def regenerate_init(pkg_path: Path) -> None:
    """Rewrite the auto block in ``pkg_path/__init__.py``."""
    init_file = pkg_path / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

    content = init_file.read_text()
    start_idx = content.find(AUTO_START)
    end_idx = content.find(AUTO_END)

    if start_idx == -1 or end_idx == -1:
        print(f"Skipping {pkg_path}: auto markers not found")
        return

    components = _iter_component_classes(pkg_path)
    block = _build_block(components)

    new_content = content[:start_idx] + block + content[end_idx + len(AUTO_END) :]
    init_file.write_text(new_content)
    print(f"Updated {init_file}: {len(components)} component(s)")


def main() -> None:
    src = Path(__file__).resolve().parent.parent
    for pkg in COMPONENT_PACKAGES:
        regenerate_init(src / pkg)


if __name__ == "__main__":
    main()

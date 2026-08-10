"""Marker decorator for build-helper-discoverable components.

The old runtime ``Registry`` class has been removed (plan 1.0 v3 phase 7).
``@component`` only tags a class as exportable; ``utils.build_helper`` scans
packages for the marker and regenerates their ``__init__.py`` import blocks.
"""

from collections.abc import Callable
from typing import TypeVar, overload

T = TypeVar("T", bound=type)


@overload
def component(cls: T) -> T: ...


@overload
def component(group: str | None = None) -> Callable[[T], T]: ...


def component(
    cls_or_group: T | str | None = None,
) -> T | Callable[[T], T]:
    """Tag a class as a build-helper-discoverable component.

    Used without arguments on the class itself, or with an optional group name.
    Sets ``cls.__component_group__`` for downstream tooling.
    """

    if isinstance(cls_or_group, type):
        cls = cls_or_group
        cls.__component_group__ = cls.__module__.split(".")[-2]
        return cls

    group = cls_or_group

    def decorator(cls: T) -> T:
        cls.__component_group__ = (
            group if group is not None else cls.__module__.split(".")[-2]
        )
        return cls

    return decorator

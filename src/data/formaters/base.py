from typing import Protocol, TypeVar, runtime_checkable

T_In = TypeVar("T_In")
T_Out = TypeVar("T_Out")


@runtime_checkable
class Formater(Protocol[T_In, T_Out]):
    """Per-row value transform (C3).

    A ``Formater`` does **not** own x-column selection; it only formats the
    values that the dataset has already chosen to expose.
    """

    def __call__(self, row: T_In) -> T_Out: ...

from typing import Protocol


class LossFn[T_Batch, T_Output](Protocol):
    def forward(self, batch: T_Batch, output: T_Output) -> float: ...

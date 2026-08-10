from logging import getLogger
from pathlib import Path
from typing import NamedTuple, override

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from utils import component
from utils.logging import setup_logger

from .polars_dataset import PolarsDataset, PolarsDatasetConfig

logger = getLogger(__name__)
_ = setup_logger(logger)


class OrdinalDatasetConfig(PolarsDatasetConfig):
    boundaries: Tensor
    min: float
    max: float


class OrdinalDatasetOutput(NamedTuple):
    id: Tensor
    x: Tensor
    y: Tensor
    y_orig: Tensor
    mask: Tensor
    weight: Tensor


@component
class OrdinalDataset(PolarsDataset):
    config: type[OrdinalDatasetConfig] = OrdinalDatasetConfig

    def __init__(self, config: OrdinalDatasetConfig, env):
        super().__init__(config, env)
        self.min = config.min
        self.max = config.max
        self.boundaries = config.boundaries

    @override
    def _format_y(self, y: Tensor) -> Tensor:
        ordinal = torch.bucketize(y, self.boundaries, right=True, out_int32=True)
        # logger.debug(print("log", log, "ord", ordinal, "\n", "bound", self.boundaries)
        return ordinal.long()

    @override
    def __getitem__(self, idx: int) -> OrdinalDatasetOutput:
        id = torch.tensor(float("nan"))
        x = torch.tensor(float("nan"))
        y = torch.tensor(float("nan"))
        y_orig = torch.tensor(float("nan"))
        mask = torch.tensor(float("nan"))
        weight = torch.tensor(float("nan"))
        # X (input)
        x_row: tuple[list[int], ...] = self.df_x.row(idx)
        x: Tensor = torch.cat(
            [torch.tensor(token_list) for token_list in x_row]
        ).flatten()

        x = self._format_x(x)
        if self.pad and x.size(0) < self.max_len:
            x = nn.functional.pad(
                x, (0, self.max_len - x.size(0)), value=self.pad_value
            )
        if self.truncate == True & x.size(0) > self.max_len:
            x = x[: self.max_len]

        # y (target)
        if self.y:
            y_row: tuple[list[int], ...] = self.df_y.row(idx)
            y_orig: Tensor = torch.stack(
                [torch.tensor(target, dtype=torch.float32) for target in y_row]
            ).flatten()
            y = self._format_y(y_orig)

        # id (tracking)
        if self.return_id:
            id = torch.tensor(self.df_id.row(idx))

        # weights (per target class)
        if self.weights is not None:
            weight: Tensor = torch.tensor(
                [
                    torch.tensor(self.weights[target.long()], dtype=torch.float32)
                    for target in torch.atleast_1d(y)
                ]
            ).flatten()

        if self.return_mask:
            mask = (x != self.pad_value).bool().unsqueeze(0).expand(self.max_len, -1)

        out = OrdinalDatasetOutput(
            id=id, x=x, y=y, y_orig=y_orig, mask=mask, weight=weight
        )
        return out

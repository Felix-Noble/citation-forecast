import logging
import random
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import override

import torch
import torch.nn as nn
from torch import Tensor, tensor
from torch.utils.data import Dataset

from utils.logging import setup_logger

from ._registry import dataset_registry
from .polars_dataset import PolarsDataset

logger = getLogger(__name__)
_ = setup_logger(logger)


@dataclass
class Output:
    id: Tensor = tensor(float("nan"))
    x: Tensor = tensor(float("nan"))
    y: Tensor = tensor(float("nan"))
    weight: Tensor = tensor(float("nan"))
    mask: Tensor = tensor(float("nan"))


@dataset_registry
class GenerativePretrainDataset2(PolarsDataset):
    "Splits token column into two segments, input and predictions (non causal pre-training split)"

    def __init__(self, n_forward: int, n_backward: int, **kwargs):
        super().__init__(**kwargs)
        assert self.x == self.y or not self.y, "Same column expected for input & target"
        self.n_forward: int = n_forward  # n. tokens to predict forward
        self.n_backward: int = (
            n_backward  # n. tokens to 'predict' backwards from end of seq
        )

    def _gen_offset(self, length: int):
        upper = max(length - self.n_forward - 1, 0)
        return random.randint(0, upper)

    @override
    def __getitem__(self, idx: int):
        out: Output = Output()

        # X (input)
        x_row: tuple[list[int], ...] = self.df_x.row(idx)
        x: Tensor = torch.cat(
            [torch.tensor(token_list) for token_list in x_row]
        ).flatten()

        length = x.shape[0]
        offset = self._gen_offset(length)
        truncate_offset = max(offset - self.max_len, 0)

        x = x[truncate_offset:offset]

        # y (target)
        y_row: tuple[list[int], ...] = self.df_y.row(idx)
        y: Tensor = torch.cat(
            [torch.tensor(target, dtype=torch.float32) for target in y_row]
        ).flatten()

        y = y[offset - self.n_backward : offset + self.n_forward].long()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"offset: {offset} | truncate offset: {truncate_offset} | max_len: {self.max_len} | pad: {self.PAD}"
            )
            logger.debug(f"x: {x.shape}")
            logger.debug(x)
            logger.debug(f"y: {y.shape}")
            logger.debug(y)

        if self.PAD and x.size(0) < self.max_len:
            x = nn.functional.pad(
                x, (self.max_len - x.size(0), 0), value=self.PAD_VALUE
            )
        if self.PAD and y.size(0) < self.n_forward + self.n_backward:
            y = nn.functional.pad(
                y,
                (0, self.n_forward + self.n_backward - y.size(0)),
                value=self.PAD_VALUE,
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("x-pad")
            logger.debug(x.shape)
            logger.debug(x)
            logger.debug("y-pad")
            logger.debug(y.shape)
            logger.debug(y)

        out.y = y
        out.x = x

        # id (tracking)
        if self.return_id:
            id = torch.tensor(self.df_id.row(idx))
            out.id = id

        # weights (per target class)
        if self.weights is not None:
            weight: Tensor = torch.tensor(
                [
                    torch.tensor(self.weights[target.long()], dtype=torch.float32)
                    for target in torch.atleast_1d(y)
                ]
            ).flatten()
            out.weight = weight

        if self.RETURN_MASK:
            mask = (x != self.PAD_VALUE).bool().unsqueeze(0).expand(self.max_len, -1)
            out.mask = mask
        return out.__dict__

from logging import getLogger
from pathlib import Path
from typing import override

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from utils.logging import setup_logger

from ._registry import dataset_registry
from .polars_dataset import PolarsDataset

logger = getLogger(__name__)
_ = setup_logger(logger)


@dataset_registry
class BinaryCategoricalDataset(PolarsDataset):
    @override
    def _format_y(self, y: Tensor) -> Tensor:
        if y == 0:
            return torch.tensor(0, dtype=torch.float32)
        else:
            return torch.tensor(1, dtype=torch.float32)

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
class OrdinalDataset(PolarsDataset):
    @override
    def _format_y(self, y: Tensor) -> Tensor:
        ordinal = torch.round(y * (self.N_BUCKETS - 1), decimals=0).long()
        one_hot = nn.functional.one_hot(ordinal, num_classes=self.N_BUCKETS)
        return one_hot

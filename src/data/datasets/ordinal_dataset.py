from typing import override
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import Dataset

from config import Config, config
from src.utils.logging import setup_logger
from .polars_dataset import PolarsDataset
from pathlib import Path
from logging import getLogger
logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

class OrdinalDataset(PolarsDataset):
    @override
    def _format_y(self, y: Tensor) -> Tensor:
        ordinal = torch.round(y * (self.N_BUCKETS - 1), decimals=0).long()
        one_hot = nn.functional.one_hot(ordinal, num_classes=self.N_BUCKETS)
        return one_hot 


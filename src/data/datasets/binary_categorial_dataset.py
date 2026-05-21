from ._registry import dataset_registry
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

@dataset_registry
class BinaryCategoricalDataset(PolarsDataset):
    @override
    def _format_y(self, y: Tensor) -> Tensor:
        if y == 0:
            return torch.tensor(0, dtype=torch.float32)
        else:
            return torch.tensor(1, dtype=torch.float32)

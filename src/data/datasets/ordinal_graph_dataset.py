from logging import getLogger
from pathlib import Path
from typing import NamedTuple, override

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset

from utils.logging import setup_logger

from ._registry import dataset_registry
from .graph_dataset import GraphDataset, GraphDatasetConfig

logger = getLogger(__name__)
_ = setup_logger(logger)


class OrdinalDatasetConfig(GraphDatasetConfig):
    boundaries: Tensor


class OrdinalDatasetOutput(NamedTuple):
    id: Tensor
    x: Tensor
    y: Tensor
    y_orig: Tensor
    mask: Tensor
    weight: Tensor


@dataset_registry
class OrdinalGraphDataset(GraphDataset):
    config: type[OrdinalDatasetConfig] = OrdinalDatasetConfig

    def __init__(self, config: OrdinalDatasetConfig, env):
        super().__init__(config, env)
        self.boundaries = config.boundaries

    @override
    def _format_y(self, y: Tensor) -> Tensor:
        ordinal = torch.bucketize(y, self.boundaries, right=True, out_int32=True)
        # logger.debug(print("log", log, "ord", ordinal, "\n", "bound", self.boundaries)
        return ordinal.long()

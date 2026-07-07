from logging import getLogger
from typing import NamedTuple, override

import torch
from torch import Tensor

from utils.logging import setup_logger

from ._registry import dataset_registry
from .graph_dataset import GraphDataset, GraphDatasetConfig

logger = getLogger(__name__)
_ = setup_logger(logger)


class BinaryThresholdDatasetConfig(GraphDatasetConfig):
    theta: float


class BinaryThresholdDatasetOutput(NamedTuple):
    id: Tensor
    x: Tensor
    y: Tensor
    y_orig: Tensor
    mask: Tensor
    weight: Tensor


@dataset_registry
class BinaryThresholdGraphDataset(GraphDataset):
    config: type[BinaryThresholdDatasetConfig] = BinaryThresholdDatasetConfig

    def __init__(
        self,
        config: BinaryThresholdDatasetConfig,
        env,
    ):
        super().__init__(config, env)
        self.theta = config.theta

    @override
    def _format_y(self, y: Tensor) -> Tensor:
        return torch.tensor(y > self.theta, dtype=torch.float32)

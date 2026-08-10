from logging import getLogger
from typing import override

import torch
from torch import Tensor

from utils import component
from utils.logging import setup_logger

from .polars_dataset import Env, PolarsDataset, PolarsDatasetConfig

logger = getLogger(__name__)
_ = setup_logger(logger)


class BinaryThresholdDatasetConfig(PolarsDatasetConfig):
    theta: float


@component
class BinaryThresholdDataset(PolarsDataset):
    "Binary dataset with custom inclusive threshold (theta) for boundary between 0/1 class"

    config: type[BinaryThresholdDatasetConfig] = BinaryThresholdDatasetConfig

    def __init__(
        self,
        config: BinaryThresholdDatasetConfig,
        env: Env,
    ):
        super().__init__(config, env)
        self.theta = config.theta

    @override
    def _format_y(self, y: Tensor) -> Tensor:
        return torch.tensor(y > self.theta, dtype=torch.float32)

from logging import getLogger
from typing import override

import torch
from torch import Tensor

from utils import component
from utils.logging import setup_logger

from .polars_dataset import Env, PolarsDataset, PolarsDatasetConfig

logger = getLogger(__name__)
_ = setup_logger(logger)


class LogRegressDatasetConfig(PolarsDatasetConfig):
    pass

@component
class LogRegressDataset(PolarsDataset):
    "Binary dataset with custom inclusive threshold (theta) for boundary between 0/1 class"

    config: type[LogRegressDatasetConfig] = LogRegressDatasetConfig

    def __init__(
        self,
        config: LogRegressDatasetConfig,
        env: Env,
    ):
        super().__init__(config, env)

    @override
    def _format_y(self, y: Tensor) -> Tensor:
        return torch.tensor(torch.log(y + 1), dtype=torch.float32)

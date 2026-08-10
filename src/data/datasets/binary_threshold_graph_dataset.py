from collections.abc import Mapping
from logging import getLogger
from typing import Any, override

import torch
from torch import Tensor

from data.formaters import Formater, GraphFormater
from data.sources import DataSource
from utils.logging import setup_logger

from ._registry import dataset_registry
from .graph_dataset import CitationGraphDatasetConfig, GraphDataset
from .types import CitationGraphDatasetOutput

logger = getLogger(__name__)
_ = setup_logger(logger)


class BinaryThresholdDatasetConfig(CitationGraphDatasetConfig):
    theta: float


@dataset_registry
class BinaryThresholdGraphDataset(GraphDataset):
    config: type[BinaryThresholdDatasetConfig] = BinaryThresholdDatasetConfig

    def __init__(
        self,
        *,
        config: BinaryThresholdDatasetConfig,
        source: DataSource,
        formater: Formater[Mapping[str, Any], CitationGraphDatasetOutput] | None = None,
    ):
        if formater is None:
            formater = GraphFormater(
                max_len=config.max_len,
                graph_max_len=config.graph_max_len,
                top_k=config.top_k,
                pad=config.pad,
                truncate=config.truncate,
                truncate_method=config.truncate_method,
                return_mask=config.return_mask,
                return_id=config.return_id,
                weights=config.weights,
                format_y=lambda y: torch.tensor(y > config.theta, dtype=torch.float32),
            )
        super().__init__(config=config, source=source, formater=formater)
        self.theta = config.theta

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


class OrdinalDatasetConfig(CitationGraphDatasetConfig):
    boundaries: Tensor


@dataset_registry
class OrdinalGraphDataset(GraphDataset):
    config: type[OrdinalDatasetConfig] = OrdinalDatasetConfig

    def __init__(
        self,
        *,
        config: OrdinalDatasetConfig,
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
                format_y=lambda y: torch.bucketize(
                    y, config.boundaries, right=True, out_int32=True
                ).long(),
            )
        super().__init__(config=config, source=source, formater=formater)
        self.boundaries = config.boundaries

from collections.abc import Mapping
from logging import getLogger
from typing import Any, override

import torch
from torch import Tensor

from data.formaters import CitationGraphFormater, Formater
from data.sources import DataSource
from utils.logging import setup_logger

from ._registry import dataset_registry
from .citation_graph_dataset import CitationGraphDataset, CitationGraphDatasetConfig
from .types import CitationGraphDatasetOutput

logger = getLogger(__name__)
_ = setup_logger(logger)


class OrdinalDatasetConfig(CitationGraphDatasetConfig):
    boundaries: Tensor
    min: float
    max: float


@dataset_registry
class OrdinalCiteGraphDataset(CitationGraphDataset):
    config: type[OrdinalDatasetConfig] = OrdinalDatasetConfig

    def __init__(
        self,
        *,
        config: OrdinalDatasetConfig,
        source: DataSource,
        formater: Formater[Mapping[str, Any], CitationGraphDatasetOutput] | None = None,
    ):
        if formater is None:
            formater = CitationGraphFormater(
                max_len=config.max_len,
                graph_max_len=config.graph_max_len,
                pad_token_id=config.pad_token_id,
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
        self.min = config.min
        self.max = config.max
        self.boundaries = config.boundaries

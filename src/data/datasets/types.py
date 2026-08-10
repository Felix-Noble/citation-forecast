from typing import NamedTuple

from torch import Tensor


class CitationGraphDatasetOutput(NamedTuple):
    id: Tensor
    x: Tensor
    graph_x: Tensor
    y: Tensor
    weight: Tensor
    mask: Tensor
    graph_x_mask: Tensor

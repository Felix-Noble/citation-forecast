from .df_dataset import DF_Dataset
import torch
from torch import Tensor
from typing import override

class BinaryDataset(DF_Dataset):
    @override
    def _format_x(self, x: Tensor) -> Tensor:
        return x.long() 

    @override
    def _format_y(self, y: Tensor) -> Tensor:
        return torch.round(y, decimals=0)

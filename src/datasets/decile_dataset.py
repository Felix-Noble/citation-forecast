from .df_dataset import DF_Dataset
import torch
from torch import Tensor
from typing import override

class DecileDataset(DF_Dataset):
    @override
    def _format_y(self, y: Tensor) -> Tensor:
        return torch.round(y*10, decimals=0)


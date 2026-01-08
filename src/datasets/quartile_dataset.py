from .df_dataset import DF_Dataset
import torch
from torch import Tensor
from typing import override

class QuartileDataset(DF_Dataset):
    @override
    def _format_y(self, y: Tensor) -> Tensor:
        return torch.round(y*4, decimals=0)

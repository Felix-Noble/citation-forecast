from ._registry import dataset_registry
from typing import override
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import Dataset
import random

from config import Config, config
from src.utils.logging import setup_logger
from .polars_dataset import PolarsDataset
from pathlib import Path
from logging import getLogger

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

@dataset_registry
class GenerativePretrainDataset(PolarsDataset):
    " Splits token column into two segments, input and predictions (non causal pre-training split) "
    def __init__(self, n_forward: int, **kwargs):
        super().__init__(**kwargs)
        assert self.x == self.y or not self.y, 'Same column expected for input & target'
        self.n_forward = n_forward # n. tokens to predict forward
        self.processed: list[str] = [] # x/y output formatted
        self.offset: int = self._new_offset() # randomised offset to move token sequence around
    
    def _new_offset(self, min:int=0, max:int=0):
        if not max:
            max = self.max_len - self.n_forward - 1
        return random.randint(min, self.max_len - self.n_forward - 1)

    def _set_offset(self, length: int):
        rand = self._new_offset()
        self.offset = min(rand, length - self.n_forward - 1)

    def _check_processed(self):
        if 'x' in self.processed and 'y' in self.processed:
            return True
        else:
            return False

    def _process(self, name: str, length: int):
        if self._check_processed:
            self._set_offset(length)
            self.processed.clear()
        self.processed.append(name)
        
    @override
    def _format_x(self, x: Tensor) -> Tensor:
        self._process('x', x.shape[0])
        return x[:self.offset]

    @override
    def _format_y(self, y: Tensor) -> Tensor:
        self._process('y', y.shape[0])
        return y[self.offset:self.offset + self.n_forward].long()

    

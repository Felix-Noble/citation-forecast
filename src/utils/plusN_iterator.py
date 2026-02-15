# src/utils/plusN_iterator.py 
from typing import Iterator
import torch
from torch import Tensor

def plusN_iterator(iterator: Iterator[tuple[torch.Tensor, ...]], extra_iters: int) -> Iterator[tuple[torch.Tensor, ...]]:
    for i, entry in enumerate(iterator):
        yield torch.tensor(i), *entry

    extra_entry = tuple(torch.tensor(float('nan')) for _ in range(len(tuple(entry)) + 1))
    for _ in range(extra_iters):
        yield extra_entry


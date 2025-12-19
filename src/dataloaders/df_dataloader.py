import dask.dataframe as dd
import pandas as pd # pyright: ignore[reportMissingTypeStubs] 
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from typing import cast, override

class DF_DataLoader(Dataset[tuple[Tensor, ...]]):
    def __init__(self, 
                 data_path: str,
                 X: str,
                 y: str | None,
                 max_len: int,
                 pad_value: int = 0,
                 pad: bool = True,
                 truncate: bool = True,
                 ):
        super().__init__()

        if y is None:
            columns = [X]
        else:
            columns = list(set([X, y]))

        self.df: pd.DataFrame = cast(pd.DataFrame, dd.read_parquet(data_path, columns=columns).compute()) # pyright: ignore[reportPrivateImportUsage, reportUnknownMemberType]
        self.X: str = X
        self.Y: str | None = y

        self.PAD: bool = pad
        self.TRUNCATE: bool = truncate
        self.MAX_LEN: int = max_len
        self.PAD_VALUE: int = pad_value

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
         
        x = Tensor(self.df.loc[idx, self.X])
       
        if self.PAD & x.size(0) < self.MAX_LEN:
            x = nn.functional.pad(x, (0, self.MAX_LEN - x.size(0)), value=self.PAD_VALUE)
        if self.TRUNCATE & x.size(0) > self.MAX_LEN:
            x = x[:self.MAX_LEN]
        
        if self.Y is not None:
            y = Tensor(self.df.loc[idx, self.Y])
            return (x, y)
        
        return (x,)

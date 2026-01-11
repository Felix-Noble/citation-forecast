import dask
import dask.dataframe as dd
import pandas as pd  
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import Dataset
from typing import cast
from rich.console import Console

from config.config import config
from src.utils.logging import setup_logger
from pathlib import Path
from logging import getLogger
logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)
console = Console()

dask.config.set({"dataframe.convert-string": False})

class DF_Dataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, 
                 data_path: str,
                 X: str,
                 t_start: int,
                 t_end: int,
                 y: str | None = None,
                 max_len: int = int(1e32),
                 pad_value: int = 0,
                 pad: bool = True,
                 truncate: bool | str = 'drop',
                 testing: bool = False,
                 ):
        super().__init__()

        if y is None:
            columns = [X]
        else:
            columns = list(set([X, y]))
        
        self.df: dd.DataFrame = cast(dd.DataFrame, 
                                     dd.read_parquet(data_path, columns=columns + ['publication_date_int'], engine='fastparquet') # pyright: ignore[reportPrivateImportUsage, reportUnknownMemberType]
                                     ) 

        self.df = self.df[(self.df['publication_date_int'] >= t_start) & (self.df['publication_date_int'] < t_end)]
        self.df = self.df[columns]
        self.df = cast(pd.DataFrame, 
                       self.df.compute() # pyright: ignore[reportUnknownMemberType]
                       ) 
        prev_n = self.df.shape[0]
        self.df = self.df.dropna(subset=X)
        logger.info(f"Dropped {prev_n - self.df.shape[0]} missing '{X}' rows")
        if y is not None:
            prev_n = self.df.shape[0]
            self.df = self.df.dropna(subset=y)
            logger.info(f"Dropped {prev_n - self.df.shape[0]} missing '{y}' rows")

        if truncate == 'drop':
            prev_n = self.df.shape[0]
            self.df[f'{X}_len'] =  self.df[X].apply(lambda x : len(x))# pyright: ignore[reportUnknownMemberType]
            self.df = self.df.loc[self.df[f'{X}_len'] <= max_len, columns]
            logger.info(f"Dropped {prev_n - self.df.shape[0]:,} '{X}' len > {max_len:,} | {self.df.shape[0]:,} remaining") # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        if testing:
            self.df = self.df.sample(n=20000, replace=True)

        self.df.reset_index(drop=True, inplace=True) 

        self.X: str = X
        self.Y: str | None = y

        self.PAD: bool = pad
        self.TRUNCATE: bool | str = truncate
        self.MAX_LEN: int = max_len
        self.PAD_VALUE: int = pad_value

    def __len__(self) -> int:
        return self.df.shape[0]

    def _format_X(self, x: Tensor) -> Tensor:
        return x

    def _format_y(self, y: Tensor) -> Tensor:
        return y

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        x: Tensor = torch.tensor(self.df.loc[idx, self.X], dtype=torch.float32)

        if self.PAD & x.size(0) < self.MAX_LEN:
            x = nn.functional.pad(x, (0, self.MAX_LEN - x.size(0)), value=self.PAD_VALUE)
        if self.TRUNCATE == True & x.size(0) > self.MAX_LEN:
            x = x[:self.MAX_LEN]
       
        x = self._format_X(x)
        if self.Y is not None:
            y = torch.tensor(self.df.loc[idx, self.Y], dtype=torch.float32)
            return x, self._format_y(y)

        return x, torch.tensor(float('nan'), dtype=torch.float32)

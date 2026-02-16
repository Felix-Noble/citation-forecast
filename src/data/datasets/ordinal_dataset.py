import polars as pl
import pandas as pd  
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import Dataset

from config import Config, config
from src.utils.logging import setup_logger
from pathlib import Path
from logging import getLogger
logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

class OrdinalDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, 
                 data_path: str,
                 X: str,
                 y: str,
                 t_start: int,
                 t_end: int,
                 config: Config = config,
                 pad: bool = True,
                 truncate: bool | str = 'drop',
                 return_mask: bool = False,
                 dry_run: bool = False,
                 ):
        super().__init__()
        self.X: str = X
        self.y: str = y
        self.t_start: int = t_start
        self.t_end: int = t_end
        self.MAX_LEN: int = config.model.max_len
        self.PAD_VALUE: int = config.model.pad_value
        self.N_BUCKETS: int = config.model.n_out
        self.PAD: bool = pad
        self.RETURN_MASK: bool = return_mask
        self.TRUNCATE: bool | str = truncate

        if y is None:
            columns = [self.X]
        else:
            columns = list(set([self.X, self.y]))
        columns += ['publication_date_int']        

        self.lf: pl.LazyFrame = (
                pl.scan_parquet(data_path)
                .select(columns)
                .filter((pl.col('publication_date_int') >= self.t_start) |
                        pl.col('publication_date_int') < self.t_end)
                ) 
        if dry_run:
            self.lf = self.lf.slice(0, (config.train.batch_size * 3) - 1)

        self.df: pd.DataFrame = self.lf.collect().to_pandas()
        prev_n = self.df.shape[0]
        self.df = self.df.dropna(subset=self.X)
        logger.info(f"Dropped {prev_n - self.df.shape[0]} missing '{self.X}' rows")
        if self.y is not None:
            prev_n = self.df.shape[0]
            self.df = self.df.dropna(subset=self.y)
            logger.info(f"Dropped {prev_n - self.df.shape[0]} missing '{self.y}' rows")

        if truncate == 'drop':
            prev_n = self.df.shape[0]
            self.df[f'{self.X}_len'] =  self.df[self.X].apply(lambda x : len(x))# pyright: ignore[reportUnknownMemberType]
            self.df = self.df.loc[self.df[f'{self.X}_len'] <= self.MAX_LEN, columns]
            logger.info(f"Dropped {prev_n - self.df.shape[0]:,} '{self.X}' len > {self.MAX_LEN:,} | {self.df.shape[0]:,} remaining") # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        self.df.reset_index(drop=True, inplace=True) 

    def __len__(self) -> int:
        return self.df.shape[0]

    def _format_X(self, x: Tensor) -> Tensor:
        return x.long()

    def _format_y(self, y: Tensor) -> Tensor:
        return torch.round(y * (self.N_BUCKETS - 1), decimals=0).long()

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        x: Tensor = torch.tensor(self.df.loc[idx, self.X], dtype=torch.float32)

        if self.PAD & x.size(0) < self.MAX_LEN:
            x = nn.functional.pad(x, (0, self.MAX_LEN - x.size(0)), value=self.PAD_VALUE)
        if self.TRUNCATE == True & x.size(0) > self.MAX_LEN:
            x = x[:self.MAX_LEN]
       
        x = self._format_X(x)
        y = torch.tensor(float('nan'), dtype=torch.float32) 
        if self.y is not None:
            y = torch.tensor(self.df.loc[idx, self.y], dtype=torch.float32)
            y = self._format_y(y)

        if self.RETURN_MASK:
            mask = (x != self.PAD_VALUE).bool().unsqueeze(0).expand(self.MAX_LEN, -1)
            return x, y, mask
        else:
            return x, y

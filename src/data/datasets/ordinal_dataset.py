from datetime import date
import polars as pl
import pandas as pd  
from typing import override
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
                 X: str | list[str],
                 y: str | list[str],
                 t_start: date,
                 t_end: date,
                 config: Config = config,
                 pad: bool = False,
                 truncate: bool | str = 'drop',
                 return_mask: bool = False,
                 dry_run: bool = False,
                 ):
        super().__init__()

        if isinstance(X, str):
            self.x: list[str] = [X]
        else:
            self.x: list[str] = X
        if isinstance(y, str):
            self.y: list[str] = [y]
        else:
            self.y: list[str] = y
        columns = self.x + self.y + ['publication_date']        

        self.t_start = t_start
        self.t_end = t_end
        self.MAX_LEN: int = config.model.max_len
        self.PAD_VALUE: int = config.model.pad_token_id
        self.N_BUCKETS: int = config.model.n_out
        self.PAD: bool = pad
        self.RETURN_MASK: bool = return_mask
        self.TRUNCATE: bool | str = truncate

        self.lf: pl.LazyFrame = (
                pl.scan_parquet(list(Path(data_path).glob('*.par*')))
                .select(columns)
                .filter((pl.col('publication_date') >= self.t_start) & (pl.col('publication_date') < self.t_end))
                ) 

        if dry_run:
            self.lf = self.lf.slice(0, (config.train.batch_size * 3) - 1)

        self.df: pd.DataFrame = self.lf.collect().to_pandas()
        prev_n = self.df.shape[0]
        self.df = self.df.dropna(subset=self.x)
        logger.info(f"Dropped {prev_n - self.df.shape[0]} missing '{self.x}' rows")
        if self.y:
            prev_n = self.df.shape[0]
            self.df = self.df.dropna(subset=self.y)
            logger.info(f"Dropped {prev_n - self.df.shape[0]} missing '{self.y}' rows")

        if truncate == 'drop':
            prev_n = self.df.shape[0]
            self.df['example_len'] = 0
            for col in self.x:
                self.df[f'{col}_len'] =  self.df[col].apply(lambda x : len(x))# pyright: ignore[reportUnknownMemberType]
                self.df['example_len'] += self.df[f'{col}_len'] 
            self.df = self.df.loc[self.df['example_len'] <= self.MAX_LEN, columns]
            logger.info(f"Dropped {prev_n - self.df.shape[0]:,} '{self.x}' len > {self.MAX_LEN:,} | {self.df.shape[0]:,} remaining") # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]

        self.df.reset_index(drop=True, inplace=True) 

    def __len__(self) -> int:
        return self.df.shape[0]

    def _format_x(self, x: Tensor) -> Tensor:
        return x.long()

    def _format_y(self, y: Tensor) -> Tensor:
        ordinal = torch.round(y * (self.N_BUCKETS - 1), decimals=0).long()
        one_hot = nn.functional.one_hot(ordinal, num_classes=self.N_BUCKETS)
        return one_hot 

    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        x: Tensor = torch.cat(
            [torch.from_numpy(arr.copy()) for arr in self.df.loc[idx, self.x].to_list()],
                                 ).flatten()
        if self.PAD and x.size(0) < self.MAX_LEN:
            x = nn.functional.pad(x, (0, self.MAX_LEN - x.size(0)), value=self.PAD_VALUE)
        if self.TRUNCATE == True & x.size(0) > self.MAX_LEN:
            x = x[:self.MAX_LEN]
       
        x = self._format_x(x)
        y = torch.tensor(float('nan'), dtype=torch.float32) 
        if self.y:
            y: Tensor = torch.tensor(
                    self.df.loc[idx, self.y].to_list(),
                    dtype=torch.float32
                                     ).flatten()
            y = self._format_y(y)

        if self.RETURN_MASK:
            mask = (x != self.PAD_VALUE).bool().unsqueeze(0).expand(self.MAX_LEN, -1)
            return x, y, mask
        else:
            return x, y

from src.utils import export_parquet
from datetime import date
import polars as pl
import os
import shutil
from typing import override
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import Dataset

from config import Config, config, env
from src.utils.logging import setup_logger
from pathlib import Path
from logging import getLogger
logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

class PolarsDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, 
                 data_path: str,
                 X: str | list[str],
                 y: str | list[str],
                 t_start: date,
                 t_end: date,
                 max_len: int,
                 weights: list[float] | None=None,
                 time_col: str = 'publication_date',
                 config: Config = config,
                 pad: bool = False,
                 truncate: bool | str = 'drop',
                 return_mask: bool = False,
                 dry_run: bool = False,
                 name: str = 'dataset',
                 auto_remove: bool = False,
                 **kwargs,
                 ):
        super().__init__()
        assert not (weights is None and not y), 'Weights cannot be given when no y is given'

        if isinstance(X, str):
            self.x: list[str] = [X]
        else:
            self.x: list[str] = X
        if isinstance(y, str):
            self.y: list[str] = [y]
        else:
            self.y: list[str] = y
        columns = self.x + self.y + [ time_col ]        

        self.t_start = t_start
        self.t_end = t_end
        self.weights = weights
        self.time_col = time_col
        self.MAX_LEN: int = max_len 
        self.PAD_VALUE: int = config.model.pad_token_id
        self.N_BUCKETS: int = config.model.n_out
        self.PAD: bool = pad
        self.RETURN_MASK: bool = return_mask
        self.TRUNCATE: bool | str = truncate
        self.name: str = name
        self.hot_path: Path = Path('./temp') / 'hot' / self.name
        if dry_run:
            self.hot_path: Path = Path('./temp') / 'hot' / f'{self.name}-DRY'

        self.x_hot_path: Path = self.hot_path / 'x.ipc'
        self.y_hot_path: Path = self.hot_path / 'y.ipc'

        if self.hot_path.exists() and auto_remove:
            logger.info(f'Found data at {self.hot_path}, deleting')
            shutil.rmtree(self.hot_path)
        if not self.hot_path.exists():
            os.makedirs(self.hot_path, exist_ok=True)
            files = list(Path(data_path).glob('*.par*'))
            
            lf: pl.LazyFrame = (
                    pl.scan_parquet(files)
                    .select(columns)
                    .filter((pl.col(self.time_col) >= self.t_start) & (pl.col(self.time_col) < self.t_end))
                    )

            if dry_run:
                lf = lf.slice(0, (config.train.batch_size * 3) - 1)
 
            lf = lf.drop_nulls(columns)

            if truncate == 'drop':
                lf = lf.with_columns(
                        total_len = pl.lit(0, dtype=pl.Int32)
                        )

                for col in self.x:
                    if f'{col}_len' in list(lf.schema.keys()):
                        logger.debug(f'Adding {col} len from precalculated {col}_len col ')
                        lf = lf.with_columns(
                                total_len = pl.col('total_len') + pl.col(f'{col}_len')
                                )
                    else:
                        logger.debug(f'Calculating {col} len')
                        lf = lf.with_columns(
                                total_len = pl.col('total_len') + pl.col(col).list.len()
                                )
                lf = lf.filter(
                        pl.col('total_len') <= self.MAX_LEN
                        )
                lf = lf.drop('total_len')

            lf = lf.drop([self.time_col]) 
            
            rows = lf.select(pl.len()).collect(engine='streaming').item()
            logger.info(f'Saving {rows:,} rows where summed length of {self.x} <= {self.MAX_LEN} to hotpath: {self.name}')
            
            lf.select(self.x).sink_ipc(self.x_hot_path)
            lf.select(self.y).sink_ipc(self.y_hot_path)

        self.df_x: pl.DataFrame = pl.read_ipc(self.x_hot_path, memory_map=True)         
        self.df_y: pl.DataFrame = pl.read_ipc(self.y_hot_path, memory_map=True)         
        logger.info(f'Hot path {self.hot_path} loaded') 
    
    def __len__(self) -> int:
        return len(self.df_x)

    def _format_x(self, x: Tensor) -> Tensor:
        return x.long()
    
    def _format_y(self, y: Tensor) -> Tensor:
        return y
   
    @override
    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        x_row: tuple[list[int], ...] = self.df_x.row(idx)
        x: Tensor = torch.cat(
            [torch.tensor(token_list) for token_list in x_row]
            ).flatten()
        if self.PAD and x.size(0) < self.MAX_LEN:
            x = nn.functional.pad(x, (0, self.MAX_LEN - x.size(0)), value=self.PAD_VALUE)
        if self.TRUNCATE == True & x.size(0) > self.MAX_LEN:
            x = x[:self.MAX_LEN]
       
        x = self._format_x(x)

        y = torch.tensor(float('nan'), dtype=torch.float32) 
        if self.y:
            y_row: tuple[list[int], ...] = self.df_y.row(idx)
            y: Tensor = torch.tensor(
                [torch.tensor(target, dtype=torch.float32) for target in y_row]
                ).flatten()
            y = self._format_y(y)

        if self.weights is None:
            weight = torch.tensor(float('nan'), dtype=torch.float32) 
        else:
            weight: Tensor = torch.tensor(
                [torch.tensor(self.weights[target.long()], dtype=torch.float32) for target in torch.atleast_1d(y)]
                ).flatten()

        if self.RETURN_MASK:
            mask = (x != self.PAD_VALUE).bool().unsqueeze(0).expand(self.MAX_LEN, -1)
            return x, y, mask, weight
        else:
            return x, y, weight

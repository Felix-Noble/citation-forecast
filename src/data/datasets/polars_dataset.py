from src.utils import export_parquet
from datetime import date
import polars as pl
import os
import shutil
from typing import override
from dataclasses import dataclass
import torch.nn as nn
import torch
from torch import Tensor, tensor
from torch.utils.data import Dataset

from config import Config, config, env
from src.utils.logging import setup_logger
from pathlib import Path
from logging import getLogger
logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

@dataclass
class Output:
    id: Tensor = tensor(float('nan'))
    x: Tensor = tensor(float('nan'))
    y: Tensor = tensor(float('nan'))
    weight: Tensor = tensor(float('nan'))
    mask: Tensor = tensor(float('nan'))

class PolarsDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, 
                 data_path: str,
                 X: str | list[str],
                 y: str | list[str],
                 t_start: date | None,
                 t_end: date | None,
                 max_len: int,
                 weights: list[float] | None=None,
                 time_filter: bool=False,
                 time_col: str = 'publication_date',
                 return_id: bool = False,
                 id_col: str = 'id',
                 config: Config = config,
                 pad: bool = False,
                 truncate: str = 'drop',
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
            self.x = X
        if isinstance(y, str):
            self.y: list[str] = [y]
        else:
            self.y = y
        columns = list(set(self.x + self.y))
        
        if time_filter:
            columns += [ time_col ]        
        if return_id:
            columns += id_col

        self.t_start = t_start
        self.t_end = t_end
        self.weights = weights
        self.time_col = time_col
        self.return_id = return_id 
        self.id_col = id_col
        self.max_len: int = max_len 
        self.PAD_VALUE: int = config.model.pad_token_id
        self.N_BUCKETS: int = config.model.n_out
        self.PAD: bool = pad
        self.RETURN_MASK = return_mask
        self.TRUNCATE = truncate
        self.name = name
        self.hot_path: Path = Path('./.temp') / 'hot' / self.name
        if dry_run:
            self.hot_path = Path('./.temp') / 'hot' / f'{self.name}-DRY'

        self.x_hot_path: Path = self.hot_path / 'x.ipc'
        self.y_hot_path: Path = self.hot_path / 'y.ipc'
        self.id_hot_path: Path = self.hot_path / 'id.ipc'

        if self.hot_path.exists() and auto_remove:
            logger.info(f'Found data at {self.hot_path}, deleting')
            shutil.rmtree(self.hot_path)
        if not self.hot_path.exists():
            os.makedirs(self.hot_path, exist_ok=True)
            files = list(Path(data_path).glob('*.par*'))
            
            lf: pl.LazyFrame = (
                    pl.scan_parquet(files)
                    .select(columns)
                    )

            if self.t_start is not None and time_filter:
                    lf = lf.filter((pl.col(self.time_col) >= self.t_start))
            if self.t_end is not None and time_filter:
                    lf = lf.filter(pl.col(self.time_col) < self.t_end) 

            if dry_run:
                lf = lf.slice(0, (config.train.batch_size * 3))
 
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
                        pl.col('total_len') <= self.max_len
                        )
                lf = lf.drop('total_len')

            if time_filter:
                lf = lf.drop([self.time_col]) 
            
            rows = lf.select(pl.len()).collect(engine='streaming').item()
            logger.info(f'Saving {rows:,} rows where summed length of {self.x} <= {self.max_len} to hotpath: {self.name}')
            
            lf.select(self.x).sink_ipc(self.x_hot_path)
            lf.select(self.y).sink_ipc(self.y_hot_path)
            if self.return_id:
                lf.select([self.id_col]).sink_ipc(self.id_hot_path)

        self.df_x: pl.DataFrame = pl.read_ipc(self.x_hot_path, memory_map=True)         
        self.df_y: pl.DataFrame = pl.read_ipc(self.y_hot_path, memory_map=True)         
        if self.return_id:
            self.df_id: pl.DataFrame = pl.read_ipc(self.id_hot_path, memory_map=True)         
        logger.info(f'Hot path {self.hot_path} loaded') 
    
    def __len__(self) -> int:
        return len(self.df_x)

    def _format_x(self, x: Tensor) -> Tensor:
        return x.long()
    
    def _format_y(self, y: Tensor) -> Tensor:
        return y
   
    @override
    def __getitem__(self, idx: int):
        out: Output = Output()

        # X (input)
        x_row: tuple[list[int], ...] = self.df_x.row(idx)
        x: Tensor = torch.cat(
            [torch.tensor(token_list) for token_list in x_row]
            ).flatten()

        x = self._format_x(x)
        if self.PAD and x.size(0) < self.max_len:
            x = nn.functional.pad(x, (0, self.max_len - x.size(0)), value=self.PAD_VALUE)
        if self.TRUNCATE == True & x.size(0) > self.max_len:
            x = x[:self.max_len]
       
        out.x = x

        # y (target)
        if self.y:
            y_row: tuple[list[int], ...] = self.df_y.row(idx)
            y: Tensor = torch.cat(
                [torch.tensor(target, dtype=torch.float32) for target in y_row]
                ).flatten()
            y = self._format_y(y)
            out.y = y
         
        # id (tracking)
        if self.return_id:
            id = torch.tensor(self.df_id.row(idx))
            out.id = id
        
        # weights (per target class)
        if self.weights is not None:
            weight: Tensor = torch.tensor(
                [torch.tensor(self.weights[target.long()], dtype=torch.float32) for target in torch.atleast_1d(y)]
                ).flatten()
            out.weight = weight

        if self.RETURN_MASK:
            mask = (x != self.PAD_VALUE).bool().unsqueeze(0).expand(self.max_len, -1)
            out.mask = mask

        return out.__dict__

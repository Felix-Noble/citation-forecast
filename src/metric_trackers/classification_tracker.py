from typing import NamedTuple
from dataclasses import dataclass
import pandas as pd
from rich.table import Table
from rich.console import Console
from sklearn.metrics import roc_auc_score # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs] 
import torch
from pathlib import Path
from logging import getLogger
from src.utils.logging import setup_logger
from config.config import config

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

class StoreParams(NamedTuple):
    name: str
    batch_shape: tuple[int, ...]
    buffer_size: int
    buffer_device: torch.device
    max_store: int
    n_examples: int
 
@dataclass 
class Store():
    name: str
    buffer: torch.Tensor
    store: list[torch.Tensor]
    buffer_cursor: torch.Tensor
    store_cursor: int

class MetricTuple(NamedTuple):
    " Metric Tuple: stores named metric scores / weight values for dataframe concatenations"
    score: float
    weight: float
    
class ClassificationTracker:
    """
        Classification Metrics Tracker:

        Stores intermediate outputs on GPU, calcualtes results, displays them

        args:        
            output_shape: shape out output by model
            output_buffer_size: n. of outputs that will be stored in 'fast' buffer
            output_max_size: n. of outputs that will be stored before moving to CPU

    """
    def __init__(self,
                store_params: tuple[StoreParams, ...],
                dtype: torch.dtype,
                device: torch.device,
                 ):

        self.dtype = dtype
        self.device: torch.device = device
        self.store_params: dict[str, StoreParams] = {params.name: params for params in store_params}
        self.stores: dict[str, Store] = {params.name: self._init_store(params) for params in store_params}

        self.metric_store: dict[str, list[MetricTuple]] = {}
        self.metric_df: pd.DataFrame = pd.DataFrame()

        # rich console 
        self.console: Console = Console()

    def _init_store(self, params: StoreParams) -> Store:
        return Store(
            name=params.name,
            buffer=torch.empty((params.buffer_size, *params.batch_shape), dtype=self.dtype, device=params.buffer_device),
            store=[],
            buffer_cursor=torch.tensor(0, dtype=torch.int32, device=self.device),
            store_cursor=0,
        )
    def _reset_store(self, store: Store) -> Store:
        self.stores[store.name] = self._init_store(self.store_params[store.name])
        return self.stores[store.name]

    def clear(self) -> None:
        for store in self.stores.values():
            _ = self._reset_store(store)
            
    def _flush_buffer(self, store: Store) -> None:
        " Flushed bufffer to CPU, resets cursor"
        store.store.append(store.buffer.cpu())
        store.buffer = torch.empty_like(store.buffer)
        store.buffer_cursor = torch.zeros_like(store.buffer_cursor)
        store.store_cursor += store.store[-1].size(0)

    def _write_buffer(self, value: torch.Tensor, store: Store) -> None:
        " Writes value to store buffer "
        store.buffer[store.buffer_cursor] = value.detach()
        store.buffer_cursor.add_(1)

    def process_value(self, value: torch.Tensor, store: Store | str) -> None:
        """Stores outputs in buffer, syncs buffer to CPU when full and flushes, calculates metrics when store full and clears
            input: 
                value: Tensor to be stored
                store: store or name of store
        """
        if isinstance(store, str):
            if store not in self.stores.keys():
                logger.error(f"Store '{store}' not found ")
                return 
            store = self.stores[store]

        if torch.any(torch.isnan(value)):
            logger.error('NaN values passed to process_value, not writing to buffer')
            return

        buffer_full = store.buffer_cursor >= store.buffer.size(0) 
        if buffer_full:
            _ = self._flush_buffer(store)
        _ = self._write_buffer(value, store)

    def _gather_store(self, 
                      store: Store | None=None,
                      store_name: str | None=None,
                      ) -> torch.Tensor:
        """Concatenates and flattens all stored buffers 

        out shape: (batch, *output_shape)
        """
        if store is None:
            if store_name is None:
                logger.error('Store and Store_name are None, cannot gather store, returning NaN tensor')
                return torch.tensor(float('nan'))
            store = self.stores[store_name]
        _ = self._flush_buffer(store)

        all_values: torch.Tensor = torch.concatenate(store.store, dim=0)
        self._reset_store(store)
        return all_values.flatten(end_dim=1)

    def log_metric(self,
                    name: str,
                    value: float,
                    weight: float | int,
                    ):
        if name not in self.metric_store.keys():
            self.metric_store[name] = []
        self.metric_store[name].append(MetricTuple(value, weight))

    def calc_metrics(self, 
                logit_store_name: str,
                y_store_name: str,
                prefix: str = "",
                ):
        logits = self._gather_store(store_name=logit_store_name)
        probs = torch.softmax(
            logits,
            dim=1,
        )
        y_true = self._gather_store(store_name=y_store_name)
        shortest_len = logits.size(0) if logits.size(0) < y_true.size(0) else y_true.size(0)
        logger.debug(f'Before cut: logits: {logits.shape}, y: {y_true.shape}')
        logits, y_true = logits[:shortest_len], y_true[:shortest_len]
        logger.debug(f'After cut: logits: {logits.shape}, y: {y_true.shape}')
        try:
            roc_auc = roc_auc_score( # pyright: ignore[reportUnknownVariableType]
                y_true.long().numpy(), 
                probs.numpy(), 
                multi_class='ovo', 
                average='weighted')
            self.log_metric(f'{prefix}_roc_auc', roc_auc, probs.shape[0]) # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
            return

    def _aggregate_metrics(self) -> dict[str, float]:
        " Aggregates metrics stored as named tuples "
        aggregate_metrics = {k: float('nan') for k in self.metric_store.keys()} 
        for metric in aggregate_metrics.keys():
            df: pd.DataFrame = pd.DataFrame(self.metric_store[metric])
            score: float = (df['score'] * df['weight']).sum() / (df['weight'].sum())
            aggregate_metrics[metric] = score

        return aggregate_metrics 

    def report(self, 
               progress_bar,
               epoch: int | None=None, 
               aggregate_metrics: dict[str, float] | None=None
               ) -> None:
        " Generates rich.Table, renders as string via capture(), returns"
        if aggregate_metrics is None:
            aggregate_metrics = self._aggregate_metrics()

        cols: list[str] = ['cyan', 'green']
        table: Table = Table(show_header=True, pad_edge=True, padding=(0,1)) 
        table.add_column('Epoch', style='cyan')
        for i, (k) in enumerate(aggregate_metrics.keys()):
            table.add_column(k, style=cols[i % len(cols)])

        row_data = [str(epoch) if epoch is not None else 'NA']
        row_data.extend(str(v) for v in aggregate_metrics.values())
        table.add_row(*row_data)

        progress_bar.console.print(table)

from typing import Callable, NamedTuple
from dataclasses import dataclass
import pandas as pd
from rich.table import Table
from rich.console import Console
from sklearn.metrics import roc_auc_score # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs] 
import torch

class StoreParams(NamedTuple):
    name: str
    batch_shape: tuple[int, ...]
    n_batches: int
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
                store_params: list[StoreParams],
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
            buffer=torch.empty((params.n_batches, *params.batch_shape), dtype=self.dtype, device=self.device),
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

    def process_value(self, value: torch.Tensor, store: Store) -> None:
        " Stores outputs in buffer, syncs buffer to CPU when full and flushes, calculates metrics when store full and clears"
        buffer_full = store.buffer_cursor >= store.buffer.size(0) 
        if buffer_full:
            _ = self._flush_buffer(store)
        _ = self._write_buffer(value, store)

    def _gather_store(self, store: Store) -> torch.Tensor:
        """Concatenates and flattens all stored buffers 

        out shape: (batch, *output_shape)
        """
        _ = self._flush_buffer(store)

        all_values: torch.Tensor = torch.concatenate(store.store, dim=0)
        self._reset_store(store)
        return all_values.flatten(end_dim=1)

    def _calc_roc_auc(self, logits: torch.Tensor, y_true: torch.Tensor):
        probs = torch.softmax(
                torch.flatten(logits, end_dim=1),
                dim=1,
        )
        score = roc_auc_score(y_true, probs, multi_class='ovo', average='weighted') 
        return score

    def _log_metric(self,
                    name: str,
                    value: float,
                    weight: float | int,
                    ):
        if name not in self.metric_store.keys():
            self.metric_store[name] = []
        self.metric_store[name].append(MetricTuple(value, weight))

    def process(self):
        outputs, y_true = self._gather_store()
        roc_auc = self._calc_roc_auc(outputs, y_true)
        self._log_metric('roc_auc', roc_auc, 1)

    def _aggregate_metrics(self) -> dict[str, float]:
        " Aggregates metrics stored as named tuples "
        aggregate_metrics = {k: float('nan') for k in self.metric_store.keys()} 
        for metric in aggregate_metrics.keys():
            df: pd.DataFrame = pd.DataFrame(self.metric_store[metric])
            score: float = (df['score'] * df['weight']).sum() / (df['weight'].sum())
            aggregate_metrics[metric] = score

        return aggregate_metrics 

    def report(self, aggregate_metrics: dict[str, float] | None=None) -> str:
        " Generates rich.Table, renders as string via capture(), returns"
        if aggregate_metrics is None:
            aggregate_metrics = self._aggregate_metrics()

        cols: list[str] = ['cyan', 'green']
        table: Table = Table(show_header=False, pad_edge=False) 
        for i, (k, v) in enumerate(aggregate_metrics.items()):
            table.add_column(style=cols[i % len(cols)])
            table.add_row(k, str(v)) 

        with self.console.capture() as capture:
            self.console.print(table)

        return capture.get()

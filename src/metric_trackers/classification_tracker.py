from typing import Callable, NamedTuple
import pandas as pd
from rich.table import Table
from rich.console import Console
from sklearn.metrics import roc_auc_score # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs] 
import torch

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
                 output_shape: tuple[int, ...],
                 output_buffer_size: int,
                 output_store_size: int,
                 n_examples: int,
                 dtype: torch.dtype,
                 device: torch.device,
                 ):
        self.output_shape: tuple[int, ...] = output_shape
        self.output_buffer_size: int = output_buffer_size 
        self.output_store_size: int = output_store_size
        self.device: torch.device = device

        self.metric_store: dict[str, list[MetricTuple]] = {}
        self.metric_df: pd.DataFrame = pd.DataFrame()
        #self.y_true: torch.Tensor = torch.tensor(False)

        # rich console 
        self.console: Console = Console()

        # Ouptut buffers 
        self._init_buffer()
        self._init_store()

        # Cursors
        self._init_cursors()    

        self.store_output: Callable[[torch.Tensor], None]
        # select store_output method 
        if output_buffer_size >= n_examples:
            # TODO: add logging
            self.store_output = self._store_output_fast
        else:
            self.store_output = self._store_output_cpu

    def _init_buffer(self) -> None:
         self.output_buffer: torch.Tensor = torch.empty((self.output_buffer_size, *self.output_shape))
    def _init_store(self) -> None:
        self.output_store: list[torch.Tensor] = []       
    def _init_cursors(self) -> None:
        self.buffer_cursor: torch.Tensor = torch.tensor(0, device=self.device, dtype=torch.int32)
        self.output_store_cursor: int = 0

    def clear(self) -> None:
        self._init_buffer()
        self._init_store()
        self._init_cursors()

    def _flush_buffer(self) -> None:
        " Flushed bufffer to CPU, resets cursor"
        self.output_store.append(self.output_buffer.cpu())
        self.output_buffer = torch.empty_like(self.output_buffer)
        _ = self.buffer_cursor.fill_(0)
        self.output_store_cursor += self.output_store[-1].size(0)

    def _store_output_fast(self, output: torch.Tensor) -> None:
        " Stores outputs in buffer only "
        #test = output.detach()
        self.output_buffer[self.buffer_cursor] = output.detach()
        _ = self.buffer_cursor.add_(1)

    def _store_output_cpu(self, output: torch.Tensor) -> None:
        " Stores outputs in buffer, syncs buffer to CPU when full and flushes, calculates metrics when store full and clears"
        buffer_full = self.buffer_cursor >= self.output_buffer_size

        if buffer_full:
            self._flush_buffer()

        self.output_buffer[self.buffer_cursor] = output.detach()
        _ = self.buffer_cursor.add_(1)
        
        store_full = self.output_store_cursor >= self.output_store_size

        if store_full:
            pass

    def calc_metrics(self):
        return ('Function under construction')
        if not self.y_true:
            raise ValueError('Y True attribute has not been set, set tracker.y_true before calling _calc_metrics')

        _ = self._flush_buffer()

        outputs: torch.Tensor = torch.concatenate(self.output_store, dim=0)
        self.output_store.clear()

        self.roc_auc.append(
            { 'score': float(roc_auc_score( # pyright: ignore[reportUnknownArgumentType]
                                           self.y_true, 
                                           outputs,
                                           average = 'macro',
                                           )),
             'weight': float(outputs.size(0))
             }
        )

        del outputs
        self.y_true = torch.tensor(False)

    def _log_metric(self,
                    name: str,
                    value: float,
                    weight: float | int,
                    ):
        if name not in self.metric_store.keys():
            self.metric_store[name] = []
        self.metric_store[name].append(MetricTuple(value, weight))

    def _aggregate_metrics(self) -> dict[str, float]:
        " Aggregates metrics stored as named tuples "
        aggregate_metrics = {k: float('nan') for k in self.metric_store.keys()} 
        for metric in aggregate_metrics.keys():
            df: pl.DataFrame = pl.DataFrame(self.metric_store[metric])
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

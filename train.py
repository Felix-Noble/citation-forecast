#!/usr/bin/env python3
# train.py
# pyright: strict
# basedpyright: reccomend
from typing import Iterator
from config.config import config, Config
from src.types.valid_paths import VALID_PATHS
from src.utils.logging import setup_logger
from src.metric_trackers.classification_tracker import ClassificationTracker
from src.datasets.df_dataset import DF_Dataset
from src.datasets.decile_dataset import DecileDataset

from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress, TextColumn, BarColumn
import typer
from pathlib import Path
from logging import getLogger

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)
 
app = typer.Typer()

def init_dataloader(
    dataset: Dataset[tuple[Tensor, ...] | Tensor],
    config: Config = config,
) -> DataLoader[tuple[Tensor, ...] | Tensor]:

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=1,
        prefetch_factor=None,
        pin_memory=True,
        shuffle=True,
    )
    return dataloader

def init_progress() -> tuple[Progress, ...]:
    epoch_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TextColumn('[bold green]{task.elapsed:.0f} < {task.remaining:.0f}[/bold green]') 
    )     
    example_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TextColumn('[bold green]{task.elapsed:.0f} < {task.remaining:.0f}[/bold green]') 
    )
    mem_util_progress = Progress(
        TextColumn('[yellow] {task.description}', justify='right'),
        BarColumn(bar_width=30),
        TextColumn('[task.completed]{task.completed:.1f}/{task.total:.1f} MiB'),
        TextColumn('[progress.completed]{task.percentage:>3.0f}%'),
    )
    
    epoch_progress.start()
    example_progress.start()
    mem_util_progress.start()
    return epoch_progress, example_progress, mem_util_progress

def init_mtrack_params(config: Config = config):
    from src.metric_trackers.classification_tracker import StoreParams
    gpu = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cpu = torch.device('cpu')
    train_logits = StoreParams(
        'train_logits',
        batch_shape=(config.train.batch_size, 11),
        buffer_size=4,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )
    train_y = StoreParams(
        'train_y',
        batch_shape=(config.train.batch_size, ),
        buffer_size=4,
        buffer_device=cpu,
        max_store=-1,
        n_examples=-1,
    )

    val_logits = StoreParams(
        'val_logits',
        batch_shape=(config.train.batch_size, 11),
        buffer_size=4,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )
    val_y = StoreParams(
        'val_y',
        batch_shape=(config.train.batch_size, ),
        buffer_size=4,
        buffer_device=cpu,
        max_store=-1,
        n_examples=-1,
    )

    return train_logits, train_y, val_logits, val_y

def plusN_iterator(iterator: Iterator[tuple[torch.Tensor, ...]], extra_iters: int) -> Iterator[tuple[torch.Tensor, ...]]:
    for entry in iterator:
        yield(entry)
    for _ in range(extra_iters):
        yield torch.tensor(float('nan')), torch.tensor(float('nan'))

def eval_model(model) -> None:
    return

@app.command()
def main(
    data_path: VALID_PATHS = typer.Argument( 
        help='data path relative to config.data.staged'
    )
) -> None:
    " Main Loop "
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Running on {device}')

    dataset = DecileDataset(
        data_path=str(config.data.staged / data_path),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        max_len=500,
        pad=True,
        pad_value=0,
        testing=True,
    )
    dataloader = init_dataloader(dataset)
    model = nn.Linear(
        dataset.MAX_LEN, 11, device=device
    ) 
    metric_tracker = ClassificationTracker(
        init_mtrack_params(),
        dtype=torch.float32,
        device=device,
    )

    epoch_progress, example_progress, mem_util_progress = init_progress() 
    epochs_done = epoch_progress.add_task('Epochs', total=config.train.epochs)
    examples_done = example_progress.add_task('Examples', total=len(dataloader))
    max_mem = 0
    if torch.cuda.is_available():
        _, max_mem = torch.cuda.mem_get_info()
    mem_used = mem_util_progress.add_task('Mem Util', total=max_mem* (1/(1024**2)))
    compute_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    forward_finished = torch.cuda.Event()
    batch_copy_finished = torch.cuda.Event()
    output_copy_finished = torch.cuda.Event()

    for epoch in range(1, config.train.epochs + 1):
        example_progress.update(examples_done, completed=0)
        iterator = plusN_iterator(iter(dataloader), extra_iters=1)
        with torch.cuda.stream(copy_stream):
            current_X, current_y = next(iterator)
            metric_tracker.process_value(current_y, 'train_y')
            current_X = current_X.to(device, non_blocking=True)
            current_y = current_y.to(device, non_blocking=True)
            batch_copy_finished.record()
            output_copy_finished.record()

        for next_X, next_y in iterator:
            with torch.cuda.stream(compute_stream):
                output_copy_finished.wait()
                batch_copy_finished.wait()
                out = model(current_X)
                forward_finished.record()

            with torch.cuda.stream(copy_stream):
                next_X_gpu = next_X.to(device, non_blocking=True)
                batch_copy_finished.record()
                metric_tracker.process_value(next_y, 'train_y')
                next_y_gpu = next_y.to(device, non_blocking=True)
                forward_finished.wait()
                metric_tracker.process_value(out, 'train_logits')
                output_copy_finished.record()
                current_X = next_X_gpu
                current_y = next_y_gpu
            example_progress.update(examples_done, advance=config.train.batch_size)
            mem_use, _ = torch.cuda.mem_get_info()
            mem_util_progress.update(mem_used, complete=mem_use * (1/(1024**2)))

        metric_tracker.calc_metrics(
            logit_store_name='train_logits',
            y_store_name='train_y',
            prefix='train'
        )
        _ = metric_tracker.report(
            progress_bar=epoch_progress, 
            epoch=epoch,
        )

        epoch_progress.update(epochs_done, advance=1)

        metric_tracker.clear()

if __name__ == '__main__':
    app()
    '''
    logger.info('Starting test')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    logger.info(f'Device: {device}')
    n_out = 3
    n_examps = 50
    batch_size = 5
    n_batches = n_examps // batch_size

    test_x = torch.randn((n_batches, batch_size, 20))
    test_y = torch.randint(0, n_out, (n_batches * batch_size,))
    metric_tracker = ClassificationTracker(
        (batch_size, n_out), n_batches, n_examps, n_examps, torch.float32, device
    )
    model: torch.nn.Linear = torch.nn.Linear(20, n_out)
     
    for examp in test_x:
        out: torch.Tensor = model(examp) # pyright: ignore[reportAny]
        metric_tracker.store_output(out)

    outputs = torch.softmax(metric_tracker._stack_buffer(), dim=1).cpu()
    roc_auc = roc_auc_score(test_y, outputs, multi_class='ovo', average='weighted')
    metric_tracker._log_metric('train roc_auc', roc_auc, 1)
    report = metric_tracker.report()
    print(report)
    '''

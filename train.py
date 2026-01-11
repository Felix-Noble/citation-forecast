#!/usr/bin/env python3
# train.py
from typing import Iterator, Type
from config.config import config, Config, EXPERIMENT_NAME
from src.types.valid_paths import VALID_PATHS
from src.utils.logging import setup_logger
from src.metric_trackers.classification_tracker import ClassificationTracker
from src.datasets.quartile_dataset import QuartileDataset 
from src.models.registry import get_model

from concurrent.futures import ThreadPoolExecutor
import mlflow
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
from rich.progress import Progress, TextColumn, BarColumn
import typer
from pathlib import Path
from logging import getLogger
TESTING = False
logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)
 
app = typer.Typer(pretty_exceptions_enable=not TESTING)

def isnan_async(loss: torch.float32):
    if torch.any(torch.isnan(loss)):
        raise ValueError('Loss is NaN, interrupting training')

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

def init_mlflow(
        model_name: str,
        data_path: str,
    ):

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlf_experiment = mlflow.set_experiment(EXPERIMENT_NAME)
    mlf_run = mlflow.start_run(run_name=model_name)
    param_dict = {}
    for k,v in config.__dict__.items():
        mod_params = {f'{k}-{sk}': val for sk, val in v.__dict__.items()}
        param_dict.update(mod_params)
    param_dict['data-raw'] = str(param_dict['data-raw']) + data_path
    mlflow.log_params(param_dict)
    return mlf_experiment, mlf_run

def init_progress() -> tuple[Progress, ...]:
    epoch_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TextColumn('[bold green]{task.elapsed:.0f} < {task.remaining:.0f}[/bold green]'),
        disable=TESTING,
    )     
    example_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TextColumn('[bold green]{task.elapsed:.0f} < {task.remaining:.0f}[/bold green]'),
        disable=TESTING,
    )
    mem_util_progress = Progress(
        TextColumn('[yellow] {task.description}', justify='right'),
        BarColumn(bar_width=30),
        TextColumn('[task.completed]{task.completed:.1f}/{task.total:.1f} MiB'),
        TextColumn('[progress.completed]{task.percentage:>3.0f}%'), 
        disable=TESTING,
    )

    epoch_progress.start()
    example_progress.start()
    mem_util_progress.start()
    return epoch_progress, example_progress, mem_util_progress

def init_mtrack_params(
        device: torch.device,
        config: Config = config, 
):
    from src.metric_trackers.classification_tracker import StoreParams
    gpu = device
    cpu = torch.device('cpu')
    train_logits = StoreParams(
        'train_logits',
        batch_shape=(config.train.batch_size, 5),
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

    train_loss = StoreParams(
        'train_loss',
        batch_shape=(config.train.batch_size, ),
        buffer_size=4,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )

    test_logits = StoreParams(
        'test_logits',
        batch_shape=(config.train.batch_size, 5),
        buffer_size=4,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )

    test_y = StoreParams(
        'test_y',
        batch_shape=(config.train.batch_size, ),
        buffer_size=4,
        buffer_device=cpu,
        max_store=-1,
        n_examples=-1,
    )

    test_loss = StoreParams(
        'test_loss',
        batch_shape=(config.train.batch_size, ),
        buffer_size=4,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )
    return train_logits, train_y, train_loss, test_logits, test_y, test_loss

def init_loss_fun(config: Config = config):
    valid_loss_funcs = {
        'CrossEntropyLoss' : nn.CrossEntropyLoss()
    } 
    if config.train.loss_fn not in valid_loss_funcs.keys():
        raise ValueError(f'Loss Function {config.train.loss_fn} not expected, select from {", ".join(valid_loss_funcs.keys())}')
    return valid_loss_funcs[config.train.loss_fn]

def init_optimizer(model: nn.Module,
                   config: Config = config
                   )-> Type[torch.optim.Optimizer]:
    valid_optimizers = {
        'AdamW': torch.optim.AdamW(
            model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        ),
    }
    if config.train.optimizer not in valid_optimizers.keys():
        raise ValueError(f'Optimizer {config.train.optimizer} not expected, select from {", ".join(valid_optimizers.keys())}')
    return valid_optimizers[config.train.optimizer]

def plusN_iterator(iterator: Iterator[tuple[torch.Tensor, ...]], extra_iters: int) -> Iterator[tuple[torch.Tensor, ...]]:
    for i, entry in enumerate(iterator):
        yield i, *entry

    for _ in range(extra_iters):
        yield torch.nan, torch.tensor(float('nan')), torch.tensor(float('nan'))

def eval_model(
    model: nn.Module,
    loss_fn,
    dataloader: DataLoader,
    example_progress,
    examples_done,
    metric_tracker: ClassificationTracker,
    compute_stream: torch.cuda.Stream,
    copy_stream: torch.cuda.Stream,
    device: torch.device,
) -> None:
    forward_finished = torch.cuda.Event()
    batch_copy_finished = torch.cuda.Event()
    output_copy_finished = torch.cuda.Event()
    
    model.eval()
    with torch.no_grad():
        for epoch in range(1, config.train.epochs + 1):
            example_progress.update(examples_done, completed=0)
            data_iterator = plusN_iterator(iter(dataloader), extra_iters=1)
            with torch.cuda.stream(copy_stream):
                batch_i, current_X, current_y = next(data_iterator)
                metric_tracker.process_value(current_y, 'test_y')
                current_X = current_X.to(device, non_blocking=True).long()
                current_y = current_y.to(device, non_blocking=True).long()
                batch_copy_finished.record()
                output_copy_finished.record()

            for next_batch_i, next_X, next_y in data_iterator:
                with torch.cuda.stream(compute_stream):
                    output_copy_finished.wait()
                    batch_copy_finished.wait()
                    out = model(current_X)
                    loss = loss_fn(out.squeeze(-1), current_y)
                    forward_finished.record() # 'forward' finished later to allow to loss -> cpu copy

                with torch.cuda.stream(copy_stream):
                    next_X_gpu = next_X.to(device, non_blocking=True)
                    batch_copy_finished.record()
                    metric_tracker.process_value(next_y, 'test_y')
                    next_y_gpu = next_y.to(device, non_blocking=True)
                    forward_finished.wait()
                    loss_cpu = loss.item()
                    if not torch.any(torch.isnan(next_X)):
                        metric_tracker.log_metric('test_loss', loss_cpu, 1)
                        metric_tracker.process_value(out, 'test_logits')
                        mlflow.log_metric('test_loss', loss_cpu, step=batch_i * config.train.batch_size)
                    output_copy_finished.record()
                    current_X = next_X_gpu.long()
                    current_y = next_y_gpu.long()
                    batch_i = next_batch_i

                example_progress.update(examples_done, advance=config.train.batch_size)

        _ = metric_tracker.calc_metrics(
            logit_store_name='test_logits',
            y_store_name='test_y',
            prefix='test'
        )

@app.command()
def main(
    data_path: VALID_PATHS = typer.Argument( 
        help='data path relative to config.data.staged'
    )
) -> None:
    " Main Loop "
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() and not TESTING else torch.device('cpu')
    logger.info(f'Running on {device}' + ' TESTING' if TESTING else '')

    model = get_model(config.model.model_name)(config.model, device, torch.float32)

    train_dataset = QuartileDataset(
        data_path=str(config.data.staged / data_path),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        t_start=config.data.train_start,
        t_end=config.data.train_end,
        max_len=500,
        pad=True,
        pad_value=0,
        testing=TESTING
    )
    test_dataset = QuartileDataset(
        data_path=str(config.data.staged / data_path),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        t_start=config.data.test_start,
        t_end=config.data.test_end,
        max_len=500,
        pad=True,
        pad_value=0,
        testing=TESTING
    )

    train_dataloader = init_dataloader(train_dataset)
    test_dataloader = init_dataloader(test_dataset)

    metric_tracker = ClassificationTracker(
        init_mtrack_params(device=device),
        dtype=torch.float32,
        device=device,
    )
    mlf_experiment, mlf_run = init_mlflow(
        model.MODEL_NAME,
        data_path,
    )

    executor = ThreadPoolExecutor(max_workers=2)
    epoch_progress, example_progress, mem_util_progress = init_progress() 
    epochs_done = epoch_progress.add_task('Epochs', total=config.train.epochs)
    examples_done = example_progress.add_task('Examples', total=len(train_dataloader) * config.train.batch_size)
    max_mem = 0
    if torch.cuda.is_available():
        _, max_mem = torch.cuda.mem_get_info()
    mem_used = mem_util_progress.add_task('Mem Util', total=max_mem* (1/(1024**2)))
    compute_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    forward_finished = torch.cuda.Event()
    batch_copy_finished = torch.cuda.Event()
    output_copy_finished = torch.cuda.Event()
    
    loss_fn = init_loss_fun()
    optimizer = init_optimizer(model)

    for epoch in range(1, config.train.epochs + 1):
        example_progress.update(examples_done, completed=0)
        train_iterator = plusN_iterator(iter(train_dataloader), extra_iters=1)
        with torch.cuda.stream(copy_stream):
            batch_i, current_X, current_y = next(train_iterator)
            metric_tracker.process_value(current_y, 'train_y')
            current_X = current_X.to(device, non_blocking=True).long()
            current_y = current_y.to(device, non_blocking=True).long()
            batch_copy_finished.record()
            output_copy_finished.record()

        for next_batch_i, next_X, next_y in train_iterator:
            with torch.cuda.stream(compute_stream):
                output_copy_finished.wait()
                batch_copy_finished.wait()
                out = model(current_X)
                loss = loss_fn(out.squeeze(-1), current_y)
                forward_finished.record() # 'forward' finished later to allow to loss -> cpu copy
                loss.backward()
                optimizer.step()
                optimizer.zero_grad() 

            with torch.cuda.stream(copy_stream):
                next_X_gpu = next_X.to(device, non_blocking=True)
                batch_copy_finished.record()
                metric_tracker.process_value(next_y, 'train_y')
                next_y_gpu = next_y.to(device, non_blocking=True)
                forward_finished.wait()
                loss_cpu = loss.item()
                if not torch.any(torch.isnan(next_X)):
                    metric_tracker.log_metric('train_loss', loss_cpu, 1)
                    metric_tracker.process_value(out, 'train_logits')
                    mlflow.log_metric('train_loss', loss_cpu, step=batch_i * config.train.batch_size)
                    executor.submit(isnan_async, loss_cpu)
                output_copy_finished.record()
                current_X = next_X_gpu.long()
                current_y = next_y_gpu.long()
                batch_i = next_batch_i

            example_progress.update(examples_done, advance=config.train.batch_size)
            mem_use, _ = torch.cuda.mem_get_info()
            mem_util_progress.update(mem_used, complete=mem_use * (1/(1024**2)))

        _ = metric_tracker.calc_metrics(
            logit_store_name='train_logits',
            y_store_name='train_y',
            prefix='train'
        )
       
        if epoch % config.train.eval_interval == 0:
            eval_model(
                model,
                loss_fn,
                test_dataloader,
                example_progress,
                examples_done,
                metric_tracker,
                compute_stream,
                copy_stream,
                device,
                    )
        if epoch & config.train.checkpoint_interval == 0:
            pass
            # checkpoint here

        metrics = metric_tracker.report(
            progress_bar=epoch_progress, 
            epoch=epoch,
        )
        # filter metrics logged in main loop
        metrics = {k:v for k,v in metrics.items() if 'loss' not in k}
        mlflow.log_metrics(metrics, step=epoch * config.train.batch_size * len(train_dataset))

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

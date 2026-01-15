#!/usr/bin/env python3
# train.py
from typing import Iterator
from config.config import config, Config, EXPERIMENT_NAME
from src.types.valid_paths import VALID_PATHS
from src.utils.logging import setup_logger
from src.metric_trackers.classification_tracker import ClassificationTracker
from src.datasets.quartile_dataset import QuartileDataset 
from src.models.registry import get_model

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import os
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.traceback import install
import typer
from pathlib import Path
from logging import getLogger
import mlflow
tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.end_run()

# 3. Enable Math Attention (The failsafe - slowest but guaranteed to work)
# env variables to spoof hipblaslt into working 
#HSA_OVERRIDE_GFX_VERSION=11.0.0 
#ROCBLAS_USE_HIPBLASLT=1 
#TORCH_BLAS_PREFER_HIPBLASLT=1
# logging verbosity
#PYTORCH_TUNABLEOP_ENABLED=1
#PYTORCH_TUNABLEOP_VERBOSE=1

torch.backends.cuda.matmul.allow_tf32 = False 
torch.set_float32_matmul_precision('high')

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

logger.info(f'Mlflow connection established at {tracking_uri}') 

app = typer.Typer(pretty_exceptions_enable=False)
install(
    show_locals=False,       # Turn off local variables (reduces noise)
    extra_lines=1,           # Minimize context lines around the error (default is 3)
    width=100,               # Prevents wrapping on wide screens for a cleaner look
    word_wrap=False
)

def isnan_async(loss: torch.float32):
    if torch.any(torch.isnan(loss)):
        raise ValueError('Loss is NaN, interrupting training')

def init_dataloader(
    dataset: Dataset[tuple[Tensor, ...] | Tensor],
    config: Config = config,
) -> DataLoader[tuple[Tensor, ...] | Tensor]:

    # TODO: add sample check here and integrate sampler

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=config.train.shuffle,
    )
    return dataloader

def get_params(params):
    """ Get params from dict, ignore clutter """
    ignore_keys = ['name', 'optimizer', 'last_epoch', 'step_count']
    return {k:v for k,v in params.items() if k not in ignore_keys and not str(k).startswith('_')}.items()

def get_scheduler_params(scheduler):
    """ Fetch and format scheduler (and sub-scheduler) parameters """
    params = {}
    params['name'] = type(scheduler).__name__
    if hasattr(scheduler, '_schedulers'):
        params['names'] = []
        params['milestones'] = scheduler._milestones
        for i, sub_scheduler in enumerate(scheduler._schedulers):
            params['names'].append(type(sub_scheduler).__name__)
            sub_params = get_scheduler_params(sub_scheduler)
            params.update({f'{i}-{sub_params["name"]}-{k}':v for k,v in get_params(sub_params)})
    else:
        params.update(get_params(scheduler.__dict__))

    return params

def get_optim_params(optimizer):
    """ Placeholder for optimizer param getter """
    # optim params currenty in config.train
    pass

def log_params(data_path, scheduler):
    param_dict = {}
    for k,v in config.__dict__.items():
        mod_params = {f'{k}-{sk}': val for sk, val in v.__dict__.items()}
        param_dict.update(mod_params)
    param_dict['data-raw'] = str(param_dict['data-raw']) + data_path
    scheduler_params = get_scheduler_params(scheduler)
    param_dict.update({f'scheduler-{k}':v for k,v in scheduler_params.items()})
    mlflow.log_params(param_dict)

def log_lrs(scheduler, step: int):
    lrs = {f'lr-{i}':v for i,v in enumerate(scheduler.get_last_lr())}
    mlflow.log_metrics(lrs, step=step)
   
def init_progress(disable=False) -> tuple[Progress, ...]:
    epoch_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TimeElapsedColumn(),
        TextColumn('<'), 
        TimeRemainingColumn(),
        speed_estimate_period=60.0 * 60.0 * 10, # hours
        disable = disable,
    )     
    example_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TimeElapsedColumn(),
        TextColumn('<'), 
        TimeRemainingColumn(),
        speed_estimate_period=60.0 * 30, # mins
        disable= disable,
    )   
    eval_example_progress = Progress(
        TextColumn('[bold blue] {task.description}', justify='left'),
        BarColumn(bar_width=40),
        TextColumn('[task.completed]{task.completed}/{task.total}'),
        TextColumn('[progress.percentage]{task.percentage:>3.0f}%'),
        TimeElapsedColumn(),
        TextColumn('<'), 
        TimeRemainingColumn(),
        speed_estimate_period=60.0 * 30, # mins
        disable= disable,
    )
    mem_util_progress = Progress(
        TextColumn('[yellow] {task.description}', justify='right'),
        BarColumn(bar_width=30),
        TextColumn('[task.completed]{task.completed:.1f}/{task.total:.1f} MiB'),
        disable= disable,
    )

    epoch_progress.start()
    example_progress.start()
    eval_example_progress.start()
    #mem_util_progress.start() # no report on strix halo
    return epoch_progress, example_progress, eval_example_progress, mem_util_progress

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
        buffer_size=1,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )
    train_y = StoreParams(
        'train_y',
        batch_shape=(config.train.batch_size, ),
        buffer_size=1,
        buffer_device=cpu,
        max_store=-1,
        n_examples=-1,
    )

    train_loss = StoreParams(
        'train_loss',
        batch_shape=(config.train.batch_size, ),
        buffer_size=1,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )

    test_logits = StoreParams(
        'test_logits',
        batch_shape=(config.train.batch_size, 5),
        buffer_size=1,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )

    test_y = StoreParams(
        'test_y',
        batch_shape=(config.train.batch_size, ),
        buffer_size=1,
        buffer_device=cpu,
        max_store=-1,
        n_examples=-1,
    )

    test_loss = StoreParams(
        'test_loss',
        batch_shape=(config.train.batch_size, ),
        buffer_size=1,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )
    return train_logits, train_y, train_loss, test_logits, test_y, test_loss


def plusN_iterator(iterator: Iterator[tuple[torch.Tensor, ...]], extra_iters: int) -> Iterator[tuple[torch.Tensor, ...]]:
    for i, entry in enumerate(iterator):
        yield i, *entry

    for _ in range(extra_iters):
        yield torch.nan, torch.tensor(float('nan')), torch.tensor(float('nan'))

def eval_model(
    step_offset:int,
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
        data_iterator = plusN_iterator(iter(dataloader), extra_iters=1)
        with torch.cuda.stream(copy_stream):
            batch_i, current_X, current_y = next(data_iterator)
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
                next_y_gpu = next_y.to(device, non_blocking=True)
                forward_finished.wait()
                loss_cpu = loss.detach().item()
                if not torch.any(torch.isnan(next_X)):
                    metric_tracker.log_metric('test_loss', loss_cpu, 1)
                    metric_tracker.process_values((current_y, out.detach()), ('test_y', 'test_logits'))
                    mlflow.log_metric('test_loss', loss_cpu, step = step_offset + (batch_i * config.train.batch_size))
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
        help='data path relative to config.data.staged'),
    artifact_path: str = '/home/fnoble/Dropbox/experiment-tracking/artifacts',
    compile: bool = False,
    testing: bool = False,
    progress: bool = True,
    gpu: bool = True,
) -> None:
    " Main Loop "

    from config.config import Loss_fn, Optimizer, init_lr_scheduler 
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu')
    logger.info(f'Running on {device}{" TESTING" if testing else ""}')

    model = get_model(config.model)(config.model, device, torch.float32)
    if compile:
        model.compile(fullgraph=False, mode='default')

    loss_fn = Loss_fn()
    optimizer = Optimizer(
        model.parameters(),
        lr = config.train.lr,
        weight_decay = config.train.weight_decay,
    )

    scheduler = init_lr_scheduler(optimizer)
    _ = log_lrs(scheduler, 0)

    train_dataset = QuartileDataset(
        data_path=str(config.data.staged / data_path),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        t_start=config.data.train_start,
        t_end=config.data.train_end,
        max_len=500,
        pad=True,
        pad_value=0,
        testing=testing
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
        testing=testing
    )

    examples_per_epoch = len(train_dataset) # update this for when sampling is introduced
    train_dataloader = init_dataloader(train_dataset)
    test_dataloader = init_dataloader(test_dataset)

    metric_tracker = ClassificationTracker(
        init_mtrack_params(device=device),
        dtype=torch.float32,
        device=device,
        buffer=False,
    )

    executor = ThreadPoolExecutor(max_workers=2)
    epoch_progress, example_progress, eval_example_progress, mem_util_progress = init_progress(disable = not progress) 
    epochs_done = epoch_progress.add_task('Epochs', total=config.train.epochs)
    examples_done = example_progress.add_task('Train Examples', total=len(train_dataloader) * config.train.batch_size)
    eval_examples_done = eval_example_progress.add_task('Eval Examples', total=len(test_dataloader) * config.train.batch_size)
    max_mem = 0
    if torch.cuda.is_available():
        _, max_mem = torch.cuda.mem_get_info()
    mem_used = mem_util_progress.add_task('Mem Util', total=max_mem* (1/(1024**2)))

    compute_stream = torch.cuda.Stream()
    copy_stream = torch.cuda.Stream()
    forward_finished = torch.cuda.Event()
    batch_copy_finished = torch.cuda.Event()
    output_copy_finished = torch.cuda.Event()

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model.MODEL_NAME, nested=True):
        log_params(data_path, scheduler)

        for epoch in range(1, config.train.epochs + 1):
            model.train()

            example_progress.reset(examples_done, description='Train examps', total=len(train_dataloader) * config.train.batch_size)
            eval_example_progress.reset(examples_done, description='Eval examps', total=len(test_dataloader) * config.train.batch_size)

            train_iterator = plusN_iterator(iter(train_dataloader), extra_iters=1)
            with torch.cuda.stream(copy_stream):
                batch_i, current_X, current_y = next(train_iterator)
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
                    next_y_gpu = next_y.to(device, non_blocking=True)
                    forward_finished.wait()
                    loss_cpu = loss.detach().item()
                    if not torch.any(torch.isnan(next_X)):
                        metric_tracker.log_metric('train_loss', loss_cpu, 1)
                        metric_tracker.process_values((current_y, out.detach()), ('train_y', 'train_logits'))
                        mlflow.log_metric('train_loss-batch', loss_cpu, step = (epoch-1) * examples_per_epoch + batch_i * config.train.batch_size)
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

            if epoch % config.train.checkpoint_interval == 0:
                save_dir = os.path.join(artifact_path, EXPERIMENT_NAME, str(mlf_run.info.run_id))
                save_path = os.path.join(save_dir, f'epoch-{epoch}.pt')
                os.makedirs(save_dir, exist_ok=True)
                checkpoint = model.state_dict()
                torch.save(checkpoint, save_path)
                mlflow.log_artifact(save_path)
                # save model state and run id, load each on restart (pass as option)
            
            _ = log_lrs(scheduler, epoch * examples_per_epoch)
            scheduler.step() 

            if epoch % config.train.eval_interval == 0:
                eval_model(
                    step_offset=(epoch-1) * examples_per_epoch,
                    model=model,
                    loss_fn=loss_fn,
                    dataloader=test_dataloader,
                    example_progress=eval_example_progress,
                    examples_done=eval_examples_done,
                    metric_tracker=metric_tracker,
                    compute_stream=compute_stream,
                    copy_stream=copy_stream,
                    device=device,
                        )

            metrics = metric_tracker.report(
                progress_bar=epoch_progress, 
                epoch=epoch,
            )
            # filter metrics logged in main loop
            mlflow.log_metrics(metrics, step = epoch, synchronous=False)

            epoch_progress.update(epochs_done, advance=1)

            metric_tracker.clear()

if __name__ == '__main__':
    app()

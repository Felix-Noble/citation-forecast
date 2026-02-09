#!/usr/bin/env python3
# train.py
from typing import Iterator
from config.config import config, Config, EXPERIMENT_NAME
from src.types.valid_paths import VALID_PATHS
from src.utils.logging import setup_logger
from src.metric_trackers.classification_tracker import ClassificationTracker
from src.datasets.tertiary_dataset import TertiaryDataset  as _dataset
from src.models.registry import get_model

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import time
import os
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.traceback import install
import typer
from pathlib import Path
from logging import getLogger
import mlflow
tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)

# 3. Enable Math Attention (The failsafe - slowest but guaranteed to work)
# env variables to spoof hipblaslt into working 
#HSA_OVERRIDE_GFX_VERSION=11.0.0 
#ROCBLAS_USE_HIPBLASLT=1 
#TORCH_BLAS_PREFER_HIPBLASLT=1
# logging verbosity
#PYTORCH_TUNABLEOP_ENABLED=1
#PYTORCH_TUNABLEOP_VERBOSE=1

#torch.backends.cuda.matmul.allow_tf32 = False 

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

def isnan_async(loss):
    if torch.any(torch.isnan(loss)):
        logger.error('Loss is NaN, interrupting training')
        time.sleep(5)
        raise ValueError('Loss is NaN, interrupting training')

def init_dataloader(
    dataset: Dataset[tuple[Tensor, ...] | Tensor],
    config: Config = config,
    sampler = None,
) -> DataLoader[tuple[Tensor, ...] | Tensor]:

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        shuffle=config.train.shuffle,
        sampler = sampler,
        drop_last=True,
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
        buffer_size=100,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )
    train_y = StoreParams(
        'train_y',
        batch_shape=(config.train.batch_size, ),
        buffer_size=100,
        buffer_device=cpu,
        max_store=-1,
        n_examples=-1,
    )

    train_loss = StoreParams(
        'train_loss',
        batch_shape=(config.train.batch_size, ),
        buffer_size=100,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )

    test_logits = StoreParams(
        'test_logits',
        batch_shape=(config.train.batch_size, 5),
        buffer_size=100,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )

    test_y = StoreParams(
        'test_y',
        batch_shape=(config.train.batch_size, ),
        buffer_size=100,
        buffer_device=cpu,
        max_store=-1,
        n_examples=-1,
    )

    test_loss = StoreParams(
        'test_loss',
        batch_shape=(config.train.batch_size, ),
        buffer_size=100,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )
    return train_logits, train_y, train_loss, test_logits, test_y, test_loss


def plusN_iterator(iterator: Iterator[tuple[torch.Tensor, ...]], extra_iters: int) -> Iterator[tuple[torch.Tensor, ...]]:
    for i, entry in enumerate(iterator):
        yield torch.tensor(i), *entry

    extra_entry = tuple(torch.tensor(float('nan')) for _ in range(len(tuple(entry)) + 1))
    for _ in range(extra_iters):
        yield extra_entry

def eval_model(
    model: nn.Module,
    loss_fn,
    dataloader: DataLoader,
    example_progress,
    examples_done,
    stream,
    metric_tracker: ClassificationTracker,
    device: torch.device,
) -> None:

    model.eval()
    with torch.no_grad():
        for batch_i, (X, y, mask) in enumerate(dataloader):
            metric_tracker.process_values((y,), ('test_y',))
            with torch.cuda.stream(stream):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
            torch.cuda.synchronize()

            out = model(X, mask)
            loss = loss_fn(out.squeeze(-1), y)

            #loss_cpu = loss.detach().item()
            #metric_tracker.log_metric('test_loss', loss_cpu, X.shape[0])
            metric_tracker.process_values((out.detach(), loss.detach()), ('test_logits', 'test_loss'))
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
    artifact_path: str = '/home/fnoble/experiment-tracking/artifacts',
    compile: bool = False,
    testing: bool = False,
    progress: bool = True,
    gpu: bool = True,
) -> None:
    " Main Loop "

    from config.config import Loss_fn, Optimizer, init_lr_scheduler 
    torch.set_float32_matmul_precision(config.train.mat_mul_precision)
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu')

    model = get_model(config.model)(config.model, device, config.model.dtype)
    logger.info(f'Running "{model.MODEL_NAME}" on {device}{" TESTING" if testing else ""}')
    if compile:
        model.compile(fullgraph=True, mode='default')

    loss_fn = Loss_fn()
    optimizer = Optimizer(
        model.parameters(),
        lr = config.train.lr,
        weight_decay = config.train.weight_decay,
    )

    scheduler = init_lr_scheduler(optimizer)

    train_dataset = _dataset(
        data_path=str(config.data.staged / data_path),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        t_start=config.data.train_start.toordinal(),
        t_end=config.data.train_end.toordinal(),
        max_len=config.data.max_len,
        pad=True,
        return_mask=True,
        pad_value=config.model.pad_token,
        testing=testing
    )
    test_dataset = _dataset(
        data_path=str(config.data.staged / data_path),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        t_start=config.data.test_start.toordinal(),
        t_end=config.data.test_end.toordinal(),
        max_len=config.data.max_len,
        pad=True,
        return_mask=True,
        pad_value=config.model.pad_token,
        testing=testing
    )
    class PortionSampler(torch.utils.data.Sampler):
        def __init__(self, data_source, num_samples):
            self.data_source = data_source
            self.num_samples = num_samples

        def __iter__(self):
            # Generate random indices for the whole dataset
            indices = torch.randperm(len(self.data_source))
            # Return only a slice of them
            return iter(indices[:self.num_samples].tolist())

        def __len__(self):
            return self.num_samples

    examples_per_epoch = len(train_dataset) # update this for when sampling is introduced
    train_sampler = None
    if config.train.sample:
        examples_per_epoch = config.train.sample
        train_sampler = PortionSampler(train_dataset, config.train.sample)

    train_dataloader = init_dataloader(train_dataset, sampler=train_sampler)
    test_dataloader = init_dataloader(test_dataset)
    n_batches = len(train_dataloader) 

    metric_tracker = ClassificationTracker(
        init_mtrack_params(device=device),
        dtype=torch.float32,
        device=device,
        buffer=False,
    )

    executor = ThreadPoolExecutor(max_workers=2)
    stream = torch.cuda.Stream()
    epoch_progress, example_progress, eval_example_progress, mem_util_progress = init_progress(disable = not progress) 
    epochs_done = epoch_progress.add_task('Epochs', total=config.train.epochs)
    examples_done = example_progress.add_task('Train Examples', total=len(train_dataloader) * config.train.batch_size)
    eval_examples_done = eval_example_progress.add_task('Eval Examples', total=len(test_dataloader) * config.train.batch_size)
    max_mem = 0
    if torch.cuda.is_available():
        _, max_mem = torch.cuda.mem_get_info()
    mem_used = mem_util_progress.add_task('Mem Util', total=max_mem* (1/(1024**2)))

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=model.MODEL_NAME, nested=True):

        mlf_run = mlflow.active_run()
        log_params(data_path, scheduler)

        for epoch in range(1, config.train.epochs + 1):
            model.train()

            example_progress.reset(examples_done, description='Train examps', total=len(train_dataloader) * config.train.batch_size)
            eval_example_progress.reset(examples_done, description='Eval examps', total=len(test_dataloader) * config.train.batch_size)

            for batch_i, (X, y, mask) in enumerate(train_dataloader):
                metric_tracker.process_values((y,), ('train_y',))
                with torch.cuda.stream(stream):
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                torch.cuda.synchronize()

                out = model(X, mask)
                loss = loss_fn(out.squeeze(-1), y)


                loss.backward()
                if batch_i % config.train.opttim_step_interval == 0 or batch_i == n_batches:
                    optimizer.step()
                    optimizer.zero_grad() 

                loss_cpu = loss.detach().cpu().item()
                metric_tracker.log_metric('train_loss', loss_cpu, X.shape[0])
                metric_tracker.process_values((out.detach(), ), ('train_logits', ))
                executor.submit(isnan_async, loss_cpu)
                mlflow.log_metric('train_loss-batch', loss_cpu, synchronous=False, step=int((epoch-1) * examples_per_epoch + batch_i * config.train.batch_size))

                example_progress.update(examples_done, advance=config.train.batch_size)
                #mem_use, _ = torch.cuda.mem_get_info()
                #mem_util_progress.update(mem_used, complete=mem_use * (1/(1024**2)))

            if epoch % config.train.checkpoint_interval == 0:
                save_dir = os.path.join(artifact_path, EXPERIMENT_NAME, str(mlf_run.info.run_id))
                save_path = os.path.join(save_dir, f'epoch-{epoch}.pt')
                os.makedirs(save_dir, exist_ok=True)
                checkpoint = model.state_dict()
                torch.save(checkpoint, save_path)
                mlflow.log_artifact(save_path)
                # save model state and run id, load each on restart (pass as option)
            
            _ = log_lrs(scheduler, epoch)
            scheduler.step() 

            if epoch % config.train.eval_interval == 0:
                eval_model(
                    model=model,
                    loss_fn=loss_fn,
                    dataloader=test_dataloader,
                    example_progress=eval_example_progress,
                    examples_done=eval_examples_done,
                    stream=stream,
                    metric_tracker=metric_tracker,
                    device=device,
                        )

            _ = metric_tracker.calc_metrics(
                logit_store_name='train_logits',
                y_store_name='train_y',
                prefix='train'
            )
            metrics = metric_tracker.report(
                progress_bar=epoch_progress, 
                epoch=epoch,
            )

            mlflow.log_metrics(metrics, step = epoch, synchronous=False)

            epoch_progress.update(epochs_done, advance=1)

            metric_tracker.clear()

if __name__ == '__main__':
    app()

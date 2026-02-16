#!/usr/bin/env python3
# train.py
from config import Config, config, env, Loss_fn, Optimizer
from src.types.valid_paths import VALID_PATHS
from src.utils.logging import setup_logger
from src.builders import build_ordinal_dataset, build_lr_scheduler, build_dataloader, build_tracker_params
from src.samplers import PortionSampler
from src.trackers import ClassificationTracker
from src.datasets import OrdinalDataset
from src.mlflow import log_params, log_lr
from src.callbacks import isnan_async
from src.eval import eval
from src.models.registry import get_model

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Iterator
from concurrent.futures import ThreadPoolExecutor
import os
from rich.traceback import install
import typer
from contextlib import nullcontext
from pathlib import Path
from logging import getLogger
import mlflow
mlflow.set_tracking_uri(env.TRACKING_URI)

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

logger.info(f'Mlflow connection established at {env.TRACKING_URI}') 

app = typer.Typer(pretty_exceptions_enable=False)

@app.command()
def main(
    run_name: str = typer.Option(
        '',
        '--name', '-n',
        help = 'MLflow run name'
        ),
    run_suffix: str = typer.Option(
        '',
        '--suffix', '-s',
        help = 'MLflow run suffix (to model name)'
        ),
    compile: bool=typer.Option(
        False,
        '--compile/--no-compile', '-c',
        help = 'compile model'
        ),
    dry_run: bool=typer.Option(
        False,
        '--dry-run', '--dry',
        help = 'Run through minimal samples for testing',
        ),
    progress: bool=typer.Option(
        True,
        '--progress/--no-progress',
        help = 'Show Epoch/Example progress bars',
        ),
    gpu: bool=typer.Option(
        True,
        '--gpu/--no-gpu',
        help = 'Use GPU as device'
        ),
) -> None:
    " Main Loop "
    assert not (run_name and run_suffix), "Either 'run-name' or 'run-suffix' must be specified"
    assert (run_name or run_suffix), "One of 'run-name' or 'run-suffix' must be specified"
    torch.set_float32_matmul_precision(config.train.mat_mul_precision)
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu')
    assert (device == 'cuda' or not gpu), 'No GPU available on this device, use --no-gpu option'

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
    metric_tracker = ClassificationTracker(
        build_tracker_params(device=device),
        dtype=torch.float32,
        device=device,
        buffer=False,
        )

    scheduler = build_lr_scheduler(optimizer)
    train_dataset, test_dataset = build_ordinal_datasets(dataset=config.train.dataset, dry_run=dry_run)

    examples_per_epoch = len(train_dataset) # update this for when sampling is introduced
    train_sampler = None
    if config.train.sample:
        examples_per_epoch = config.train.sample
        train_sampler = PortionSampler(train_dataset, config.train.sample)

    train_dataloader = build_dataloader(train_dataset, sampler=train_sampler)
    test_dataloader = build_dataloader(test_dataset)
    n_batches = len(train_dataloader) 

    executor = ThreadPoolExecutor(max_workers=2)

    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    stream_context = torch.cuda.stream(stream) if torch.cuda.is_available() else nullcontext()
    stream_sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda : None

    ( epoch_progress, 
      example_progress, 
      eval_example_progress, 
      mem_util_progress ) = build_progress_bars(disable = not progress) 

    epochs_done = epoch_progress.add_task('Epochs', total=config.train.epochs)
    examples_done = example_progress.add_task('Train Examples', total=len(train_dataloader) * config.train.batch_size)
    eval_examples_done = eval_example_progress.add_task('Eval Examples', total=len(test_dataloader) * config.train.batch_size)

    mlflow.set_experiment(env.EXPERIMENT)
    with mlflow.start_run(run_name=run_name):

        mlf_run = mlflow.active_run()
        log_params(config.train.dataset, scheduler)

        for epoch in range(1, config.train.epochs + 1):
            model.train()

            example_progress.reset(examples_done, description='Train examps', total=len(train_dataloader) * config.train.batch_size)
            eval_example_progress.reset(examples_done, description='Eval examps', total=len(test_dataloader) * config.train.batch_size)

            for batch_i, (X, y, mask) in enumerate(train_dataloader):
                metric_tracker.process_values((y,), ('train_y',))
                with stream_context:
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                stream_sync()

                out = model(X, mask)
                loss = loss_fn(out.squeeze(-1), y)

                loss.backward()
                if batch_i % config.train.opttim_step_interval == 0 or batch_i == n_batches:
                    optimizer.step()
                    optimizer.zero_grad() 

                loss_cpu = loss.detach().cpu().item()
                metric_tracker.log_metric('train_loss', loss_cpu, X.shape[0])
                metric_tracker.process_values((out.detach(), ), ('train_logits', ))
                executor.submit(isnan_async, loss_cpu, logger)
                mlflow.log_metric('train_loss-batch', loss_cpu, synchronous=False, step=int((epoch-1) * examples_per_epoch + batch_i * config.train.batch_size))

                example_progress.update(examples_done, advance=config.train.batch_size)

            if epoch % config.train.checkpoint_interval == 0:
                save_dir = os.path.join(env.ARTIFACT_LOC, env.EXPERIMENT, str(mlf_run.info.run_id))
                save_path = os.path.join(save_dir, f'epoch-{epoch}.pt')
                os.makedirs(save_dir, exist_ok=True)
                checkpoint = model.state_dict()
                torch.save(checkpoint, save_path)
                mlflow.log_artifact(save_path)
                # save model state and run id, load each on restart (pass as option)
            
            log_lrs(scheduler, epoch) 
            scheduler.step() 

            if epoch % config.train.eval_interval == 0:
                eval(
                    model=model,
                    loss_fn=loss_fn,
                    dataloader=test_dataloader,
                    example_progress=eval_example_progress,
                    examples_done=eval_examples_done,
                    stream_context=stream_context,
                    stream_sync=stream_sync,
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

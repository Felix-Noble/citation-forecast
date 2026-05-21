from config import config, env
from src.utils.logging import setup_logger
from src.builders import \
        build_dataset, \
        build_progress_bars, \
        build_lr_scheduler, \
        build_dataloader, \
        build_train_tracker, \
        build_model, \
        build_loss, \
        build_optimizer
from src.data import PortionSampler 
from src.data.datasets import BinaryCategoricalDataset, OrdinalDataset
from .eval import eval_model
from .callbacks import isnan_async
from .tracking import MetricTracker, BinaryClassificationTracker, log_params, log_lrs

import torch
import os
import typer
from contextlib import nullcontext
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import warnings
from logging import getLogger

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)
warnings.filterwarnings('ignore')

app = typer.Typer(pretty_exceptions_enable=False)

@app.callback(invoke_without_command=True)
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
    load_id: str = typer.Option(
        '',
        '--load-id',
        help='MLflow run id to load checkpoint from'
        ),
    load_epoch: int = typer.Option(
        False,
        '--load-epoch',
        help='Epoch to load checkpoint from'
        ),
    parent_id: str | None = typer.Option(
        None, 
        '--parent-id',
        help='MLflow run id to set as parent'
        ),
    start_epoch: int = typer.Option(
        False,
        '--start-epoch',
        help='Epoch to load checkpoint from'
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
    assert ((device == torch.device('cuda')) or not gpu), 'No GPU available on this device, use --no-gpu option'
    assert (load_id and load_epoch) or (not load_epoch and not load_id), 'load id/epoch only work together'
    model = build_model(device=device)
    start_epoch: int =  start_epoch if start_epoch else (load_epoch + 1 if load_epoch else 1) 
    TEMP_DIR: Path = Path('./temp/checkpoints') / load_id 

    if run_suffix:
        run_name = config.model.model_name + '-' + str(run_suffix)
    if dry_run:
        run_name += '-DRY'

    logger.info(f'Run: "{run_name}" (Model: {config.model.model_name}) | Device: {device}{" | DRY-RUN" if dry_run else ""}')
    if compile:
        model.compile(fullgraph=False, mode='default')

    loss_fn = build_loss()
    optimizer = build_optimizer(
        model.parameters(),
        lr = config.train.lr,
        weight_decay = config.train.weight_decay,
    )
    metric_tracker = build_train_tracker(
        dtype=torch.float32,
        device=device,
        config=config,
            )

    scheduler = build_lr_scheduler(optimizer)

    train_dataset = build_dataset(
        data_path=str(env.STAGED_LOC / config.train.train_dataset),
        dataset=config.train.dataset_class,
        X=['title_tokens', 'abstract_tokens'],
        y=['citation_normalized_percentile'],
        t_start=config.train.train_start,
        t_end=config.train.train_end,
        weights=config.train.loss.weights,
        config=config,
        max_len=config.model.max_len,
        return_mask=True,
        pad=True,
        dry_run=dry_run,
        name='train-dataset',
        auto_remove=True,
        **config.train.dataset_kwargs
    )

    test_dataset = build_dataset(
        data_path=str(env.STAGED_LOC / config.train.test_dataset),
        dataset=config.train.dataset_class,
        X=['title_tokens', 'abstract_tokens'],
        y=['citation_normalized_percentile'],
        t_start=config.train.test_start,
        t_end=config.train.test_end,
        config=config,
        max_len=config.model.max_len_eval,
        return_mask=True,
        pad=True,
        dry_run=dry_run,
        name='test-dataset',
        auto_remove=True,
        **config.train.dataset_kwargs
    )

    examples_per_epoch = len(train_dataset) # update this for when sampling is introduced
    train_sampler = None
    if config.train.sample:
        examples_per_epoch = config.train.sample
        train_sampler = PortionSampler(train_dataset, config.train.sample)

    train_dataloader = build_dataloader(train_dataset, shuffle=config.train.shuffle, sampler=train_sampler)
    test_dataloader = build_dataloader(test_dataset)
    n_batches = len(train_dataloader) 
    grad_accumulation_steps_gpu = torch.tensor(config.train.grad_accumulation_steps, device=device) 

    executor = ThreadPoolExecutor(max_workers=2)

    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    stream_context = torch.cuda.stream(stream) if torch.cuda.is_available() else nullcontext()
    stream_sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda : None

    ( epoch_progress, 
      example_progress, 
      eval_example_progress,) = build_progress_bars(disable = not progress) 

    import mlflow
    mlflow.set_tracking_uri(env.TRACKING_URI)

    from mlflow.tracking import MlflowClient
    client: MlflowClient = MlflowClient()
    mlflow.set_experiment(env.EXPERIMENT)
    logger.info(f'Mlflow connection established at {env.TRACKING_URI}') 
    if load_id and load_epoch:
        checkpoint_file = TEMP_DIR / f'epoch-{load_epoch}.pt'
        if checkpoint_file.exists():
            logger.info(f'Loading weights file (load_epoch: {load_epoch}) from {TEMP_DIR}')
        else:
            os.makedirs(TEMP_DIR, exist_ok=True)
            logger.info(f'Fetching weights file for (load_epoch: {load_epoch})')
            _ = client.download_artifacts(
                    load_id, 
                    str( f'epoch-{load_epoch}.pt' ),
                    str( TEMP_DIR  )
            )
        model.load_state_dict(
            torch.load(
                str( TEMP_DIR / f'epoch-{load_epoch}.pt' ),
                weights_only=True,
                map_location=device
                ),
            )
        logger.info('Model state loaded')

    with mlflow.start_run(run_name=run_name, parent_run_id=parent_id):
        mlflow.log_artifact('config/config.py')
        mlflow.log_artifact(model.filepath)
        mlf_run = mlflow.active_run()
        log_params(train_dataset, test_dataset, scheduler)

        epoch_progress.start()
        example_progress.start()
        eval_example_progress.start()

        epochs_done = epoch_progress.add_task('Epochs', total=config.train.epochs)
        examples_done = example_progress.add_task('Train Examples', total=len(train_dataloader) * config.train.batch_size)
        eval_examples_done = eval_example_progress.add_task('Eval Examples', total=len(test_dataloader) * config.train.batch_size)
        for epoch in range(start_epoch, start_epoch + config.train.epochs + 1):
            model.train()

            example_progress.reset(examples_done, description='Train examps', total=len(train_dataloader) * config.train.batch_size)
            eval_example_progress.reset(examples_done, description='Eval examps', total=len(test_dataloader) * config.train.batch_size)

            for batch_i, (X, y, mask, weight) in enumerate(train_dataloader):
                metric_tracker.process_values((y,), ('train_y',))
                with stream_context:
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                stream_sync()

                logits, probs, sigma = model(X, mask)
                loss = loss_fn(weight=weight, logits=logits, probs=probs, sigma=sigma, target=y)
                loss_cpu = loss.detach().item()
                sigma_cpu = torch.mean(sigma.detach()).item() 

                loss = loss / grad_accumulation_steps_gpu 

                loss.backward()
                if batch_i % config.train.grad_accumulation_steps == 0 or batch_i == n_batches:
                    optimizer.step()
                    optimizer.zero_grad() 

                metric_tracker.log_metric('train_loss', loss_cpu, X.shape[0])
                metric_tracker.log_metric('train_sigma', sigma_cpu, X.shape[0])
                metric_tracker.process_values((logits.detach(), probs.detach()), ('train_logits', 'train_probs'))
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
                eval_model(
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
                    prefix='test',
                    step=epoch,
                )

            _ = metric_tracker.calc_metrics(
                prefix='train',
                step=epoch,
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

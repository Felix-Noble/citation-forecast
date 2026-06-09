from config import Config, TrainConfig, config, env
from src.data import PortionSampler 
from src.utils.logging import setup_logger
from src.builders import \
        build_dataset, \
        build_eval_example_progress, \
        build_epoch_progress, \
        build_dataloader, \
        build_eval_tracker, \
        build_model, \
        build_loss \

from src.data.datasets import BinaryCategoricalDataset, OrdinalDataset
from src.training.eval import eval_model
from src.training.tracking import MetricTracker, BinaryClassificationTracker, log_params, log_lrs
import ast
import logging
from logging import getLogger
import copy
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
from typing import NamedTuple
import typer 
import os
import sys

class DateTimeVals(NamedTuple):
    year: int 
    month: int 
    day: int 

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

app = typer.Typer(pretty_exceptions_enable=False)

@app.callback(invoke_without_command=True)

def main(
        model_name: str = typer.Option(
            '',
            '--model-name', '-m',
            help='Model registry name'
            ),
        prefix: str = typer.Option(
            '',
            '--prefix',
            help='Prefix to mlflow run name'
            ),
        run_id: str = typer.Option(
            '',
            '--run-id', '-id',
            help='MLflow run ID to load model checkpoint from'
            ),
        epoch: str = typer.Option(
            '',
            '--epoch', '-e',
            help='Epoch to load checkopoint from'
            ),
        start_date: datetime | None = typer.Option(
            None, 
            '--start-date', '-s',
            help='Datetime to start eval from'
            ),
        end_date: datetime | None = typer.Option(
            None, 
            '--end-date', '-e',
            help='Datetime to end eval at'
            ),
        interval: int | None = typer.Option(
            None,
            '--interval', '-i',
            help='Interval to loop over successive time segments via'
            ),
        interval_unit: str = typer.Option(
            'y',
            '--interval-unit',
            help='Unit of interval quantity'
            ),
        experiment: str = typer.Option(
            '',
            '--experiment',
            help='MLflow experiment to load from, overwrites env var'
            ),
        dataset_path: str = typer.Option(
            '',
            '--dataset-path',
            help='Path to local dataset to load, overwrite config var',
            ),
        dataset: str = typer.Option(
            '',
            '--dataset',
            help='Name of dataset class to load, determins formatting of x/y columns'
            ),
        dataset_kwargs: str = typer.Option(
            '',
            '--dataset-kwargs',
            help='Key word agrs to pass to dataset construcor'
            ),
        x_columns: list[str] = typer.Option(
            [],
            '--x-column', '-x',
            help='Input columns to load from dataset'
            ),
        y_columns: list[str] = typer.Option(
            [],
            '--y-column', '-y',
            help='Target columns to load from dataset'
            ),
        tracking_uri: str = typer.Option(
            '',
            '--tracking-uri',
            help='MLflow tracking uri, overwrites env var'
            ),
        temp_dir: Path = typer.Option(
            './.temp/',
            '--temp-dir',
            help='Folder to store temp data in (model config/weights)'
            ),
        clean_up: bool = typer.Option(
                False,
                '--clean-up', 
                help='Delete temporary files after run completes'
                ),
        dry_run: bool = typer.Option(
                False,
                '--dry-run',
                help='Dry run with subset of dataset'
                ),
    ctx: typer.Context = None,
        ) -> None:
    # Required option checks 
    assert run_id, 'Provide a run id'
    assert epoch, 'Provide an epoch to load checkpoint from'
    assert start_date is not None, 'Provide a start date'
    assert interval is not None, 'Provide an interval'
    assert dataset, 'Privide a dataset formatting class'

    # Initialise environment
    assert start_date or not interval, 'Specify an interval with start_date'
    assert interval or not end_date, 'Specify an end_date with interval'

    EXPERIMENT: str = experiment if experiment else env.EXPERIMENT + '-EVAL'
    dataset_path: str = dataset_path if dataset_path else config.train.test_dataset
    dataset_kwargs: dict = ast.literal_eval(dataset_kwargs) if dataset_kwargs else ast.literal_eval(config.train.dataset_kwargs)
    TEMP_DIR: Path = temp_dir / run_id
    CHECKPOINT_DIR = TEMP_DIR / 'checkpoints'
    PREDICTIONS_DIR = TEMP_DIR / 'predictions'
    TRACKING_URI: str = tracking_uri if tracking_uri else env.TRACKING_URI
        # TODO assert that tracking URI is listening/connected
    t_delta_map = {
            'y': DateTimeVals(year=interval, month=0, day=0),
            } 
    assert interval_unit in t_delta_map.keys(), f'Interval unit must be one of: {list(t_delta_map.keys())}'
    T_DELTA: DateTimeVals = t_delta_map.get(interval_unit)
    window_progress_bar = build_epoch_progress()
    example_progress_bar = build_eval_example_progress()

    import warnings 
    warnings.filterwarnings('ignore')

    ## Initialise PyTorch
    import torch
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    stream_context = torch.cuda.stream(stream) if torch.cuda.is_available() else nullcontext()
    stream_sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda : None
    import torch._logging
    torch._logging.set_logs(all=logging.ERROR)


    logger.info(f'MLflow experiment: {EXPERIMENT}')
    
    # Initialise MLflow
    import mlflow
    mlflow.set_tracking_uri(TRACKING_URI)
    from mlflow.tracking import MlflowClient
    client: MlflowClient = MlflowClient()
    mlflow.set_experiment(EXPERIMENT)

    logger.info(f'MLflow tracking URI connected: {TRACKING_URI}')
    
    # Fetch model artifacts, TODO: add model arch file as artifact
    os.makedirs(CHECKPOINT_DIR, exist_ok=True) 
    if os.path.exists(CHECKPOINT_DIR / 'eval_config.py'):
        logger.info(f'Loading Config file from {TEMP_DIR}')
    else:
        logger.info('Fetching config file')
        client.download_artifacts(
                run_id, 
                str( 'config.py' ), 
                str( CHECKPOINT_DIR ),
                                  )
        os.rename(str( CHECKPOINT_DIR / 'config.py' ), str( CHECKPOINT_DIR / 'eval_config.py' ))
    if os.path.exists( CHECKPOINT_DIR / f'epoch-{epoch}.pt'):
        logger.info(f'Loading weights file (Epoch: {epoch}) from {TEMP_DIR}')
    else:
        logger.info(f'Fetching weights file for (Epoch: {epoch})')
        client.download_artifacts(
                run_id, 
                str( f'epoch-{epoch}.pt' ),
                str( TEMP_DIR  )
                                  )

    # Load model specific config
    sys.path.append(os.path.abspath(str(CHECKPOINT_DIR)))
    from eval_config import Config as EvalConfig
    eval_config = EvalConfig()
    model = build_model(device=device, config=eval_config)
    model.compile(mode='max-autotune')

    model.load_state_dict(
            torch.load(
                str( CHECKPOINT_DIR / f'epoch-{epoch}.pt' ),
                weights_only=True,
                map_location=device
                ),
            )
    model.eval()
    metric_tracker = build_eval_tracker(
            config=eval_config, 
            device=torch.device('cpu'), 
            dtype=torch.float32,
            )
    loss_fn = build_loss(config=eval_config)
    logger.info(f'Model {eval_config.model.model_name} loaded to {device}')
    logger.info(f'Starting temporal iteration: from ... to... interval...') 

    current_t_start: datetime = start_date
    with mlflow.start_run(run_name=f'{prefix}{dataset_path}-{run_id}'):

        mlflow.log_params(ctx.params)
        while True:

            current_t_end: datetime = datetime(
                    current_t_start.year + T_DELTA.year,
                    current_t_start.month + T_DELTA.month,
                    current_t_start.day + T_DELTA.day,
                                       )
            if end_date is not None and current_t_end > end_date:
               current_t_end = end_date 
            
            logger.debug(f'Windwo from - {current_t_start} to - {current_t_end}')
            window_config = copy.deepcopy(config)
            window_config_args = config.train.__dict__ 
            window_config_args['test_start']  = current_t_start
            window_config_args['test_end']  = current_t_end
            window_config_args['batch_size'] = config.train.batch_size
            window_config.train = TrainConfig(**window_config_args)
            
            metric_tracker.export = True
            metric_tracker.export_loc = PREDICTIONS_DIR / f'{window_config.train.test_start.year}'

            test_dataset = build_dataset(
                    data_path=str(env.STAGED_LOC / dataset_path),
                    dataset=dataset,
                    X=x_columns,
                    y=y_columns,
                    t_start=config.train.test_start,
                    t_end=config.train.test_end,
                    max_len=config.model.max_len_eval,
                    id_col='id',
                    return_id=True,
                    config=eval_config,
                    return_mask=True,
                    pad=True,
                    dry_run=dry_run,
                    name='eval-dataset',
                    auto_remove=True,
                    **dataset_kwargs
                    )

            sampler = None
            if config.train.sample:
                sampler = PortionSampler(test_dataset, config.train.sample)

            test_dataloader = build_dataloader(
                    dataset=test_dataset,
                    config=window_config,
                    sampler=sampler
                    )
               
            example_progress_bar.start()
            example_progress = example_progress_bar.add_task('Examples', total=len(test_dataloader) * window_config.train.batch_size)
            eval_model(
                    model=model,
                    loss_fn=loss_fn,
                    dataloader=test_dataloader,
                    example_progress=example_progress_bar,
                    examples_done=example_progress,
                    stream_context=stream_context,
                    stream_sync=stream_sync,
                    metric_tracker=metric_tracker,
                    device=device,
                    config=config,
                       )
            _ = metric_tracker.calc_metrics(
                prefix='test',
                step=current_t_start.year,
            )
            
            metrics = metric_tracker.report(
                    progress_bar=example_progress_bar,
                    epoch=current_t_start,
                    )
            mlflow.log_metrics(
                    metrics, 
                    step=current_t_start.year,
                    timestamp=current_t_start.year,
                    synchronous=False,
                               )
            metric_tracker.clear()
            example_progress_bar.reset(example_progress, description='Examples ', total=len(test_dataset))

            current_t_start = current_t_end 
            if end_date is not None and current_t_start >= end_date:
                break

    # ingest dataset overwrite argument
    # loop over start/end time via interval
        # build datasets/dataloader 
        # Track via experiment with -EVAl suffix, post results with datetime objects as 'step' and 'timestamp'

    logger.info('Eval Finished')



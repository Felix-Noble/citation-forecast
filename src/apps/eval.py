import copy
import logging
import os
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import NamedTuple

import typer

import config
from builders import (
    build_epoch_progress,
    build_eval_example_progress,
    build_eval_tracker,
    build_loss,
)
from config import env
from data import PortionSampler
from data.dataloaders import DataLoader
from data.sources import LocalStagedSource
from eval import eval_model
from training.tracking import (
    log_lrs,
    log_params,
)
from utils.logging import setup_logger


class DateTimeVals(NamedTuple):
    year: int
    month: int
    day: int


logger = getLogger(__name__)
_ = setup_logger(logger)

app = typer.Typer(pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def main(
    prefix: str = typer.Option("", "--prefix", help="Prefix to mlflow run name"),
    run_id: str = typer.Option(
        "", "--run-id", "-id", help="MLflow run ID to load model checkpoint from"
    ),
    epoch: str = typer.Option(
        "", "--epoch", "-e", help="Epoch to load checkopoint from"
    ),
    start_date: datetime | None = typer.Option(
        None, "--start-date", "-s", help="Datetime to start eval from"
    ),
    end_date: datetime | None = typer.Option(
        None, "--end-date", "-e", help="Datetime to end eval at"
    ),
    interval: int | None = typer.Option(
        None,
        "--interval",
        "-i",
        help="Interval to loop over successive time segments via",
    ),
    interval_unit: str = typer.Option(
        "y", "--interval-unit", help="Unit of interval quantity"
    ),
    experiment: str = typer.Option(
        "", "--experiment", help="MLflow experiment to load from, overwrites env var"
    ),
    dataset_path: str = typer.Option(
        "",
        "--dataset-path",
        help="Path to local dataset to load, overwrite config var",
    ),
    tracking_uri: str = typer.Option(
        "", "--tracking-uri", help="MLflow tracking uri, overwrites env var"
    ),
    temp_dir: Path = typer.Option(
        "./.temp/",
        "--temp-dir",
        help="Folder to store temp data in (model config/weights)",
    ),
    clean_up: bool = typer.Option(
        False, "--clean-up", help="Delete temporary files after run completes"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Dry run with subset of dataset"
    ),
    ctx: typer.Context = None,
) -> None:
    # Required option checks
    assert run_id, "Provide a run id"
    assert epoch, "Provide an epoch to load checkpoint from"
    assert start_date is not None, "Provide a start date"
    assert interval is not None, "Provide an interval"

    # Initialise environment
    assert start_date or not interval, "Specify an interval with start_date"
    assert interval or not end_date, "Specify an end_date with interval"

    EXPERIMENT: str = experiment if experiment else env.EXPERIMENT + "-EVAL"
    TEMP_DIR: Path = temp_dir / run_id
    CHECKPOINT_DIR = TEMP_DIR / "checkpoints"
    PREDICTIONS_DIR = env.STAGED_LOC / "eval" / run_id / f"run-{prefix}"
    if PREDICTIONS_DIR.exists():
        logger.warning(f"Removing data in {PREDICTIONS_DIR}")
        time.sleep(5)
        shutil.rmtree(PREDICTIONS_DIR)
    os.makedirs(PREDICTIONS_DIR)
    TRACKING_URI: str = tracking_uri if tracking_uri else env.TRACKING_URI
    # TODO assert that tracking URI is listening/connected
    t_delta_map = {
        "y": DateTimeVals(year=interval, month=0, day=0),
    }
    assert interval_unit in t_delta_map.keys(), (
        f"Interval unit must be one of: {list(t_delta_map.keys())}"
    )
    T_DELTA: DateTimeVals = t_delta_map.get(interval_unit)
    window_progress_bar = build_epoch_progress()
    example_progress_bar = build_eval_example_progress()

    import warnings

    warnings.filterwarnings("ignore")

    ## Initialise PyTorch
    import torch

    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    stream_context = (
        torch.cuda.stream(stream) if torch.cuda.is_available() else nullcontext()
    )
    stream_sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None
    import torch._logging

    torch._logging.set_logs(all=logging.ERROR)

    logger.info(f"MLflow experiment: {EXPERIMENT}")

    # Initialise MLflow
    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    from mlflow.tracking import MlflowClient

    client: MlflowClient = MlflowClient()
    mlflow.set_experiment(EXPERIMENT)

    logger.info(f"MLflow tracking URI connected: {TRACKING_URI}")

    # Fetch model artifacts, TODO: add model arch file as artifact
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(CHECKPOINT_DIR / "eval_config.py") or True:
        logger.info(f"Loading Config file from {CHECKPOINT_DIR}")
    else:
        logger.info("Fetching config file")
        client.download_artifacts(
            run_id,
            str("config.py"),
            str(CHECKPOINT_DIR),
        )
        os.rename(
            str(CHECKPOINT_DIR / "config.py"), str(CHECKPOINT_DIR / "eval_config.py")
        )
    if os.path.exists(CHECKPOINT_DIR / f"epoch-{epoch}.pt"):
        logger.info(
            f"Loading weights file (Epoch: {epoch}) from {CHECKPOINT_DIR}",
        )
    else:
        logger.info(f"Fetching weights file for (Epoch: {epoch})")
        client.download_artifacts(
            run_id,
            str(f"epoch-{epoch}.pt"),
            str(
                CHECKPOINT_DIR,
            ),
        )

    # Load model specific config
    sys.path.append(os.path.abspath(str(CHECKPOINT_DIR)))
    # from eval_config import Config as EvalConfig

    # eval_config = EvalConfig()
    eval_config = config
    model = config.model.clss(
        config=config.model.model,
        device=device,
        dtype=config.model.dtype,
    )
    model.compile(mode="max-autotune")

    model.load_state_dict(
        torch.load(
            str(CHECKPOINT_DIR / f"epoch-{epoch}.pt"),
            weights_only=True,
            map_location=device,
        ),
    )
    model.eval()
    metric_tracker = build_eval_tracker(
        config=eval_config,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    loss_fn = build_loss(config=eval_config)
    logger.info(f"Model {model.__module__} loaded to {device}")

    evaluator_class = config.train.evaluator
    evaluator = evaluator_class(
        evaluator_class.config(
            model=model,
            prefix="val",
            loss_fn=loss_fn,
            tracker=metric_tracker,
            stream=stream_context,
            device=device,
        )
    )
    current_t_start: datetime = start_date
    with mlflow.start_run(
        run_name=f"{prefix}-{model.__module__.split('.')[-1]}-{dataset_path}-{run_id}"
    ):
        mlflow.log_params(ctx.params)
        while True:
            current_t_end: datetime = datetime(
                current_t_start.year + T_DELTA.year,
                current_t_start.month + T_DELTA.month,
                current_t_start.day + T_DELTA.day,
            )
            if end_date is not None and current_t_end > end_date:
                current_t_end = end_date

            window_config = copy.deepcopy(config.data.test.dataset)
            window_config.t_start = current_t_start
            window_config.t_end = current_t_end
            logger.debug(
                f"Windwo from - {window_config.t_start} to - {window_config.t_end}"
            )

            metric_tracker.export = True
            metric_tracker.export_loc = (
                PREDICTIONS_DIR / f"{window_config.t_start.year}"
            )

            test_source = LocalStagedSource(
                path=config.env.STAGED_LOC,
                name=window_config.loc,
            )
            test_dataset = config.data.test.clss(
                config=window_config,
                source=test_source,
            )

            test_dataloader = DataLoader(
                dataset=test_dataset,
                config=config.data.test.loader,
            )

            example_progress_bar.start()
            example_progress = example_progress_bar.add_task(
                "Examples",
                total=len(test_dataset)
                if config.data.test.loader.samples is None
                else config.data.test.loader.samples,
            )

            for batch_i, batch in enumerate(test_dataloader):
                _ = evaluator.step(batch)
                example_progress_bar.update(
                    example_progress, advance=config.data.test.loader.batch_size
                )
            _ = metric_tracker.calc_metrics(
                prefix="val",
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
            mlflow.log_metric("examples", len(test_dataset), step=current_t_start.year)
            metric_tracker.clear()
            example_progress_bar.reset(
                example_progress, description="Examples ", total=len(test_dataset)
            )

            current_t_start = current_t_end
            if end_date is not None and current_t_start >= end_date:
                break

    # ingest dataset overwrite argument
    # loop over start/end time via interval
    # build datasets/dataloader
    # Track via experiment with -EVAl suffix, post results with datetime objects as 'step' and 'timestamp'

    logger.info("Eval Finished")

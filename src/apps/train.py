import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
import typer
from torch.utils.data import Dataset

import config
from builders import (
    build_loss,
    build_lr_scheduler,
    build_optimizer,
    build_progress_bars,
    build_train_tracker,
)
from config import env
from data import PortionSampler
from data.dataloaders import DataLoader
from training.callbacks import isnan_async
from training.tracking import (
    BinaryClassificationTracker,
    MetricTracker,
    calc_metrics,
    log_lrs,
    log_params,
)
from utils import get_root_dir
from utils.logging import setup_logger

from .eval import eval_model

logger = getLogger(__name__)
_ = setup_logger(logger)
warnings.filterwarnings("ignore")

app = typer.Typer(pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def main(
    run_name: str = typer.Option(
        "",
        "--name",
        "-n",
        help="MLflow run name",
    ),
    run_suffix: str = typer.Option(
        "",
        "--suffix",
        "-s",
        help="MLflow run suffix (to model name)",
    ),
    compile: str = typer.Option(
        "",
        "--compile",
        "-c",
        help="compile model",
    ),
    fullgraph: bool = typer.Option(
        False,
        "--fullgraph",
        help="Require fullgraph during compilation",
    ),
    load_id: str = typer.Option(
        "",
        "--load-id",
        help="MLflow run id to load checkpoint from",
    ),
    load_epoch: int = typer.Option(
        False,
        "--load-epoch",
        help="Epoch to load checkpoint from",
    ),
    parent_id: str | None = typer.Option(
        None,
        "--parent-id",
        help="MLflow run id to set as parent",
    ),
    start_epoch: int = typer.Option(
        False,
        "--start-epoch",
        help="Epoch to load checkpoint from",
    ),
    subsample: int | None = typer.Option(
        None,
        help="Sub sample of dataset for testing",
    ),
    progress: bool = typer.Option(
        True,
        "--progress/--no-progress",
        help="Show Epoch/Example progress bars",
    ),
    gpu: bool = typer.Option(
        True,
        "--gpu/--no-gpu",
        help="Use GPU as device",
    ),
) -> None:
    "Main Loop"
    assert not (run_name and run_suffix), (
        "Either 'run-name' or 'run-suffix' must be specified"
    )
    assert run_name or run_suffix, "One of 'run-name' or 'run-suffix' must be specified"
    torch.set_float32_matmul_precision(config.cuda.mat_mul_precision)
    device: torch.device = (
        torch.device("cuda")
        if torch.cuda.is_available() and gpu
        else torch.device("cpu")
    )
    assert (device == torch.device("cuda")) or not gpu, (
        "No GPU available on this device, use --no-gpu option"
    )
    assert (load_id and load_epoch) or (not load_epoch and not load_id), (
        "load id/epoch only work together"
    )

    model = config.model.clss(
        config=config.model.model,
        device=device,
        dtype=config.model.dtype,
    )
    start_epoch: int = (
        start_epoch if start_epoch else (load_epoch + 1 if load_epoch else 1)
    )
    TEMP_DIR: Path = Path("./temp/checkpoints") / load_id
    model_name = config.model.clss.__name__
    if run_suffix:
        run_name: str = model_name + "-" + str(run_suffix)
    if subsample:
        run_name += "-DRY"

    logger.info(
        f'Run: "{run_name}" (Model: {model_name}) | Device: {device}{" | DRY-RUN" if subsample else ""}'
    )
    if compile:
        model.compile(fullgraph=fullgraph, mode=compile)  # type: ignore

    loss_fn = build_loss(config)
    optimizer = build_optimizer(
        model.parameters(),  # type: ignore
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        config=config,
    )
    metric_tracker = build_train_tracker(
        dtype=torch.float32,
        device=device,
        config=config,
    )

    scheduler = build_lr_scheduler(optimizer, config)

    train_dataset = config.data.train.clss(
        config=config.data.train.dataset,
        env=config.env,
    )
    test_dataset = config.data.test.clss(
        config=config.data.test.dataset,
        env=config.env,
    )

    if config.data.train.loader.samples is None:
        train_examples_per_epoch = len(train_dataset)
    else:
        train_examples_per_epoch = config.data.train.loader.shuffle

    if config.data.test.loader.samples is None:
        test_examples_per_epoch = len(test_dataset)
    else:
        test_examples_per_epoch = config.data.test.loader.shuffle

    train_dataloader = DataLoader(
        dataset=train_dataset,
        config=config.data.train.loader,
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        config=config.data.test.loader,
    )

    n_batches = len(train_dataloader)
    grad_accumulation_steps_gpu = torch.tensor(
        config.train.accumulation_steps, device=device
    )

    executor = ThreadPoolExecutor(max_workers=2)

    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    stream_context = (
        torch.cuda.stream(stream) if torch.cuda.is_available() else nullcontext()
    )
    stream_sync = torch.cuda.synchronize if torch.cuda.is_available() else lambda: None

    (
        epoch_progress,
        example_progress,
        eval_example_progress,
    ) = build_progress_bars(disable=not progress)

    trainer_class = config.train.trainer
    trainer_config = trainer_class.config(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        tracker=metric_tracker,
        stream=stream_context,
        device=device,
        accumulation_steps=config.train.accumulation_steps,
        examples_per_epoch=train_examples_per_epoch,
    )
    trainer = trainer_class(trainer_config)
    import mlflow

    mlflow.set_tracking_uri(env.TRACKING_URI)
    from mlflow.tracking import MlflowClient

    client: MlflowClient = MlflowClient()
    mlflow.set_experiment(env.EXPERIMENT)
    logger.info(f"Mlflow connection established at {env.TRACKING_URI}")
    if load_id and load_epoch:
        checkpoint_file = TEMP_DIR / f"epoch-{load_epoch}.pt"
        if checkpoint_file.exists():
            logger.info(
                f"Loading weights file (load_epoch: {load_epoch}) from {TEMP_DIR}"
            )
        else:
            os.makedirs(TEMP_DIR, exist_ok=True)
            logger.info(f"Fetching weights file for (load_epoch: {load_epoch})")
            _ = client.download_artifacts(
                load_id, str(f"epoch-{load_epoch}.pt"), str(TEMP_DIR)
            )
        model.load_state_dict(
            torch.load(
                str(TEMP_DIR / f"epoch-{load_epoch}.pt"),
                weights_only=True,
                map_location=device,
            ),
        )
        logger.info("Model state loaded")

    with mlflow.start_run(run_name=run_name, parent_run_id=parent_id):
        # mlflow.log_artifact("./config")
        model_file = (
            get_root_dir()
            / "src"
            / (f"{model.__module__}".replace(".", os.sep) + ".py")
        )

        mlflow.log_artifact(str(model_file))
        mlf_run = mlflow.active_run()
        # log_params(train_dataset, test_dataset, scheduler, config)

        epoch_progress.start()
        example_progress.start()
        eval_example_progress.start()

        epochs_done = epoch_progress.add_task("Epochs", total=config.train.epochs)
        examples_done = example_progress.add_task(
            "Train Examples", total=train_examples_per_epoch
        )
        eval_examples_done = eval_example_progress.add_task(
            "Eval Examples", total=test_examples_per_epoch
        )
        for epoch in range(start_epoch, start_epoch + config.train.epochs + 1):
            trainer.start_epoch(epoch)
            log_lrs(scheduler, epoch)
            model.train()

            example_progress.reset(
                examples_done,
                description="Train examps",
                total=len(train_dataloader) * config.train.batch_size,
            )
            eval_example_progress.reset(
                examples_done,
                description="Eval examps",
                total=len(test_dataloader) * config.train.batch_size,
            )

            for batch_i, batch in enumerate(train_dataloader):
                loss = trainer.step(batch)

                # metric_tracker.process_values((batch.y.clone(),), ('train_y',))
                #                with stream_context:
                #                    x = batch.x.to(device, non_blocking=True)
                #                    y = batch.y.to(device, non_blocking=True)
                #                    mask = batch.mask.to(device, non_blocking=True)
                #                stream_sync()
                #
                #                torch.compiler.cudagraph_mark_step_begin()
                #                logits, probs = model.forward(x, mask)
                #                loss = loss_fn(
                #                    weight=batch.weight, logits=logits, probs=probs, target=y
                #                )
                #
                #                loss_cpu = loss.detach().clone()
                #                loss = loss / grad_accumulation_steps_gpu
                #
                #                loss.backward()
                #                if (
                #                    batch_i + 1
                #                ) % config.train.grad_accumulation_steps == 0 or batch_i == n_batches:
                #                    optimizer.step()
                #                    optimizer.zero_grad(set_to_none=True)
                #
                #                loss_cpu = loss_cpu.item()
                #                metric_tracker.log_metric("train_loss", loss_cpu, x.shape[0])
                # metric_tracker.process_values((logits.detach().clone(), probs.detach().clone()), ('train_logits', 'train_probs'))

                #                executor.submit(isnan_async, loss, logger)
                #                mlflow.log_metric(
                #                    "train_loss-batch",
                #                    loss,
                #                    synchronous=False,
                #                    step=int(
                #                        (epoch - 1) * examples_per_epoch
                #                        + batch_i * config.train.batch_size
                #                    ),
                #                )
                #
                #                metrics: dict[str, torch.Tensor] = calc_metrics(
                #                    config=config, probs=probs, targets=y
                #                )
                #                for k, v in metrics.items():
                #                    metric_tracker.log_metric(f"train_{k}", v.item(), x.shape[0])
                #
                #                del loss, metrics, probs
                #
                example_progress.update(examples_done, advance=config.train.batch_size)

            if epoch % config.train.checkpoint_interval == 0:
                save_dir = os.path.join(
                    env.ARTIFACT_LOC, env.EXPERIMENT, str(mlf_run.info.run_id)
                )
                save_path = os.path.join(save_dir, f"epoch-{epoch}.pt")
                os.makedirs(save_dir, exist_ok=True)
                checkpoint = model.state_dict()
                torch.save(checkpoint, save_path)
                mlflow.log_artifact(save_path)
                # save model state and run id, load each on restart (pass as option)

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
                    prefix="test",
                    step=epoch,
                )
            _ = metric_tracker.calc_metrics(
                prefix="train",
                step=epoch,
            )

            metrics = metric_tracker.report(
                progress_bar=epoch_progress,
                epoch=epoch,
            )
            mlflow.log_metrics(metrics, step=epoch, synchronous=False)

            epoch_progress.update(epochs_done, advance=1)

            metric_tracker.clear()


if __name__ == "__main__":
    app()

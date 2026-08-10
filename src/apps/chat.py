import ast
import copy
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import NamedTuple

import torch
import typer
from transformers import AutoTokenizer

from builders import (
    build_dataloader,
    build_dataset,
    build_epoch_progress,
    build_eval_example_progress,
    build_eval_tracker,
    build_loss,
    build_model,
)
from data import PortionSampler
from data.datasets import BinaryCategoricalDataset, OrdinalDataset
from training.tracking import (
    BinaryClassificationTracker,
    MetricTracker,
    log_lrs,
    log_params,
)
from utils.logging import setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)

app = typer.Typer(pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def main(
    run_id: str = typer.Option(
        "", "--run-id", "-id", help="MLflow run ID to load model checkpoint from"
    ),
    epoch: str = typer.Option(
        "", "--epoch", "-e", help="Epoch to load checkopoint from"
    ),
    max_response_len: int = typer.Option(
        30, help="Maximum response length to a prompt"
    ),
    tokenizer_repo: str = typer.Option(
        "openai/gpt-oss-120b",
        help="Tokenizer",
    ),
    experiment: str = typer.Option(
        "",
        help="MLflow experiment",
    ),
    tracking_uri: str = typer.Option(
        "",
        help="MLflow tracking_uri",
    ),
    temp_dir: Path = typer.Option(
        "./.temp/",
        "--temp-dir",
        help="Folder to store temp data in (model config/weights)",
    ),
    ctx: typer.Context = None,
):
    TEMP_DIR: Path = temp_dir / run_id
    CHECKPOINT_DIR: Path = TEMP_DIR / "checkpoints"
    TRACKING_URI: str = tracking_uri if tracking_uri else env.TRACKING_URI
    EXPERIMENT: str = experiment if experiment else env.EXPERIMENT + "-EVAL"

    device: torch.device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    import mlflow

    mlflow.set_tracking_uri(TRACKING_URI)
    client: mlflow.MlflowClient = mlflow.MlflowClient()
    mlflow.set_experiment(EXPERIMENT)

    logger.info(f"MLflow tracking URI connected: {TRACKING_URI}")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(CHECKPOINT_DIR / "eval_config.py"):
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
        logger.info(f"Loading weights file (Epoch: {epoch}) from {CHECKPOINT_DIR}")
    else:
        logger.info(f"Fetching weights file for (Epoch: {epoch})")
        client.download_artifacts(run_id, str(f"epoch-{epoch}.pt"), str(CHECKPOINT_DIR))

    # Load model specific config
    sys.path.append(os.path.abspath(str(CHECKPOINT_DIR)))
    from eval_config import Config as EvalConfig

    eval_config = EvalConfig()
    model = build_model(device=device, config=eval_config)
    model.compile(mode="max-autotune")

    model.load_state_dict(
        torch.load(
            str(CHECKPOINT_DIR / f"epoch-{epoch}.pt"),
            weights_only=True,
            map_location=device,
        ),
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_repo,
        add_eos_token=True,
        add_bos_token=True,
        use_fast=False,
    )
    if not os.path.exists(tokenizer_repo):
        tokenizer.save_pretrained(tokenizer_repo)

    while True:
        prompt = input(">>")
        if prompt == "q":
            quit()
        tokens = tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
            truncation=False,
            padding=False,
        )["input_ids"].to(device)
        with torch.no_grad():
            out_tokens = model.generate(tokens, max_response_len)
        out = tokenizer.decode(out_tokens)
        print(f"---\n{''.join(out)}\n---")

import copy
import logging
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import NamedTuple

import polars as pl
import typer

from builders import (
    build_dataloader,
    build_datasets,
    build_epoch_progress,
    build_eval_example_progress,
    build_eval_tracker,
    build_loss,
    build_model,
)
from data.datasets import BinaryCategoricalDataset, OrdinalDataset
from eval import eval_model
from training.tracking import (
    BinaryClassificationTracker,
    MetricTracker,
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
    dataset: str = typer.Option("", "--dataset", "-d", help="Dataset name"),
    start_time: datetime | None = typer.Option(
        None, "--start-time", "-s", help="Start date/time to filter by, inlcusive"
    ),
    end_time: datetime | None = typer.Option(
        None, "--end-time", "-e", help="End date/time to filter by, exclusive"
    ),
    time_col: str = typer.Option(
        "publication_date", "--time-col", "-tc", help="Date/Time col in dataset."
    ),
    describe_cols: list[str] = typer.Option(
        [], "--column", "-c", help="Columns to describe"
    ),
    buckets: list[float] = typer.Option(
        [],
        "--bucket",
        "-b",
        help="Borders of buckets to describe count by, creates list of inclusive upper limits",
    ),
):
    lf = pl.scan_parquet(list((env.STAGED_LOC / dataset).glob("*.par*")))

    if start_time is not None:
        lf = lf.filter(pl.col(time_col) >= start_time)
    if end_time is not None:
        lf = lf.filter(pl.col(time_col) < end_time)

    n_buckets = len(buckets)
    for col in describe_cols:
        bucket_counts = {
            f"{buckets[i]}-{buckets[i + 1]}": None for i in range(n_buckets - 1)
        }
        weights = {k: v for k, v in bucket_counts.items()}

        for i in range(n_buckets - 1):
            bucket_counts[f"{buckets[i]}-{buckets[i + 1]}"] = (
                lf.filter((pl.col(col) >= buckets[i]) & (pl.col(col) < buckets[i + 1]))
                .select(pl.len())
                .collect(engine="streaming")
                .item()
            )
        total = sum([v for v in bucket_counts.values()])
        for k in weights.keys():
            weights[k] = total / ((n_buckets - 1) * bucket_counts[k])

        print()
        logger.info(f"Describing {col}")
        logger.info("counts: ")

        logger.info(bucket_counts)
        logger.info("weights: ")
        logger.info(weights)

        description = lf.select(pl.col(col)).describe()
        logger.info(description)

    logger.info("Finished Describe")

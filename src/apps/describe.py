from datetime import datetime
from logging import getLogger
from typing import NamedTuple

import polars as pl
import typer

import config
from config import env
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
    filter: bool = typer.Option(False),
):
    lf = pl.scan_parquet(list((env.STAGED_LOC / dataset).glob("*.par*")))

    if start_time is not None:
        lf = lf.filter(pl.col(time_col) >= start_time)
    if end_time is not None:
        lf = lf.filter(pl.col(time_col) < end_time)
    if filter:
        lf = lf.filter(config.data.train.dataset.filter)

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
            if bucket_counts[k] == 0:
                weights[k] = float("nan")
                continue
            weights[k] = total / ((n_buckets - 1) * bucket_counts[k])

        proportions = {k: round((v / total) * 100, 1) for k, v in bucket_counts.items()}

        print()
        logger.info(f"Describing {col}")
        logger.info("counts: ")

        logger.info([f"{k}:" + f"{v:,}" for k, v in bucket_counts.items()])

        logger.info("proportions: ")
        logger.info([f"{k}:" + f"{v:,}" for k, v in proportions.items()])

        logger.info("weights: ")
        logger.info(weights)

        description = lf.select(pl.col(col)).describe()
        logger.info(description)

    logger.info("Finished Describe")

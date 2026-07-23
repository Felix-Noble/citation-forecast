import os

os.environ["POLARS_MAX_THREADS"] = "32"
import math
import shutil
from datetime import datetime
from logging import getLogger
from pathlib import Path

import polars as pl
import typer
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from config import config, env
from data.preprocess import clean_step, tokenise_step
from utils import export_parquet, setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)
app = typer.Typer(pretty_exceptions_enable=False)


@app.callback(invoke_without_command=True)
def main(
    input_path: Path = typer.Option(
        "",
        "--input-path",
        "-i",
        help="Path to input data",
    ),
    input_dataset: str = typer.Option(
        "", "--input-dataset", "-d", help="Dataset name to use as input"
    ),
    output_path: str = typer.Option(
        "",
        "--output",
        "-o",
        help='Name of exported dataset, defaults to "origin"-engineered',
    ),
    len_cols: list[str] = typer.Option(
        [],
        "--calc-len",
        "-l",
        help='Columns to calculate the len of, placed in "colname_len"',
    ),
    years_to_first: bool = typer.Option(
        True,
        "--years-to-first",
        help="Calulate years to first citation -> colname: years_to_first_citation",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Test/Dry run with 500 sample slice of data",
    ),
) -> None:
    # safety checks
    assert input_path or input_dataset, "Provide path to or name of an input dataset"

    # setup
    if input_path:
        origin = input_path
    if input_dataset:
        origin = ""
    logger.info(f"Loading data from {origin}")
    lf = pl.scan_parquet(list(origin.glob("*.par*")))

    if dry_run:
        lf = lf.slice(0, 50)

    if not output_path:
        output = origin.parent / f"{origin.name}-engineered"
    else:
        output = output_path

    logger.info(f"Engineering {origin} to {output}")

    progress_bar = Progress(
        TextColumn("[bold blue] {task.description}", justify="left"),
        BarColumn(bar_width=40),
        TextColumn("[task.completed]{task.completed}/{task.total}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TextColumn("<"),
        TimeRemainingColumn(),
        speed_estimate_period=60.0 * 10,  # mins
    )
    progress_bar.start()
    n_rows = lf.select(pl.len()).collect(engine="streaming").item()
    progress = progress_bar.add_task("Clean/Filter", total=n_rows)

    # Customisable operations
    for col in len_cols:
        msg = lambda: logger.info(f"Calculating len of {col}, dtype: {dtype}")
        dtype = lf.schema[col]
        if dtype.is_(pl.Utf8):
            msg()
            lf = lf.with_columns(pl.col(col).str.len_chars().alias(f"{col}_len"))
        elif dtype.base_type().is_(pl.List):
            msg()
            lf = lf.with_columns(pl.col(col).list.len().alias(f"{col}_len"))
        else:
            logger.error(f"Len operation not supported for {dtype}")
            raise TypeError(f"Len operation not supported for {dtype}")

    # Fixed operations
    if years_to_first:
        lf = lf.with_columns(
            years_to_first_citation=(
                pl.col("counts_by_year_years").list.get(0, null_on_oob=True)
                - pl.col("publication_date").dt.year()
            )
            .with_columns(cited_by_delta_first=pl.col("cited_by_dates").list.min())
                cited_by_delta_first=pl.col("cited_by_dates").list.min()
                - pl.col("publication_date")
            )
        )
    if dry_run:
        lf = lf.collect()
        print(lf)
        os.makedirs("./temp", exist_ok=True)
        lf.write_json("./temp/engineer-dry-run.json")

    else:
        os.makedirs(output, exist_ok=True)
        export_parquet(
            lf=lf,
            destination=output,
            n_partitions=64,
            progress_bar=progress_bar,
            progress_task=progress,
            compression_level=1,
        )

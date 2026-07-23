import math
import os
import shutil
from datetime import datetime
from logging import getLogger
from pathlib import Path

import numpy as np
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

import config
from data.preprocess import clean_step, tokenise_step
from utils import export_parquet, setup_logger

logger = getLogger(__name__)
_ = setup_logger(logger)
app = typer.Typer(pretty_exceptions_enable=False)


def run_dbscan_on_chunk(df_chunk: pl.DataFrame) -> pl.DataFrame:
    """Helper to process a single materialized chunk row-by-row."""
    results = []
    for row in df_chunk.iter_rows(named=True):
        vals = row["counts_per_year_delta"]
        freqs = row["counts_per_year_counts"]

        if not vals:
            continue

        X = np.array(vals).reshape(-1, 1)
        weights = np.array(freqs)

        db = DBSCAN(eps=15, min_samples=5).fit(X, sample_weight=weights)
        labels = db.labels_

        unique_labels = set(labels) - {-1}

        total_weight = weights.sum()
        noise_weight = weights[labels == -1].sum() if -1 in labels else 0
        noise_ratio = noise_weight / total_weight if total_weight > 0 else 0.0

        centroids = []
        n_members = []
        for label in sorted(unique_labels):
            mask = labels == label
            centroid = np.average(X[mask], axis=0, weights=weights[mask])
            centroids.append(float(centroid[0]))
            n_members.append(sum(labels == label))

        results.append(
            {
                "id": row["id"],
                "n_clusters": len(unique_labels),
                "centroids": centroids,
                "n_members": n_members,
                "noise_ratio": noise_ratio,
            }
        )
    return pl.DataFrame(results)


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
    n_partitions: int = typer.Option(
        64,
        help="No. of partitions to slice data into",
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
    assert origin, "Origin must be set"
    logger.info(f"Loading data from {origin}")
    lf_whole = pl.scan_parquet(list(origin.glob("*.par*")))

    if dry_run:
        lf_whole = lf_whole.slice(0, 50)

    if not output_path:
        output = origin.parent / f"{origin.name}-engineered"
    else:
        output = Path(output_path)
    os.makedirs(str(output), exist_ok=False)

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
    n_rows = lf_whole.select(pl.len()).collect(engine="streaming").item()
    progress = progress_bar.add_task("Rows", total=n_rows)
    part_progress = progress_bar.add_task("Parts", total=n_partitions)
    rows_per_part = math.ceil(n_rows / n_partitions)

    lf_whole = lf_whole.with_row_index("idx").with_columns(
        (pl.col("idx") // rows_per_part).clip(0, n_partitions - 1).alias("part")
    )
    # 3. Sink each part
    for i in range(n_partitions):
        lf = lf_whole.filter(pl.col("part") == i).drop(["idx", "part"])
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
                counts_by_year_delta=(
                    pl.col("counts_by_year_years")
                    - pl.col("publication_date").dt.year()
                )
            ).with_columns(
                counts_by_year_delta_first=pl.col("counts_by_year_delta").list.first()
            )
            lf_cited_dates = (
                lf.explode("referenced_works")
                .group_by("referenced_works")
                .agg(
                    [
                        pl.col("publication_date").alias("cited_by_dates"),
                        pl.col("referenced_works").first().alias("id"),
                    ]
                )
            )
            lf = (
                lf.join(
                    lf_cited_dates,
                    on="id",
                    how="inner",
                )
                .with_columns(
                    missing=pl.col("cited_by_count")
                    - pl.col("cited_by_dates").list.len()
                )
                .with_columns(
                    pl.col("cited_by_dates")
                    .list.eval(pl.element().dt.year())
                    .alias("cited_by_years"),
                )
                .with_columns(
                    cited_by_delta_years=pl.col("cited_by_years")
                    - pl.col("publication_date").dt.year()
                )
                .with_columns(
                    cited_by_delta_years_first=pl.col(
                        "cited_by_delta_years"
                    ).list.min(),
                    cited_by_delta_days_first=pl.col("cited_by_dates").list.min()
                    - pl.col("publication_date"),
                )
            )

            if dry_run:
                lf = lf.collect()
                print(lf)
                os.makedirs("./temp", exist_ok=True)
                lf.write_json("./temp/engineer-dry-run.json")
                break

            output_fname = output / f"part_{i}.parquet"
            lf.sink_parquet(
                output_fname,
                statistics=True,
                compression="zstd",
                compression_level=1,
            )
            n_written = pl.scan_parquet(output_fname).select(pl.len()).collect(engine="streaming").item()
            progress_bar.update(progress,advance=n_written)
            progress_bar.update(part_progress,advance=1)

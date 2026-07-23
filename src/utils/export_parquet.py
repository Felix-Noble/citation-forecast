import math
from pathlib import Path

import polars as pl
from rich.progress import Progress, TaskID


def measure_lf(lf: pl.LazyFrame) -> int:
    return lf.select(pl.len()).collect(engine="streaming").item()


def export_parquet(
    lf: pl.LazyFrame,
    destination: Path,
    n_partitions: int,
    progress_bar: Progress | None = None,
    progress_task: TaskID | None = None,
    compression_level: int = 1,
) -> None:

    n_rows = measure_lf(lf)
    rows_per_part = math.ceil(n_rows / n_partitions)

    lf = lf.with_row_index("idx").with_columns(
        (pl.col("idx") // rows_per_part).clip(0, n_partitions - 1).alias("part")
    )
    # 3. Sink each part
    for i in range(n_partitions):
        (
            lf.filter(pl.col("part") == i)
            .drop(["idx", "part"])
            .sink_parquet(
                destination / f"part_{i}.parquet",
                statistics=True,
                compression="zstd",
                compression_level=compression_level,
            )
        )
        if progress_bar is not None and progress_task is not None:
            progress_bar.update(progress_task, advance=rows_per_part)

import os
import shutil
from datetime import date, timedelta
from logging import getLogger
from pathlib import Path
from typing import NamedTuple, Protocol, override

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import Dataset

from utils.logging import setup_logger

from .polars_dataset import plot_target_distribution_polars

logger = getLogger(__name__)
_ = setup_logger(logger)


class CitationGraphDatasetConfig(BaseModel):
    loc: str
    x: list[str]
    y: list[str]
    meta_cols: list[str]
    filter: pl.Expr | None
    weights: Tensor | None
    max_len: int
    graph_max_len: int
    shuffle: bool = True
    sample: int | None
    return_mask: bool
    pad: bool
    pad_token_id: int
    truncate: bool
    truncate_method: str
    name: str
    auto_remove: bool
    time_col: str
    t_start: date | None
    t_end: date | None
    return_id: bool
    id_col: str = "id"
    subsample: int | None
    category_cols: list[str]
    sort_cols: list[str]
    top_k: int
    offset: str
    period: str
    add_x: list[str]
    max_mem_rows: int
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Env(Protocol):
    STAGED_LOC: Path


class CitationGraphDatasetOutput(NamedTuple):
    id: Tensor
    x: Tensor
    graph_x: Tensor
    y: Tensor
    weight: Tensor
    mask: Tensor
    graph_x_mask: Tensor


class CitationGraphDataset[T_Config](Dataset[CitationGraphDatasetOutput]):
    config = CitationGraphDatasetConfig

    def __init__(
        self,
        config: CitationGraphDatasetConfig,
        env: Env,
    ):
        super().__init__()

        assert not (config.weights is None and not config.y), (
            "Weights cannot be given when no y is given"
        )
        assert not (config.shuffle and config.sample is not None), (
            "Shuffle and Sample cannot both be used"
        )
        self.data_path = env.STAGED_LOC / config.loc
        self.meta_cols = config.meta_cols
        self.filter = config.filter
        self.t_start = config.t_start
        self.t_end = config.t_end
        self.weights = config.weights
        self.time_col = config.time_col
        self.return_id = config.return_id
        self.id_col = config.id_col
        self.max_len = config.max_len
        self.graph_max_len = config.graph_max_len
        self.pad = config.pad
        self.pad_value = config.pad_token_id
        self.return_mask = config.return_mask
        self.truncate = config.truncate
        self.truncate_method = config.truncate_method
        self.name = config.name
        self.hot_path: Path = Path("./.temp") / "hot" / self.name
        self.subsample = config.subsample
        self.x = config.x
        self.y = config.y
        self.add_x = config.add_x

        self.category_cols = config.category_cols
        self.sort_cols = config.sort_cols
        self.offset = config.offset
        self.period = config.period
        self.top_k = config.top_k

        if self.subsample is not None:
            self.hot_path = Path("./.temp") / "hot" / f"{self.name}-DRY"

        x_columns = list(
            set(
                self.x
                + self.meta_cols
                + self.category_cols
                + self.sort_cols
                + self.add_x
                + [self.time_col]
            )
        )
        columns = list(set(x_columns + self.y + [config.id_col, "referenced_works"]))
        self.x_hot_path: Path = self.hot_path / "x.ipc"
        self.add_x_hot_path: Path = self.hot_path / "add_x.ipc"
        self.y_hot_path: Path = self.hot_path / "y.ipc"
        self.id_hot_path: Path = self.hot_path / "id.ipc"

        if self.hot_path.exists() and config.auto_remove:
            logger.info(f"Found data at {self.hot_path}, deleting")
            shutil.rmtree(self.hot_path)

        files = list(Path(self.data_path).glob("*.par*"))

        lf: pl.LazyFrame = pl.scan_parquet(files).select(columns)

        lf = lf.drop_nulls(columns)

        if self.filter is not None:
            lf = lf.filter(self.filter)

        if self.t_end is not None and config.time_col:
            lf = lf.filter(pl.col(self.time_col) < self.t_end)

        self.figure_log = plot_target_distribution_polars(lf, self.y[0])
        self.figure = plot_target_distribution_polars(lf, self.y[0], False)

        lf = lf.with_columns(
            graph_x_single=pl.concat_list(self.add_x),
        )

        graph_x_df = (
            lf.explode("referenced_works")
            .join(
                lf.select(
                    [self.id_col, "graph_x_single"]
                    + self.category_cols
                    + self.sort_cols
                ),
                left_on="referenced_works",
                right_on=self.id_col,
                suffix="_referenced",
            )
            .filter(
                pl.any_horizontal(
                    [
                        pl.col(cat_col) == pl.col(f"{cat_col}_referenced")
                        for cat_col in self.category_cols
                    ]
                )
            )
            .group_by(self.id_col)
            .agg(
                [
                    pl.col("graph_x_single_referenced")
                    .sort_by(
                        [f"{col}_referenced" for col in self.sort_cols],
                        descending=True,
                    )
                    .head(self.top_k)
                    .list.explode()
                    .alias("graph_x")
                ]
            )
        )

        graph_x_df = graph_x_df.filter(pl.col("graph_x").list.len() >= 1)
        lf = (
            lf.join(graph_x_df.lazy(), on=self.id_col, how="inner")
            .with_columns(
                x=pl.concat_list(self.x),
                y=pl.concat_list(self.y),
            )
            .drop_nulls(["graph_x", "x", "y"])
        )

        if self.t_start is not None and config.time_col:
            lf = lf.filter((pl.col(self.time_col) >= self.t_start))

        if self.truncate and self.truncate_method == "drop":
            lf = lf.filter(pl.col("x").list.len() <= self.max_len)
            lf = lf.filter(pl.col("graph_x").list.len() <= self.graph_max_len)

        if self.subsample is not None:
            logger.info(f"Taking {self.subsample} subsamples")
            lf = lf.slice(0, self.subsample)

        rows = lf.select(pl.len()).collect(engine="streaming").item()
        logger.info(
            f"{rows:,} rows where x len <= {self.max_len} & total graph length <= {self.graph_max_len} to hotpath: {self.name}"
        )
        if rows > config.max_mem_rows:
            os.makedirs(self.hot_path, exist_ok=True)
            lf.select(["x", "graph_x", "y", "id"]).sink_ipc(self.x_hot_path)

        if rows is None or rows > config.max_mem_rows:
            self.df: pl.DataFrame = pl.read_ipc(self.x_hot_path, memory_map=True)
            logger.info(f"Hot path {self.hot_path} loaded")
        else:
            self.df = lf.collect(engine="streaming")
            logger.info(f"Hot path {self.hot_path} loaded into mem")

    def __len__(self) -> int:
        return len(self.df)

    def _format_x(self, x: Tensor) -> Tensor:
        return x.long()

    def _format_y(self, y: Tensor) -> Tensor:
        return y

    @override
    def __getitem__(self, idx: int) -> CitationGraphDatasetOutput:
        id = torch.tensor(float("nan"))
        x = torch.tensor(float("nan"))
        graph_x = torch.tensor(float("nan"))
        y = torch.tensor(float("nan"))
        mask = torch.tensor(float("nan"))
        graph_x_mask = torch.tensor(float("nan"))
        weight = torch.tensor(float("nan"))
        # X (input)
        row = self.df.row(idx, named=True)
        x: Tensor = torch.tensor(row["x"], dtype=torch.float32).flatten()
        x = self._format_x(x)

        if self.pad and x.size(0) < self.max_len:
            x = nn.functional.pad(
                x, (0, self.max_len - x.size(0)), value=self.pad_value
            )

        if self.truncate == True & x.size(0) > self.max_len:
            x = x[: self.max_len]
            # y (target)

        graph_x: Tensor = torch.tensor(row["graph_x"], dtype=torch.float32).flatten()
        graph_x = self._format_x(x)
        if self.pad and graph_x.size(0) < self.graph_max_len:
            graph_x = nn.functional.pad(
                graph_x, (0, self.graph_max_len - graph_x.size(0)), value=self.pad_value
            )

        if self.truncate == True & graph_x.size(0) > self.graph_max_len:
            graph_x = graph_x[: self.graph_max_len]
            # y (target)
        y = torch.tensor(row["y"], dtype=torch.float32).flatten()
        y = self._format_y(y)

        # id (tracking)
        if self.return_id:
            id = row["id"]

        # weights (per target class)
        if self.weights is not None:
            weight: Tensor = torch.tensor(
                [
                    torch.tensor(self.weights[target.long()], dtype=torch.float32)
                    for target in torch.atleast_1d(y)
                ]
            ).flatten()

        if self.return_mask:
            mask = (x != self.pad_value).bool()
            graph_x_mask = (graph_x != self.pad_value).bool()

        out = CitationGraphDatasetOutput(
            id=id,
            x=x,
            graph_x=graph_x,
            y=y,
            mask=mask,
            graph_x_mask=graph_x_mask,
            weight=weight,
        )
        return out

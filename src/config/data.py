from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl
import torch

from data import datasets as dd
from data.dataloaders import DataLoaderConfig

_dataset = dd.GraphDataset
_train_loc = "1920-2000-lowercase-2-embedded"
_test_loc = "1920-2000-lowercase-2-embedded"
_x = [
    'title abstract_embedding',
]
_y = ["cited_by_delta_years_first"]
_meta = ["field_id", "subfield_name", 'cited_by_count']  # test:topic name
_filter = (pl.col("cited_by_count") >= 1) & (pl.col('referenced_works').list.len() >= 5)
_period = 5
_offset = -6
_t_unit = "y"
_cat_cols = ["topic_name", 'subfield_name', 'field_name', 'type']
_sort_cols = ["publication_date"]
_add_x = _x
_top_k = 15

_theta = 15
_boundaries = torch.tensor([1, 10])
# 10, 100
_min = 0
_max = 7000

_weights = torch.tensor([1.07, 0.93])
_train_max_len = 400
_test_max_len = 400
_graph_max_len = 100

_train_start = date(1920, 1, 1)
_train_end = date(1990, 1, 1)
_test_start = date(1990, 1, 1)
_test_end = date(1991, 1, 1)

_pad_token_id = 199999
from .model import model
_pad_value = torch.zeros(model.embed_dim, dtype=torch.float32)
_max_mem_rows = int(1e6)
_num_workers = 2
_train_batch_size = 16
_test_batch_size = 16
_prefetch_factor = 4

_train_subsample = None
_test_subsample = None

_train_loader_samples = None#100_000
_test_loader_samples = None


@dataclass
class Train:
    clss = _dataset
    dataset = _dataset.config(
        x=_x,
        y=_y,
        meta_cols=_meta,
        filter=_filter,
        max_len=_train_max_len,
        pad_value=_pad_value,
        weights=_weights,
        shuffle=False,
        sample=None,
        loc=_train_loc,
        return_mask=True,
        id_col="id",
        truncate=True,
        truncate_method="drop",
        pad=True,
        name="train-dataset",
        auto_remove=True,
        time_col="publication_date",
        t_start=_train_start,
        t_end=_train_end,
        return_id=False,
        subsample=_train_subsample,
        theta=_theta,
        boundaries=_boundaries,
        max=_max,
        min=_min,
        category_cols=_cat_cols,
        sort_cols=_sort_cols,
        period=_period,
        offset=_offset,
        t_unit=_t_unit,
        top_k=_top_k,
        add_x=_add_x,
        graph_max_len=_graph_max_len,
        max_mem_rows=_max_mem_rows,
    )
    loader = DataLoaderConfig(
        batch_size=_train_batch_size,
        num_workers=_num_workers,
        prefetch_factor=_prefetch_factor,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
        samples=_train_loader_samples,
        drop_last=True,
    )


@dataclass
class Test:
    clss = _dataset
    dataset = _dataset.config(
        x=_x,
        y=_y,
        meta_cols=_meta,
        filter=_filter,
        max_len=_test_max_len,
        pad_value=_pad_value,
        weights=None,
        shuffle=True,
        sample=None,
        # test_dataset: str = "wikiTestShort"
        loc=_test_loc,
        return_mask=True,
        truncate=True,
        truncate_method="drop",
        pad=True,
        name="test-dataset",
        auto_remove=True,
        time_col="publication_date",
        t_start=_test_start,
        t_end=_test_end,
        return_id=True,
        id_col="id",
        subsample=_test_subsample,
        theta=_theta,
        boundaries=_boundaries,
        max=_max,
        min=_min,
        category_cols=_cat_cols,
        sort_cols=_sort_cols,
        period=_period,
        offset=_offset,
        t_unit=_t_unit,
        top_k=_top_k,
        add_x=_add_x,
        graph_max_len=_graph_max_len,
        max_mem_rows=_max_mem_rows,
    )
    loader = DataLoaderConfig(
        batch_size=_test_batch_size,
        num_workers=_num_workers,
        prefetch_factor=_prefetch_factor,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
        samples=_test_loader_samples,
        drop_last=True,
    )


train = Train()
test = Test()

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, PositiveInt

from data import datasets as dd
from data.dataloaders import DataLoaderConfig

_dataset = dd.BinaryThresholdDataset
_train_loc = "OAShort"
_test_loc = "OAShort"
_x = ["abstract_tokens"]
_y = ["cited_by_count"]
_theta = 0.0

_train_start = None
_train_end = None
_test_start = None
_test_end = None

_num_workers = 1
_prefetch_factor = 3


@dataclass
class Train:
    clss = _dataset
    dataset = _dataset.config(
        x=_x,
        y=_y,
        max_len=100,
        n_buckets=2,
        pad_token_id=1999,
        weights=None,
        shuffle=True,
        sample=None,
        # test_dataset: str = "wikiTestShort"
        loc=_train_loc,
        return_mask=True,
        truncate=True,
        truncate_method="drop",
        pad=True,
        name="test-dataset",
        auto_remove=True,
        time_col="publication_date",
        t_start=_train_start,
        t_end=_train_end,
        return_id=False,
        id_col="",
        dry_run=False,
        theta=_theta,
    )
    loader = DataLoaderConfig(
        batch_size=2,
        num_workers=_num_workers,
        prefetch_factor=_prefetch_factor,
        persistent_workers=False,
        pin_memory=True,
        shuffle=True,
        samples=None,
        drop_last=True,
    )


@dataclass
class Test:
    clss = _dataset
    dataset = _dataset.config(
        x=_x,
        y=_y,
        max_len=100,
        n_buckets=2,
        pad_token_id=1999,
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
        return_id=False,
        id_col="",
        dry_run=False,
        theta=_theta,
    )
    loader = DataLoaderConfig(
        batch_size=2,
        num_workers=_num_workers,
        prefetch_factor=_prefetch_factor,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False,
        samples=None,
        drop_last=True,
    )


train = Train()
test = Test()

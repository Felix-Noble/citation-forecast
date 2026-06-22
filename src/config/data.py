from dataclasses import dataclass
from datetime import date

from data import datasets as dd
from data.dataloaders import DataLoaderConfig

_dataset = dd.LogRegressDataset
_train_loc = "all-licensed-1800-2027-lowercase-T2A2"
_test_loc = "all-licensed-1800-2027-lowercase-T2A2"
_x = ['field_name_tokens', 'subfield_name_tokens', 'source_name_tokens', 'title_tokens', "abstract_tokens"]
_y = ["cited_by_count"]
_theta = 0

_train_start = date(1800, 1,1)
_train_end = date(1990, 1, 1) 
_test_start = date(2020, 1, 1)
_test_end = date(2021, 1, 1)

_pad_token_id=199999
_num_workers = 2
_prefetch_factor = 4
_subsample = None


@dataclass
class Train:
    clss = _dataset
    dataset = _dataset.config(
        x=_x,
        y=_y,
        max_len=300,
        n_buckets=2,
        pad_token_id=_pad_token_id,
        weights=None,
        shuffle=True,
        sample=None,
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
        subsample=_subsample,
        theta=_theta,
    )
    loader = DataLoaderConfig(
        batch_size=512 * 5,
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
        max_len=500,
        n_buckets=2,
        pad_token_id=_pad_token_id,
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
        subsample=_subsample,
        theta=_theta,
    )
    loader = DataLoaderConfig(
        batch_size=16 * 70,
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

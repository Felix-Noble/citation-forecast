# src/builders/build_dataset_ordinal.py
from src.data.datasets import OrdinalDataset, BinaryCategoricalDataset
from config import Config, config, env

def build_binary_dataset(
        dataset: str,
        x: list[str],
        y: str,
        mask: bool = True,
        pad: bool = False,
        dry_run: bool = False, 
        config: Config = config,
        ):

    dataset = BinaryCategoricalDataset(
        data_path=str(env.STAGED_LOC / dataset),
        X=x,
        y=y,
        t_start=config.train.train_start.toordinal(),
        t_end=config.train.train_end.toordinal(),
        config=config,
        return_mask=mask,
        pad=pad,
        dry_run=dry_run,
    )

    return dataset

def build_ordinal_dataset(
        dataset: str,
        x: list[str],
        y: str,
        mask: bool = True,
        pad: bool = False,
        dry_run: bool = False, 
        config: Config = config,
        ):
    dataset = OrdinalDataset(
        data_path=str(env.STAGED_LOC / dataset),
        X=x,
        y=y,
        t_start=config.train.train_start.toordinal(),
        t_end=config.train.train_end.toordinal(),
        config=config,
        pad=pad,
        return_mask=mask,
        dry_run=dry_run
    )

    return dataset

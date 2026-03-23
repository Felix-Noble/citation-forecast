# src/builders/build_dataset_ordinal.py
from src.data.datasets import OrdinalDataset, BinaryCategoricalDataset
from config import Config, config, env

def build_binary_datasets(
        dataset: str,
        dry_run: bool, 
        config: Config = config,
        ):

    train_dataset = BinaryCategoricalDataset(
        data_path=str(env.STAGED_LOC / dataset),
        X=['title_tokens', 'abstract_tokens'],
        y='cited_by_count',
        t_start=config.train.train_start.toordinal(),
        t_end=config.train.train_end.toordinal(),
        config=config,
        return_mask=False,
        pad=False,
        dry_run=dry_run,
    )

    test_dataset = BinaryCategoricalDataset(
        data_path=str(env.STAGED_LOC / dataset),
        X=['title_tokens', 'abstract_tokens'],
        y='cited_by_count',
        t_start=config.train.test_start.toordinal(),
        t_end=config.train.test_end.toordinal(),
        config=config,
        return_mask=False,
        pad=False,
        dry_run=dry_run
    )
    
    return train_dataset, test_dataset

def build_ordinal_datasets(
        dataset: str,
        dry_run: bool, 
        config: Config = config,
        ):

    train_dataset = OrdinalDataset(
        data_path=str(env.STAGED_LOC / dataset),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        t_start=config.train.train_start.toordinal(),
        t_end=config.train.train_end.toordinal(),
        config=config,
        pad=False,
        return_mask=False,
        dry_run=dry_run
    )

    test_dataset = OrdinalDataset(
        data_path=str(env.STAGED_LOC / dataset),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        t_start=config.train.test_start.toordinal(),
        t_end=config.train.test_end.toordinal(),
        config=config,
        pad=False,
        return_mask=False,
        dry_run=dry_run
    )
    
    return train_dataset, test_dataset

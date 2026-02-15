# src/builders/build_dataset_ordinal.py
from ..datasets import OrdinalDataset
from config import Config, config

def build_ordinal_dataset(
        data_path: str,
        testing: bool, 
        config: Config = config,
        ):

    train_dataset = OrdinalDataset(
        data_path=str(config.data.staged / data_path),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        t_start=config.data.train_start.toordinal(),
        t_end=config.data.train_end.toordinal(),
        config=config,
        pad=True,
        return_mask=True,
        testing=testing
    )

    test_dataset = OrdinalDataset(
        data_path=str(config.data.staged / data_path),
        X='abstract_tokens',
        y='citation_normalized_percentile',
        t_start=config.data.test_start.toordinal(),
        t_end=config.data.test_end.toordinal(),
        config=config,
        pad=True,
        return_mask=True,
        testing=testing
    )
    
    return train_dataset, test_dataset

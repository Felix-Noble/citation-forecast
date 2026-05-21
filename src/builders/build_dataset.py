from src.data.datasets import dataset_registry
from config import Config, config
from pydantic import ValidationError
from torch.utils.data import Dataset

def build_dataset(
    **kwargs
    ) -> type[Dataset]:
    dataset_name = kwargs.get('dataset') 

    if dataset_name not in dataset_registry.keys:
        raise KeyError(f'dataset "{dataset_name}" name not found. Available datasets: [ {", ".join(dataset_registry.keys)} ]')
    dataset = dataset_registry[dataset_name]

    # verify model config

    return dataset(**kwargs)

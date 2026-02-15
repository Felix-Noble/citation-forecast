# src/builders/build_dataloader.py
from config import Config, config
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

def build_dataloader(
    dataset: Dataset[tuple[Tensor, ...] | Tensor],
    config: Config = config,
    sampler = None,
) -> DataLoader[tuple[Tensor, ...] | Tensor]:

    dataloader = DataLoader(
        dataset,
        batch_size=config.train.batch_size,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        shuffle=config.train.shuffle,
        sampler = sampler,
        drop_last=True,
    )
    return dataloader

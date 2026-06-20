from pydantic import BaseModel, PositiveInt
from torch.utils.data import DataLoader, Dataset

from data import PortionSampler


class DataLoaderConfig(BaseModel):
    batch_size: PositiveInt
    num_workers: PositiveInt
    prefetch_factor: PositiveInt | None
    persistent_workers: bool
    pin_memory: bool
    shuffle: bool
    samples: PositiveInt | None
    drop_last: bool


class DLWrapper[T_Output]:
    config: type[DataLoaderConfig] = DataLoaderConfig

    def __init__(
        self,
        dataset: Dataset[T_Output],
        config: DataLoaderConfig,
    ):
        if config.samples is not None:
            sampler = PortionSampler(dataset, config.samples)
        else:
            sampler = None

        self.loader: DataLoader[T_Output] = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
            pin_memory=config.pin_memory,
            shuffle=config.shuffle,
            sampler=sampler,
            drop_last=config.drop_last,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self) -> int:
        return len(self.loader)

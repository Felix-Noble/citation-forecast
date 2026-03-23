# src/builders/build_dataloader.py
from config import Config, config
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

def build_dataloader(
    dataset: Dataset[tuple[Tensor, ...] | Tensor],
    config: Config = config,
    sampler = None,
) -> DataLoader[tuple[Tensor, ...] | Tensor]:

    def pad_to_longest_collate(
            batch: tuple[list[Tensor], ...], 
            pad_value: int=config.model.pad_token_id,
            return_mask: bool=True,
                               ) -> tuple[Tensor, ...]:
        """ pads dataset x/mask objects to same length
        
        input: 
            batch: tuple containing (x, y)
        output:
            x: tensor padded to longest length in batch 
            y: tensor unchanged
            mask: if return_mask, same shape as x
        """
        X, y = [x[0] for x in batch], [x[1] for x in batch]
        X = pad_sequence(
                X,
                batch_first=True, 
                padding_value=pad_value,
                padding_side='right',
                )
        y = torch.tensor(y)
        if return_mask:
            mask = (X != pad_value).bool().unsqueeze(1).expand(-1, X.shape[1], -1)
            return X, y, mask 
        return X, y

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
        collate_fn=pad_to_longest_collate,
    )
    return dataloader

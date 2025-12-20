#!/usr/bin/env python3
# train.py
# pyright: strict
# basedpyright: reccomend
from config.config import config, DataConfig
from src.types.valid_paths import VALID_PATHS
from src.utils.logging import setup_logger
from src.metric_trackers.classification_tracker import ClassificationTracker
from src.datasets.df_dataset import DF_Dataset

from typing import Literal
from sklearn.metrics import roc_auc_score # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import typer
from pathlib import Path
from logging import getLogger

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)
 
app = typer.Typer()

def init_dataloader(
    dataset: Dataset[tuple[Tensor, ...] | Tensor],
    device: torch.device,
    data_config: DataConfig = config.data,
) -> DataLoader[tuple[Tensor, ...] | Tensor]:

    def collate_fn(batch: torch.Tensor | tuple[Tensor, ...]):
        if isinstance(batch, tuple):
            return (torch.stack(tensor).to(device) for tensor in batch)
        return torch.stack(batch).to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        num_workers=1,
        prefetch_factor=None,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dataloader

def eval_model(model) -> None:
    return

@app.command()
def main(
    data_path: VALID_PATHS = typer.Argument( 
        help='data path relative to config.data.staged'
    )
) -> None:
    " Main Loop "
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f'Running on {device}')

    dataset = DF_Dataset(
        data_path=str(config.data.staged / data_path),
        X='abstract_tokens',
        max_len=4000,
        pad=True,
        pad_value=0
    )
    dataloader = init_dataloader(dataset, device)
    model = nn.Linear(dataset.MAX_LEN, 1) 

    for X in dataloader:
        logger.debug(f'X device {X.device}')
        out = model(X)
        logger.debug(f'out device {out.device}')
        quit()
    return

if __name__ == '__main__':
    app()
    '''
    logger.info('Starting test')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    logger.info(f'Device: {device}')
    n_out = 3
    n_examps = 50
    batch_size = 5
    n_batches = n_examps // batch_size

    test_x = torch.randn((n_batches, batch_size, 20))
    test_y = torch.randint(0, n_out, (n_batches * batch_size,))
    metric_tracker = ClassificationTracker(
        (batch_size, n_out), n_batches, n_examps, n_examps, torch.float32, device
    )
    model: torch.nn.Linear = torch.nn.Linear(20, n_out)
     
    for examp in test_x:
        out: torch.Tensor = model(examp) # pyright: ignore[reportAny]
        metric_tracker.store_output(out)

    outputs = torch.softmax(metric_tracker._stack_buffer(), dim=1).cpu()
    roc_auc = roc_auc_score(test_y, outputs, multi_class='ovo', average='weighted')
    metric_tracker._log_metric('train roc_auc', roc_auc, 1)
    report = metric_tracker.report()
    print(report)
    '''

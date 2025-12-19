# train.py
# pyright: strict
# ruff: strict
from config.config import config
from src.utils.logging import setup_logger
from src.metric_trackers.classification_tracker import ClassificationTracker

from sklearn.metrics import roc_auc_score # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
from concurrent.futures import ThreadPoolExecutor

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from logging import getLogger

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

def init_dataloader(dataset: Dataset[tuple[Tensor, ...]]) -> DataLoader[tuple[Tensor, ...]]:
    dataloader = DataLoader(
        dataset
    )
    return dataloader

def eval_model(model) -> None:
    return

def main() -> None:
    return

if __name__ == '__main__':
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

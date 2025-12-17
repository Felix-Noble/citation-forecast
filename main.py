from config.config import config
from src.utils.logging import setup_logger
from src.metric_trackers.classification_tracker import ClassificationTracker

import torch

from pathlib import Path
from logging import getLogger


logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

if __name__ == '__main__':
    logger.info('Starting test')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    logger.info(f'Device: {device}')
    test_outs = torch.randn((10,10,20))

    metric_tracker = ClassificationTracker(
        test_outs.shape, 20, 20, torch.float32, device
    )
    model: torch.nn.Module = torch.nn.Linear(20, 20)

    for out in test_outs:
        logger.info(out.shape)
        metric_tracker.store_output(model(out))
        logger.debug(metric_tracker.buffer_cursor)

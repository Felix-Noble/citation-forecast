# src/callbacks/isnan_async.py
import torch
import time

def isnan_async(
        loss,
        logger
                ):
    if torch.any(torch.isnan(loss)):
        logger.error('Loss is NaN, interrupting training')
        time.sleep(5)
        raise ValueError('Loss is NaN, interrupting training')


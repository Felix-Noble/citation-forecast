from config import Config
from src.training.losses import loss_registry
import torch

def build_loss(config: Config) -> torch.nn.modules.loss._Loss:
    """ Builds loss obj, expects PyTorch style (callable class)"""
    assert config.train.loss_fn in loss_registry.keys, f'{config.train.loss_fn} not registered, available: {", ".join(loss_registry.keys)}'
    return loss_registry[config.train.loss_fn](config)

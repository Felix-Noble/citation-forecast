from config import Config, config
from src.training.losses import loss_registry
import torch

def build_loss(config: Config = config) -> torch.nn.modules.loss._Loss:
    return loss_registry[config.train.loss_fn]()

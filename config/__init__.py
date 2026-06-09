# config/__init__.py
from .config import Config, TrainConfig, ModelConfig
from .env import Env

config: Config = Config()
env: Env = Env()

__all__ = [
        'config', 'Config', 'TrainConfig', 'ModelConfig',
        'env'
        ]

# Safety checks

if config.train.sample is not None and config.train.shuffle:
    raise ValueError('trian.shuffle and train.sample cannot both be true')


# config/__init__.py
from .config import Config
from .env import Env

config: Config = Config()
env: Env = Env()

__all__ = [
        'config', 'Config',
        'env'
        ]

# Safety checks
if config.train.sample:
    assert config.train.sample % config.train.batch_size  == 0, 'sample % batch_size must == 0'

if config.train.sample and config.train.shuffle:
    raise ValueError('trian.shuffle and train.sample cannot both be true')


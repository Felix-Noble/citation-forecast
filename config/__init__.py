# config/__init__.py
from .config import Config, Loss_fn, Optimizer
from .env import Env

config: Config = Config()
env: Env = Env()

__all__ = [ 'Config', 'config', 'env', 'Loss_fn', 'Optimizer']

# Safety checks
if config.train.sample:
    assert config.train.sample % config.train.batch_size  == 0, 'sample % batch_size must == 0'

if config.train.sample and config.train.shuffle:
    raise ValueError('trian.shuffle and train.sample cannot both be true')


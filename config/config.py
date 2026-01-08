from dataclasses import dataclass
from pathlib import Path

MLFLOW_DIR = '/home/fnoble/Dropbox/experiment-tracking'
EXPERIMENT_NAME = 'AbstractForecast-testing'

@dataclass(frozen=True)
class ModelConfig:
    encoder_layers: int = 5

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 1000
    batch_size: int = 2000
    lr: float = 1e-4
    weight_decay: float = 0.9
    loss_fn: str = 'CrossEntropyLoss'
    optimizer: str = 'AdamW'
    shuffle: bool = True
    sample: bool = False

@dataclass(frozen=True)
class DataConfig:
    raw: Path = Path('/home/fnoble/data/OpenAlex-parquet/')
    staged: Path = Path('/home/fnoble/data/staged/')

@dataclass(frozen=True)
class LogConfig:
    file: str = 'ERROR'
    console: str = 'DEBUG'

@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig() 
    train: TrainConfig = TrainConfig()
    logging: LogConfig = LogConfig()

config = Config()

# Safety checks
if config.train.sample and config.train.shuffle:
    raise ValueError('trian.shuffle and train.sample cannot both be true')

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:
    encoder_layers: int = 5

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    batch_size: int = 2
    optimizer = None     
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

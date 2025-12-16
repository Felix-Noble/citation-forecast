from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:
    encoder_layers: int = 5

@dataclass(frozen=True)
class LogConfig:
    file: str = 'ERROR'
    console: str = 'DEBUG'

@dataclass(frozen=True)
class Config:
    data_path = Path('/home/fnoble/data/')
    model: ModelConfig = ModelConfig() 
    logging: LogConfig = LogConfig()

config = Config()

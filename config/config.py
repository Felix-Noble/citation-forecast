import torch
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
EXPERIMENT_NAME = 'AbstractForecast-testing'
def init_lr_scheduler(
    optimizer,
):
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.8, 
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100,
        eta_min=1e-8,
        last_epoch=-1, # Used to implement restart from checkpoint
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        [5],
        last_epoch=-1,
    )
    return scheduler 

@dataclass(frozen=True)
class ModelConfig:
    model_name:str = 'R_RNN'
    vocab_size:int = 201_088
    eos_token: int = 200_002
    embed_dim: int = 32
    attention_dim: int = 32
    n_layers: int = 4
    n_out: int = 5

Loss_fn = torch.nn.CrossEntropyLoss
Optimizer = torch.optim.AdamW

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.9
    loss_fn: str = Loss_fn.__name__
    optimizer: str = Optimizer.__name__
    eval_interval: int = 1
    checkpoint_interval: int = 1
    shuffle: bool = True
    sample: bool = False

@dataclass(frozen=True)
class DataConfig:
    raw: Path = Path('/home/fnoble/data/OpenAlex-parquet/')
    staged: Path = Path('/home/fnoble/data/staged/')
    train_start: int = datetime(1950, 1, 1).toordinal()
    train_end: int = datetime(2017, 1, 1).toordinal()
    test_start: int = datetime(2017, 1, 1).toordinal()
    test_end: int = datetime(2025, 1, 1).toordinal()

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

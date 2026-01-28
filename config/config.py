import torch
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
EXPERIMENT_NAME = 'AF-Psych-M'

@dataclass(frozen=True)
class ModelConfig:
    model_name:str = 'MMLP2'
    vocab_size:int = 201_088
    pad_token: int = 0
    embed_dim: int = 128
    hidden_dim: int = 256
    n_layers: int = 8
    n_out: int = 3

Loss_fn = torch.nn.CrossEntropyLoss
Optimizer = torch.optim.AdamW

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 150
    batch_size: int = 32
    opttim_step_interval: int = 6 # n batches until optimizer steps
    lr: float = 1e-5 
    weight_decay: float = 0.9
    loss_fn: str = Loss_fn.__name__
    optimizer: str = Optimizer.__name__
    eval_interval: int = 1
    checkpoint_interval: int = 10
    shuffle: bool = True
    sample: bool = False
    mat_mul_precision: str = 'high'

@dataclass(frozen=True)
class DataConfig:
    raw: Path = Path('/home/fnoble/data/OpenAlex-parquet/')
    staged: Path = Path('/home/fnoble/data/staged/')
    max_len: int = 300
    train_start: datetime = datetime(2000, 1, 1)
    train_end: datetime = datetime(2000, 3, 1)
    test_start: datetime = datetime(2000, 3, 1)
    test_end: datetime = datetime(2000, 6, 1)

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

def init_lr_scheduler(
    optimizer,
):
    milestones = [15, 60]
    
    sum = 0
    for x in milestones:
        sum += x
        assert config.train.epochs != sum

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1, 
        end_factor=1.0,
        total_iters=milestones[0],
        last_epoch=-1 # Used to implement restart from checkpoint
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=milestones[1] - milestones[0],
        eta_min=1e-10,
        last_epoch=-1, # Used to implement restart from checkpoint
    )

    final_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.1,
        patience=0,

    )

    first_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        milestones[:-1],
        last_epoch=-1,
    )

    return (first_scheduler, final_scheduler), (milestones[-1], config.train.epochs)

# Safety checks
if config.train.sample and config.train.shuffle:
    raise ValueError('trian.shuffle and train.sample cannot both be true')


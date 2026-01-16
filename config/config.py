import torch
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
EXPERIMENT_NAME = 'AF-Psych-M-test'

@dataclass(frozen=True)
class ModelConfig:
    model_name:str = 'rf_rnn_static'
    vocab_size:int = 201_088
    eos_token: int = 200_002
    embed_dim: int = 512
    attention_dim: int = 512
    n_layers: int = 10
    n_out: int = 5

Loss_fn = torch.nn.CrossEntropyLoss
Optimizer = torch.optim.AdamW

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 6 
    opttim_step_interval = 120 # n batches until optimizer steps
    lr: float = 1e-5 
    weight_decay: float = 0.9
    loss_fn: str = Loss_fn.__name__
    optimizer: str = Optimizer.__name__
    eval_interval: int = 2
    checkpoint_interval: int = 2
    shuffle: bool = True
    sample: bool = False

@dataclass(frozen=True)
class DataConfig:
    raw: Path = Path('/home/fnoble/data/OpenAlex-parquet/')
    staged: Path = Path('/home/fnoble/data/staged/')
    train_start: int = datetime(1990, 1, 1).toordinal() # TODO change these to only datetime (strings for mlflow logging, convert to ordinal when needed)
    train_end: int = datetime(2010, 1, 1).toordinal()
    test_start: int = datetime(2010, 1, 1).toordinal()
    test_end: int = datetime(2012, 1, 1).toordinal()

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
    milestones = [15]
    sum = 0
    for x in milestones:
        sum += x
        assert config.train.epochs != sum

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01, 
        end_factor=1.0,
        total_iters=milestones[0],
        last_epoch=-1 # Used to implement restart from checkpoint
    )

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.train.epochs - milestones[0],
        eta_min=1e-8,
        last_epoch=-1, # Used to implement restart from checkpoint
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_scheduler],
        milestones,
        last_epoch=-1,
    )
    return scheduler 

# Safety checks
if config.train.sample and config.train.shuffle:
    raise ValueError('trian.shuffle and train.sample cannot both be true')


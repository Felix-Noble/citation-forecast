import torch
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
EXPERIMENT_NAME = 'AF-Psych-M'

top_k = (300 for _ in range(8))
top_k = (*top_k, 256, 128, 64, 32, 16, 8, 4)
n_layers = len(top_k)
selector_heads = tuple(4 for _ in range(n_layers))
process_heads = tuple(4 for _ in range(n_layers))

@dataclass(frozen=True)
class ModelConfig:
    model_name: str = 'h_attn_single'
    vocab_size: int = 201_088
    pad_token: int = 0
    dtype: torch.dtype =  torch.float32
    top_k: tuple[int, ...] = top_k
    selector_heads: tuple[int, ...] = selector_heads
    process_heads: tuple[int, ...] = process_heads
    n_layers: int = n_layers
    embed_dim: int = 512
    hidden_dim: int = 512
    n_out: int = 3
    dropout: float = 0.05

Loss_fn = torch.nn.CrossEntropyLoss
Optimizer = torch.optim.AdamW

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch_size: int = 64
    opttim_step_interval: int = 50 # n batches until optimizer steps
    lr: float = 1e-4 
    weight_decay: float = 0.9
    loss_fn: str = Loss_fn.__name__
    optimizer: str = Optimizer.__name__
    eval_interval: int = 5
    checkpoint_interval: int = 10
    shuffle: bool = False
    sample: int | None = 100_032
    mat_mul_precision: str = 'high'

@dataclass(frozen=True)
class DataConfig:
    raw: Path = Path('/home/fnoble/data/OpenAlex-parquet/')
    staged: Path = Path('/home/fnoble/data/staged/')
    max_len: int = 300
    train_start: datetime = datetime(2010, 1, 1)
    train_end: datetime = datetime(2015, 1, 1)
    test_start: datetime = datetime(2015, 1, 1)
    test_end: datetime = datetime(2016, 1, 1)

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
    milestones = [40, 80]
    
    sum = 0
    for x in milestones:
        sum += x
        assert config.train.epochs != sum

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.5, 
        end_factor=1.0,
        total_iters=milestones[0],
        last_epoch=-1 # Used to implement restart from checkpoint
    )

    cosine_oscilating_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        eta_min=config.train.lr * 0.6,
        last_epoch=-1, # Used to implement restart from checkpoint
    )

    exponential_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, 
        gamma=0.95,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [warmup_scheduler, cosine_oscilating_scheduler, exponential_scheduler],
        milestones,
        last_epoch=-1,
    )

    return scheduler

# Safety checks
if config.train.sample:
    assert config.train.sample % config.train.batch_size  == 0, 'sample % batch_size must == 0'

if config.train.sample and config.train.shuffle:
    raise ValueError('trian.shuffle and train.sample cannot both be true')


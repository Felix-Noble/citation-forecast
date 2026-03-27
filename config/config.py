import torch
from datetime import datetime
from dataclasses import dataclass

top_k = tuple(300 for _ in range(8))
top_k = (*top_k, 128, 128, 128, 128)
n_layers = len(top_k)
selector_heads = tuple(4 for _ in range(n_layers))
process_heads = tuple(4 for _ in range(n_layers))
r_layers = tuple(False if i not in (3,10) else True for i in range(n_layers))

@dataclass(frozen=True)
class LossConfig:
    beta: float = 1.0
    gamma: float = 3

@dataclass(frozen=True)
class ModelConfig:
    model_name: str = 'h_r_smooth'
    vocab_size: int = 201_088
    pad_token_id: int = 19999
    dtype: torch.dtype =  torch.float32
    top_k: tuple[int, ...] = top_k
    r_layers: tuple[bool, ...] = r_layers
    selector_heads: tuple[int, ...] = selector_heads
    process_heads: tuple[int, ...] = process_heads
    n_layers: int = n_layers
    max_len: int = 300
    embed_dim: int = 8
    hidden_dim: int = 8
    n_out: int = 6
    n_params_out: int = 1
    dropout: float = 0.05

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 200
    batch_size: int = 12
    grad_accumulation_steps: int = 50 # n batches until optimizer steps
    lr: float = 1e-4 
    lr_milestones: tuple[int, ...] = (40, 80)
    weight_decay: float = 0.9
    optimizer: str = 'AdamW'
    loss_fn: str = 'WassersteinCrossEntropyLoss'
    loss: LossConfig = LossConfig()
    eval_interval: int = 5
    checkpoint_interval: int = 10
    shuffle: bool = False
    sample: int | None = 100_032
    mat_mul_precision: str = 'high'
    dataset: str = 'psychology_clean'
    train_start: datetime = datetime(2010, 1, 1)
    train_end: datetime = datetime(2015, 1, 1)
    test_start: datetime = datetime(2015, 1, 1)
    test_end: datetime = datetime(2016, 1, 1)

@dataclass(frozen=True)
class LogConfig:
    file: str = 'ERROR'
    console: str = 'DEBUG'

@dataclass(frozen=False)
class Config:
    model: ModelConfig = ModelConfig() 
    train: TrainConfig = TrainConfig()
    logging: LogConfig = LogConfig()

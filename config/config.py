import torch
from datetime import date
from dataclasses import dataclass
from typing import NamedTuple

max_len = 300
#top_k = (*top_k, 128, 128, 128, 128)
n_layers = 16
process_heads = tuple(1 for _ in range(n_layers + 1))
r_layers = tuple(False if i not in () else True for i in range(n_layers))

@dataclass(frozen=True)
class LossConfig:
    beta: float = 1.0
    gamma: float = 3
    weights: tuple[float, ...] = (0.80, 1.3158)

@dataclass(frozen=True)
class ModelConfig:
    model_name: str = 'hr_ahead_binary'
    vocab_size: int = 201_088
    pad_token_id: int = 19999
    dtype: torch.dtype =  torch.float32
    r_layers: tuple[bool, ...] = r_layers
    process_heads: tuple[int, ...] = process_heads
    n_layers: int = n_layers
    max_len: int = max_len
    max_len_eval: int = 1000
    embed_dim: int = 8
    hidden_dim: int = 16
    n_out: int = 1
#    n_params_out: int = 0
    dropout: float = 0.05

@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 4
    batch_size: int = 1_024 
    grad_accumulation_steps: int = 1  # n batches until optimizer steps
    lr: float = 1e-3 
    lr_milestones: tuple[int, ...] = (0,)
    warmup_start_factor: float = 1e-4
    lr_eta_min: float = 1e-6
    weight_decay: float = 1e-3
    optimizer: str = 'AdamW'
    loss_fn: str = 'BinaryCrossEntropyLoss'
    loss: LossConfig = LossConfig()
    eval_interval: int = 1
    checkpoint_interval: int = 1
    shuffle: bool = True
    sample: int | None =None
    mat_mul_precision: str = 'high'
    train_dataset: str = 'all-licensed-1800-2027-lowercase-T2A2'
    test_dataset: str = 'all-licensed-1800-2027-lowercase-T2A2'
#    train_dataset: str = 'all-1800-2027-lowercase-T2'
#    test_dataset: str = 'all-1800-2027-lowercase-T2'

    dataset_class: str = 'binarythresholddataset'
    train_start: date = date(1800, 1, 1)
    train_end: date = date(2020, 1, 1)
    test_start: date = date(2020, 1, 1)
    test_end: date = date(2021, 1, 1)
    dataset_kwargs: str = "{'theta': 0.7}"

@dataclass(frozen=True)
class LogConfig:
    file: str = 'ERROR'
    console: str = 'DEBUG'

@dataclass(frozen=False)
class Config:
    model: ModelConfig = ModelConfig() 
    train: TrainConfig = TrainConfig()
    logging: LogConfig = LogConfig()

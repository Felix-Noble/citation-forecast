from dataclasses import dataclass
from datetime import date

import torch

from training.trainers import ClassifierTrainer, Trainer


@dataclass(frozen=True)
class LossConfig:
    beta: float = 0.1
    weights: None = None
    # weights: tuple[float, ...] = (0.80, 1.3158)


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "abstractorLM"
    vocab_size: int = 201_088
    pad_token_id: int = 19999
    dtype: torch.dtype = torch.float32
    process_heads: int = 1
    n_layers: int = 8
    n_abstractions: int = 6
    abstraction_heads: int = 12
    abstracted_seq_len: int = 300
    max_len: int = 1000
    max_len_eval: int = 1000
    embed_dim: int = 16
    hidden_dim: int = 32
    n_out: int = 201_088
    #    n_params_out: int = 0
    dropout: float = 0.05


@dataclass(frozen=True)
class TrainConfig:
    trainer: type[Trainer] = ClassifierTrainer
    epochs: int = 100
    batch_size: int = 64
    grad_accumulation_steps: int = 1  # n batches until optimizer steps
    lr: float = 1e-5
    lr_milestones: tuple[int, ...] = (1,)
    warmup_start_factor: float = 1e-4
    lr_eta_min: float = 1e-6
    weight_decay: float = 1e-3
    optimizer: str = "AdamW"
    loss_fn: str = "CrossEntropyLoss"
    loss: LossConfig = LossConfig()
    eval_interval: int = 1
    checkpoint_interval: int = 1
    shuffle: bool = True
    sample: int | None = None
    mat_mul_precision: str = "high"
    train_dataset: str = "wikiTestShort"
    test_dataset: str = "wikiTestShort"
    #    train_dataset: str = 'all-1800-2027-lowercase-T2'
    #    test_dataset: str = 'all-1800-2027-lowercase-T2'

    dataset_class: str = "generativepretraindataset"
    # train_start: date = date(1800, 1, 1)
    # train_end: date = date(2020, 1, 1)
    # test_start: date = date(2020, 1, 1)
    # test_end: date = date(2021, 1, 1)
    train_start: date | None = None
    train_end: date | None = None
    test_start: date | None = None
    test_end: date | None = None

    dataset_kwargs: str = "{'n_forward': 1}"


@dataclass(frozen=True)
class LogConfig:
    file: str = "ERROR"
    console: str = "DEBUG"


@dataclass(frozen=False)
class Config:
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    logging: LogConfig = LogConfig()

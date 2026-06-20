from typing import NamedTuple

from training.trainers import ClassifierTrainer


class LossConfig(NamedTuple):
    weights: list[float] | None


# exported config
trainer = ClassifierTrainer
epochs: int = 100
batch_size: int = 2
accumulation_steps: int = 1  # n batches until optimizer steps
lr: float = 1e-5
lr_milestones: tuple[int, ...] = (1,)
warmup_start_factor: float = 1e-4
lr_eta_min: float = 1e-6
weight_decay: float = 1e-3
optimizer: str = "AdamW"
loss_fn: str = "BinaryCrossEntropyLoss"
loss = LossConfig(
    weights=None,
)
eval_interval: int = 1
checkpoint_interval: int = 1

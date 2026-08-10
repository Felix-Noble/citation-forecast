# Legacy singleton training hyperparameters.
# Phase 3 keeps these values so the rewired apps continue to run while the
# first experiment file (Phase 4) is being authored.  This module is deleted in
# Phase 7.

epochs: int = 8
accumulation_steps: int = 1  # n batches until optimizer steps
lr: float = 1e-3
lr_milestones: tuple[int, ...] = (0,)
warmup_start_factor: float = 1e-5
lr_eta_min: float = 1e-6
weight_decay: float = 1e-3
optimizer: str = "AdamW"
loss_fn: str = "BinaryCrossEntropyLoss"
eval_interval: int = 1
checkpoint_interval: int = 2

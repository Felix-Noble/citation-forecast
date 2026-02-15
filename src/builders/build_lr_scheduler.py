from config import Config, config
import torch

def build_lr_scheduler(
    optimizer,
    config: Config = config
):
    milestones = config.train.lr_milestones
    
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


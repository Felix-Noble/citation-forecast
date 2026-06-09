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
        start_factor=config.train.warmup_start_factor, 
        end_factor=1.0,
        total_iters=milestones[0],
        last_epoch=-1 # Used to implement restart from checkpoint
    )

    cosine_oscilating_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.train.epochs - config.train.lr_milestones[0],
        T_mult=1,
        eta_min=config.train.lr_eta_min,
        last_epoch=-1, # Used to implement restart from checkpoint
    )

    schedulers = [warmup_scheduler, cosine_oscilating_scheduler ]
    milestones = list(milestones)
    for i, milestone in enumerate(milestones):
        if milestone < 1:
            _ = milestones.pop(i)
            _ = schedulers.pop(i)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers,
        milestones,
        last_epoch=-1,
    )

    return scheduler


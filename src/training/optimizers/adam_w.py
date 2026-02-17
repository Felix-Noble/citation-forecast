from ._registry import optim_registry
import torch

@optim_registry('AdamW')
def adamw_wrapper(**kwargs) -> torch.optim.AdamW:
    return torch.optim.AdamW(**kwargs)

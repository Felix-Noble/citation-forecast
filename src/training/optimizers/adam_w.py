from ._registry import optim_registry
import torch

@optim_registry('AdamW')
class RegisteredAdamW(torch.optim.AdamW):
    pass

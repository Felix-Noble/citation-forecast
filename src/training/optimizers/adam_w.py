from utils import component
import torch


@component
class RegisteredAdamW(torch.optim.AdamW):
    pass

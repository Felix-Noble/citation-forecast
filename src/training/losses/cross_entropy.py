from ._registry import loss_registry
import torch.nn as nn

@loss_registry('CrossEntropyLoss')
class RegisteredCrossEntropyLoss(nn.CrossEntropyLoss):
    pass

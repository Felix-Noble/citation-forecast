# >>> build_helper:auto (do not edit)
from .MAE import MAE
from .MAE_var import MAE_Var
from .binary_cross_entropy import BinaryCrossEntropyLoss
from .cross_entropy import CrossEntropyLoss
from .wasserstein_cross_entropy import WassersteinCrossEntropyLoss
from .wasserstein_entropy import WassersteinEntropyLoss
from .wasserstein_loss import WassersteinLoss
from .wasserstein_sigma import WassersteinSigmaLoss

__all__ = [
    "MAE",
    "MAE_Var",
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "WassersteinCrossEntropyLoss",
    "WassersteinEntropyLoss",
    "WassersteinLoss",
    "WassersteinSigmaLoss",
]
# <<< build_helper:auto

from .entropy import norm_entropy_loss

__all__.append("norm_entropy_loss")

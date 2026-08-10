# >>> build_helper:auto (do not edit)
from .binary_cross_entropy import BinaryCrossEntropyLoss
from .cross_entropy import CrossEntropyLoss
from .MAE import MAE
from .MAE_var import MAE_Var
from .wasserstein_cross_entropy import WassersteinCrossEntropy
from .wasserstein_entropy import WassersteinEntropyLoss
from .wasserstein_loss import WassersteinLoss
from .wasserstein_sigma import WassersteinSigmaLoss

__all__ = [
    "BinaryCrossEntropyLoss",
    "CrossEntropyLoss",
    "MAE",
    "MAE_Var",
    "WassersteinCrossEntropy",
    "WassersteinEntropyLoss",
    "WassersteinLoss",
    "WassersteinSigmaLoss",
]
# <<< build_helper:auto

from .entropy import norm_entropy_loss

__all__.append("norm_entropy_loss")

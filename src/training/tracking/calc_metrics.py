import torch
from torch import Tensor

from ..losses import norm_entropy_loss


def calc_metrics(config, probs: Tensor, targets: Tensor) -> dict[str, Tensor]:
    metrics: dict[str, Tensor] = {}
    metrics["entropy"] = norm_entropy_loss(probs)

    target_ont_hot: Tensor = torch.nn.functional.one_hot(
        targets, num_classes=config.model.n_out
    )
    metrics["mae"] = torch.mean(
        torch.abs(target_ont_hot - probs)
    )  # normalised mean absolute error

    return metrics

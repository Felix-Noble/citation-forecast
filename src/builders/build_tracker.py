import torch

import training.tracking as tracking

_tracker = tracking.BinaryClassificationTracker


def build_eval_tracker(
    device: torch.device,
    dtype: torch.dtype,
    **kwargs,
) -> tracking.MetricTracker:
    metric_tracker = _tracker(
        dtype=dtype,
        device=device,
        **kwargs,
    )
    return metric_tracker


def build_train_tracker(
    device: torch.device,
    dtype: torch.dtype,
    **kwargs,
) -> tracking.MetricTracker:
    metric_tracker = _tracker(
        dtype=dtype,
        device=device,
        **kwargs,
    )
    return metric_tracker

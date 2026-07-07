import torch

import training.tracking as tracking
from training.tracking.metric_tracker import StoreParams

_tracker = tracking.BinaryClassificationTracker


def build_eval_tracker(
    device: torch.device,
    dtype: torch.dtype,
    config,
    **kwargs,
) -> tracking.MetricTracker:
    param_tuple = (
        build_tracker_params(name="val_ids", device=device),
        build_tracker_params(name="val_logits", device=device),
        build_tracker_params(name="val_probs", device=device),
        build_tracker_params(name="val_y", device=torch.device("cpu")),
        build_tracker_params(name="val_y_orig", device=torch.device("cpu")),
        build_tracker_params(name="val_loss", device=device),
        build_tracker_params(name="val_sigma", device=device),
        build_tracker_params(name="val_preds", device=device),
        build_tracker_params(name="val_sigma", device=device),
    )
    metric_tracker = _tracker(
        param_tuple,
        dtype=dtype,
        config=config,
        device=device,
        **kwargs,
    )
    return metric_tracker


def build_train_tracker(
    device: torch.device,
    dtype: torch.dtype,
    config,
    **kwargs,
) -> tracking.MetricTracker:
    param_tuple = (
        build_tracker_params(name="train_logits", device=device),
        build_tracker_params(name="train_probs", device=device),
        build_tracker_params(name="train_y", device=torch.device("cpu")),
        build_tracker_params(name="train_y_orig", device=torch.device("cpu")),
        build_tracker_params(name="train_loss", device=device),
        build_tracker_params(name="train_ids", device=device),
        build_tracker_params(name="train_preds", device=device),
        build_tracker_params(name="train_sigma", device=device),
        build_tracker_params(name="val_ids", device=device),
        build_tracker_params(name="val_logits", device=device),
        build_tracker_params(name="val_probs", device=device),
        build_tracker_params(name="val_y", device=torch.device("cpu")),
        build_tracker_params(name="val_y_orig", device=torch.device("cpu")),
        build_tracker_params(name="val_loss", device=device),
        build_tracker_params(name="val_sigma", device=device),
        build_tracker_params(name="val_preds", device=device),
        build_tracker_params(name="val_sigma", device=device),
    )
    metric_tracker = _tracker(
        param_tuple,
        dtype=dtype,
        config=config,
        device=device,
        buffer=False,
        **kwargs,
    )
    return metric_tracker


def build_tracker_params(
    name: str,
    device: torch.device,
):
    params = StoreParams(
        name,
        batch_shape=(100, 5),
        buffer_size=100,
        buffer_device=device,
        max_store=-1,
        n_examples=-1,
    )
    return params

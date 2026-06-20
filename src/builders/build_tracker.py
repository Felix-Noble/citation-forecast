import torch

from training.tracking import BinaryClassificationTracker, MetricTracker
from training.tracking.metric_tracker import StoreParams


def build_eval_tracker(
    device: torch.device,
    dtype: torch.dtype,
    config,
    **kwargs,
) -> MetricTracker:
    param_tuple = (
        build_tracker_params(name="test_ids", device=device),
        build_tracker_params(name="test_logits", device=device),
        build_tracker_params(name="test_probs", device=device),
        build_tracker_params(name="test_y", device=torch.device("cpu")),
        build_tracker_params(name="test_loss", device=device),
    )
    metric_tracker = MetricTracker(
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
) -> MetricTracker:
    param_tuple = (
        build_tracker_params(name="test_ids", device=device),
        build_tracker_params(name="train_logits", device=device),
        build_tracker_params(name="train_probs", device=device),
        build_tracker_params(name="train_y", device=torch.device("cpu")),
        build_tracker_params(name="train_loss", device=device),
        build_tracker_params(name="train_ids", device=device),
        build_tracker_params(name="test_logits", device=device),
        build_tracker_params(name="test_probs", device=device),
        build_tracker_params(name="test_y", device=torch.device("cpu")),
        build_tracker_params(name="test_loss", device=device),
    )
    metric_tracker = MetricTracker(
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

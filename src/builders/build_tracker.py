from config import Config, config
from src.training.tracking import ClassificationTracker
from src.training.tracking.classification_tracker import StoreParams
import torch

def build_eval_tracker(
        device: torch.device,
        dtype: torch.dtype,
        config: Config = config
                  ) -> ClassificationTracker:
       param_tuple = (
            build_tracker_params(name='test_logits', device=device),
            build_tracker_params(name='test_probs', device=device),
            build_tracker_params(name='test_y', device=torch.device('cpu')),
            build_tracker_params(name='test_loss', device=device),
               ) 
       metric_tracker = ClassificationTracker(
                param_tuple,
                dtype=dtype,
                device=device,
                buffer=False,
                )
       return metric_tracker

def build_train_tracker(
        device: torch.device,
        dtype: torch.dtype,
        config: Config = config
                  ) -> ClassificationTracker:
    param_tuple = (
            build_tracker_params(name='train_logits', device=device),
            build_tracker_params(name='train_probs', device=device),
            build_tracker_params(name='train_y', device=torch.device('cpu')),
            build_tracker_params(name='train_loss', device=device),

            build_tracker_params(name='test_logits', device=device),
            build_tracker_params(name='test_probs', device=device),
            build_tracker_params(name='test_y', device=torch.device('cpu')),
            build_tracker_params(name='test_loss', device=device),
            )
    metric_tracker = ClassificationTracker(
        param_tuple,
        dtype=dtype,
        device=device,
        buffer=False,
        )
    return metric_tracker

def build_tracker_params(
        name: str,
        device: torch.device,
        config: Config = config, 
        ):
    params = StoreParams(
        name,
        batch_shape=(config.train.batch_size, 5),
        buffer_size=100,
        buffer_device=device,
        max_store=-1,
        n_examples=-1,
    )
    return params

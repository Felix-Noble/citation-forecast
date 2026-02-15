from config import Config, config
from ..trackers import StoreParams
import torch

def build_tracker_params(
        device: torch.device,
        config: Config = config, 
):
    gpu = device
    cpu = torch.device('cpu')
    train_logits = StoreParams(
        'train_logits',
        batch_shape=(config.train.batch_size, 5),
        buffer_size=100,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )
    train_y = StoreParams(
        'train_y',
        batch_shape=(config.train.batch_size, ),
        buffer_size=100,
        buffer_device=cpu,
        max_store=-1,
        n_examples=-1,
    )

    train_loss = StoreParams(
        'train_loss',
        batch_shape=(config.train.batch_size, ),
        buffer_size=100,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )

    test_logits = StoreParams(
        'test_logits',
        batch_shape=(config.train.batch_size, 5),
        buffer_size=100,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )

    test_y = StoreParams(
        'test_y',
        batch_shape=(config.train.batch_size, ),
        buffer_size=100,
        buffer_device=cpu,
        max_store=-1,
        n_examples=-1,
    )

    test_loss = StoreParams(
        'test_loss',
        batch_shape=(config.train.batch_size, ),
        buffer_size=100,
        buffer_device=gpu,
        max_store=-1,
        n_examples=-1,
    )
    return train_logits, train_y, train_loss, test_logits, test_y, test_loss

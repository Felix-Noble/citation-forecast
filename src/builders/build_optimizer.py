from dataclasses import asdict
from config import Config, config
from src.utils import Registry
from src.training.optimizers import optim_registry
import torch

optim_builder_registry = Registry()

@optim_builder_registry('AdamW')
def build_adam_w(
        params: torch.nn.parameter.Parameter,
        config: Config = config,
        **kwargs,
        ) -> torch.optim.Optimizer:
    """ # Builds AdamW
        ### overides config args if custom kwargs provided
        ## Args:
            lr: learning rate 
            weight_decay: weight decay value, between 0-1, higher > more punishing 
        """
    args = {k: v for k,v in asdict(config.train).items() if k in ('lr', 'weight_decay')}
    args.update(kwargs) 

    return optim_registry[config.train.optimizer](
            params=params,
            **args,
            )

def build_optimizer(
        params: torch.nn.parameter.Parameter,
        config: Config = config,
        **kwargs,
                    ) -> torch.optim.Optimizer:
    """ # Builds optimizer based on registry entry
        ### either config and or kwargs must contain all kwargs passed to optimizer specific func
        ## Args:
            params: torch Module parameters
            config: project config 
            kwargs: optimizer specific arguments (overides config values)
    """
    return optim_builder_registry[config.train.optimizer](
            params=params,
            config=config,
            **kwargs,
            )

# src/mlflow/log_params.py 
from config import Config, config
import mlflow
from ._get_scheduler_params import _get_scheduler_params

def log_params(
        dataset,
        scheduler,
        config: Config = config
               ):

    param_dict = {}
    for k,v in config.__dict__.items():
        mod_params = {f'{k}-{sk}': val for sk, val in v.__dict__.items()}
        param_dict.update(mod_params)

    param_dict['dataset'] =  dataset
    scheduler_params = _get_scheduler_params(scheduler)
    param_dict.update({f'scheduler-{k}':v for k,v in scheduler_params.items()})

    mlflow.log_params(param_dict)


# src/mlflow/log_params.py 
from config import Config, config
import mlflow
from ._get_scheduler_params import _get_scheduler_params

def log_params(
        train_dataset_name: str,
        train_dataset,
        test_dataset_name: str,
        test_dataset,
        scheduler,
        config: Config = config
               ):

    param_dict = {}
    for k,v in config.__dict__.items():
        mod_params = {f'{k}-{sk}': val for sk, val in v.__dict__.items()}
        param_dict.update(mod_params)

    param_dict['train_dataset'] = {
            'name': train_dataset_name,
            'n_examps': len(train_dataset)
            }
    param_dict['test_dataset'] = {
            'name': test_dataset_name,
            'n_examps': len(test_dataset)
            }

    scheduler_params = _get_scheduler_params(scheduler)
    param_dict.update({f'scheduler-{k}':v for k,v in scheduler_params.items()})

    mlflow.log_params(param_dict)


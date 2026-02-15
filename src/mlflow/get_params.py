# src/mlflow/get_params.py 

def get_params(params: dict):
    """ Get params from dict, ignore clutter """
    ignore_keys = ['name', 'optimizer', 'last_epoch', 'step_count']
    return {k:v for k,v in params.items() if k not in ignore_keys and not str(k).startswith('_')}.items()

def get_scheduler_params(scheduler):
    """ Fetch and format scheduler (and sub-scheduler) parameters """
    params = {}
    params['name'] = type(scheduler).__name__
    if hasattr(scheduler, '_schedulers'):
        params['names'] = []
        params['milestones'] = scheduler._milestones
        for i, sub_scheduler in enumerate(scheduler._schedulers):
            params['names'].append(type(sub_scheduler).__name__)
            sub_params = get_scheduler_params(sub_scheduler)
            params.update({f'{i}-{sub_params["name"]}-{k}':v for k,v in get_params(sub_params)})
    else:
        params.update(get_params(scheduler.__dict__))

    return params

def get_optim_params(optimizer):
    """ Placeholder for optimizer param getter """
    # optim params currenty in config.train
    pass


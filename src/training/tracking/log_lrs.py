# src/training/tracking/log_lrs.py 

def log_lrs(
        scheduler,
        epoch: int,
        ) -> None:
    import mlflow
    lrs = scheduler.get_last_lr()
    assert len(lrs) > 0, 'No lrs in schedueler'
    if len(lrs) > 1:
        lr_metrics = {f'lr-{i}': lr for i,lr in enumerate(lrs)}
        mlflow.log_metrics(lr_metrics, step=epoch)
    else:
        mlflow.log_metric('lr', lrs[0], step=epoch)

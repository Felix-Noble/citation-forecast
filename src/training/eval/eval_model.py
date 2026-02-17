# src/training/eval/eval_model.py 
from ..tracking import ClassificationTracker
from config import config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def eval_model(
    model: nn.Module,
    loss_fn,
    dataloader: DataLoader,
    example_progress,
    examples_done,
    stream_context,
    stream_sync,
    metric_tracker: ClassificationTracker,
    device: torch.device,
    config = config,
) -> None:

    model.eval()
    with torch.no_grad():
        for batch_i, (X, y, mask) in enumerate(dataloader):
            metric_tracker.process_values((y,), ('test_y',))
            with stream_context:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
            stream_sync()

            logits, out, sigma = model(X, mask)
            loss = loss_fn(out.squeeze(-1), sigma, y)

            loss_cpu = loss.detach().cpu().item()
            sigma_cpu = torch.mean(sigma.detach()).item()

            metric_tracker.log_metric('test_loss', loss_cpu, X.shape[0])
            metric_tracker.log_metric('test_sigma', sigma_cpu, X.shape[0])
            metric_tracker.process_values((logits.detach(), ), ('test_logits', ))
            example_progress.update(examples_done, advance=config.train.batch_size)

    _ = metric_tracker.calc_metrics(
        logit_store_name='test_logits',
        y_store_name='test_y',
        prefix='test'
    )


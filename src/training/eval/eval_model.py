# src/training/eval/eval_model.py 
from ..tracking import BinaryClassificationTracker 
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
    metric_tracker: BinaryClassificationTracker,
    device: torch.device,
    config = config,
) -> None:

    model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader):
            y = batch.y
            metric_tracker.process_values((y,), ('test_y',))
            with stream_context:
                X = batch.X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = batch.mask.to(device, non_blocking=True)
            stream_sync()

            logits, probs, sigma = model(X, mask)
            loss = loss_fn(logits=logits, probs=probs, sigma=sigma, target=y)

            loss_cpu = loss.detach().cpu().item()
            sigma_cpu = torch.mean(sigma.detach()).item()

            metric_tracker.log_metric('test_loss', loss_cpu, X.shape[0])
            metric_tracker.log_metric('test_sigma', sigma_cpu, X.shape[0])
            if batch.id is not None:
                metric_tracker.log_metric('test_id', batch.id, X.shape[0])

            metric_tracker.process_values((logits.detach(), probs.detach()), ('test_logits', 'test_probs'))
            example_progress.update(examples_done, advance=config.train.batch_size)

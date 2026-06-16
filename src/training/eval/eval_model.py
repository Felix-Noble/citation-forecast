# src/training/eval/eval_model.py 
from src.training.losses.entropy import norm_entropy_loss
from ..tracking import BinaryClassificationTracker 
from src.data.datasets.polars_dataset import Output
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
        for batch_i, batch_dict in enumerate(dataloader):
            batch = Output(**batch_dict)
            y = batch.y
            #metric_tracker.process_values((y,), ('test_y',))
            with stream_context:
                x = batch.x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = batch.mask.to(device, non_blocking=True)
            stream_sync()

            logits, probs = model(x, mask)
            loss = loss_fn(logits=logits, probs=probs, target=y)

            loss_cpu = loss.detach().cpu().item()

            metric_tracker.log_metric('test_loss', loss_cpu, x.shape[0])
            if not torch.any(torch.isnan(batch.id)):
                metric_tracker.log_metric('test_id', batch.id, x.shape[0])

            #metric_tracker.process_values((logits.detach(), probs.detach()), ('test_logits', 'test_probs'))
            example_progress.update(examples_done, advance=config.train.batch_size)
            #entropy = norm_entropy_loss(probs.detach().cpu())
            #metric_tracker.log_metric('test_entropy', entropy.item(), x.shape[0])

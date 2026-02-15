# src/eval/eval.py 
from ..tracking import ClassificationTracker
from config import config
import torch
import torch.nn as nn
import torch.utils.data.DataLoader

def eval(
    model: nn.Module,
    loss_fn,
    dataloader: DataLoader,
    example_progress,
    examples_done,
    stream,
    metric_tracker: ClassificationTracker,
    device: torch.device,
    config = config,
) -> None:

    model.eval()
    with torch.no_grad():
        for batch_i, (X, y, mask) in enumerate(dataloader):
            metric_tracker.process_values((y,), ('test_y',))
            with torch.cuda.stream(stream):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
            torch.cuda.synchronize()

            out = model(X, mask)
            loss = loss_fn(out.squeeze(-1), y)

            #loss_cpu = loss.detach().item()
            #metric_tracker.log_metric('test_loss', loss_cpu, X.shape[0])
            metric_tracker.process_values((out.detach(), loss.detach()), ('test_logits', 'test_loss'))
            example_progress.update(examples_done, advance=config.train.batch_size)

    _ = metric_tracker.calc_metrics(
        logit_store_name='test_logits',
        y_store_name='test_y',
        prefix='test'
    )


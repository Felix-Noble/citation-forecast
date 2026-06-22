# src/training/eval/eval_model.py
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from training.losses.entropy import norm_entropy_loss


class input_tuple(NamedTuple):
    x: Tensor
    y: Tensor
    mask: Tensor
    weight: Tensor


def eval_model[T_Batch](
    model: nn.Module,
    loss_fn,
    dataloader: DataLoader[T_Batch],
    example_progress,
    examples_done,
    stream_context,
    stream_sync,
    metric_tracker,
    device: torch.device,
    config,
) -> None:

    model.eval()
    with torch.no_grad():
        for batch_i, batch in enumerate(dataloader):
            y = batch.y
            x = batch.x
            mask = batch.mask
            id = batch.id
            metric_tracker.process_values((y, id), ("test_y", "test_ids"))
            with stream_context:
                batch = input_tuple(
                    x=batch.x.to(device, non_blocking=True),
                    y=batch.y.to(device, non_blocking=True),
                    mask=batch.mask.to(device, non_blocking=True),
                    weight=batch.weight.to(device, non_blocking=True),
                )
            stream_sync()

            out = model(batch)
            loss = loss_fn(output=out, batch=batch)

            loss_cpu = loss.detach().cpu().item()

            metric_tracker.log_metric("test_loss", loss_cpu, x.shape[0])
            #            metric_tracker.log_metric(
            #                "test_sigma", torch.mean(out.sigma.detach()).item(), x.shape[0]
            #            )

            metric_tracker.process_values(
                (out.logits.detach(), out.probs.detach()), ("test_logits", "test_probs")
            )
            example_progress.update(
                examples_done, advance=config.data.test.loader.batch_size
            )

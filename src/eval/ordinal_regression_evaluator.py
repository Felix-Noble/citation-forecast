from typing import NamedTuple

import torch
from torch import Tensor

from .base_evaluator import BaseEvaluator, EvaluatorConfig, EvaluatorProtocol


class Batch(NamedTuple):
    id: Tensor
    x: Tensor
    graph_x: Tensor
    y: Tensor
    mask: Tensor
    graph_x_mask: Tensor
    weight: Tensor


class OrdinalRegressionEvaluator(
    BaseEvaluator, EvaluatorProtocol[EvaluatorConfig, Batch]
):
    config: type[EvaluatorConfig] = EvaluatorConfig

    def move_to_device(self, batch: Batch) -> Batch:
        with self.stream:
            out = Batch(
                id=batch.id,
                x=batch.x.to(self.device, non_blocking=True),
                y=batch.y.to(self.device, non_blocking=True),
                graph_x=batch.graph_x.to(self.device, non_blocking=True),
                mask=batch.mask.to(self.device, non_blocking=True),
                graph_x_mask=batch.graph_x_mask.to(self.device, non_blocking=True),
                weight=batch.weight.to(self.device, non_blocking=True),
            )
        self.stream_sync()
        return out

    def _step(self, batch: Batch) -> float:
        self.tracker.process_values((batch.id.clone(),), (f"{self.prefix}_ids",))
        self.tracker.process_values((batch.y.clone(),), (f"{self.prefix}_y",))
        batch = self.move_to_device(batch)
        with torch.no_grad():
            out = self.model.forward(batch)
            loss = self.loss_fn(out, batch)
            loss_cpu = loss.detach().clone()
        # self.tracker.process_output(out, step=self.epoch_i)
        loss_cpu = loss_cpu.item()
        self.tracker.log_metric(f"{self.prefix}_loss", loss_cpu, batch.x.shape[0])
        self.tracker.process_values(
            (out.probs.detach().clone(), out.logits.detach().clone()),
            (f"{self.prefix}_probs", f"{self.prefix}_logits"),
        )

        return loss_cpu

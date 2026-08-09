from typing import NamedTuple

import torch
from torch import Tensor

from .base_evaluator import BaseEvaluator, EvaluatorConfig, EvaluatorProtocol


class Batch(NamedTuple):
    id: Tensor
    x: Tensor
    y: Tensor
    mask: Tensor
    weight: Tensor


class ClassificationEvaluator(BaseEvaluator, EvaluatorProtocol[EvaluatorConfig, Batch]):
    config: type[EvaluatorConfig] = EvaluatorConfig

    def _to_device(self, val, device):
        # Handles Tensors or nested custom objects that implement .to()
        if hasattr(val, "to") and callable(val.to):
            return val.to(device, non_blocking=True)
        return val

    def move_to_device(self, batch: Batch) -> Batch:
        with self.stream:
            # type(batch)(*...) calls the namedtuple constructor with converted items
            out = type(batch)(*(self._to_device(item, self.device) for item in batch))
        self.stream_sync()
        return out

    def _step(self, batch: Batch) -> float:
        self.tracker.process_values((batch.y.clone(),), (f"{self.prefix}_y",))
        self.tracker.process_values((batch.id.clone(),), (f"{self.prefix}_ids",))
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

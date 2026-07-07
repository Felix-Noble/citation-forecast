from typing import NamedTuple

import torch
from torch import Tensor

from .base_evaluator import BaseEvaluator, Batch, EvaluatorConfig, EvaluatorProtocol


class HSRegressionEvaluator(BaseEvaluator, EvaluatorProtocol[EvaluatorConfig, Batch]):
    config: type[EvaluatorConfig] = EvaluatorConfig

    def move_to_device(self, batch: Batch) -> Batch:
        with self.stream:
            out = Batch(
                x=batch.x.to(self.device, non_blocking=True),
                y=batch.y.to(self.device, non_blocking=True),
                mask=batch.mask.to(self.device, non_blocking=True),
            )
        self.stream_sync()
        return out

    def _step(self, batch: Batch) -> float:
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
            (out.prediction.detach().clone(), out.sigma.detach().clone()),
            (f"{self.prefix}_preds", f"{self.prefix}_sigma"),
        )

        return loss_cpu

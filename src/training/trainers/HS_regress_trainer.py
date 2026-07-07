from typing import NamedTuple

import torch
from torch import Tensor

from .base_trainer import BaseTrainer, TrainerConfig, TrainerProtocol


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    mask: Tensor


class HSRegressTrainer(BaseTrainer, TrainerProtocol[TrainerConfig, Batch]):
    config: type[TrainerConfig] = TrainerConfig

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
        self.tracker.process_values((batch.y.clone(),), ("train_y",))
        batch = self.move_to_device(batch)
        out = self.model.forward(batch)
        loss = self.loss_fn(out, batch)
        loss_cpu = loss.detach().clone()
        loss = loss / self.accumulation_steps
        loss.backward()
        if (
            self.batch_i + 1
        ) % self.accumulation_steps == 0 or self.examples_per_epoch - batch.x.shape[
            0
        ] == self.batch_steps_i:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        else:
            print("no optim step", self.batch_i)
            pass
        # self.tracker.process_output(out, step=self.epoch_i)
        loss_cpu = loss_cpu.item()

        self.tracker.log_metric(f"{self.prefix}_loss", loss_cpu, batch.x.shape[0])

        self.tracker.process_values(
            (out.prediction.detach().clone(), out.sigma.detach().clone()),
            (f"{self.prefix}_preds", f"{self.prefix}_sigma"),
        )

        return loss_cpu

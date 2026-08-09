from typing import NamedTuple

from torch import Tensor

from .base_trainer import BaseTrainer, TrainerConfig, TrainerProtocol


class Batch(NamedTuple):
    x: Tensor
    y: Tensor
    mask: Tensor
    weight: Tensor


class ClassificationTrainer(BaseTrainer, TrainerProtocol[TrainerConfig, Batch]):
    config: type[TrainerConfig] = TrainerConfig

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
        self.tracker.log_metric("train_loss", loss_cpu, batch.x.shape[0])
        self.tracker.process_values(
            (out.logits.detach().clone(), out.probs.detach().clone()),
            ("train_logits", "train_probs"),
        )

        return loss_cpu

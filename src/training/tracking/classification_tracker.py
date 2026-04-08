from .metric_tracker import MetricTracker 
from typing import NamedTuple, override
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, balanced_accuracy_score, precision_score, recall_score, mean_absolute_error # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs] 
import torch
from pathlib import Path
from logging import getLogger
from src.utils.logging import setup_logger
from config import config

logger = getLogger(Path(__file__).stem)
_ = setup_logger(logger, config.logging)

class StoreParams(NamedTuple):
    name: str
    batch_shape: tuple[int, ...]
    buffer_size: int
    buffer_device: torch.device
    max_store: int
    n_examples: int
 
@dataclass 
class Store():
    name: str
    buffer: torch.Tensor
    store: list[torch.Tensor]
    buffer_cursor: torch.Tensor
    store_cursor: int

class MetricTuple(NamedTuple):
    " Metric Tuple: stores named metric scores / weight values for dataframe concatenations"
    score: float
    weight: float
    
class ClassificationTracker(MetricTracker):
    """
        Classification Metrics Tracker:

        Stores intermediate outputs on GPU, calcualtes results, displays them

        args:        
            output_shape: shape out output by model
            output_buffer_size: n. of outputs that will be stored in 'fast' buffer
            output_max_size: n. of outputs that will be stored before moving to CPU

"""
    @override
    def calc_metrics(self, 
                prefix: str = "",
                ) -> None:
        logits = self._gather_store(store_name=f'{prefix}_logits')
        probs = self._gather_store(store_name=f'{prefix}_probs')
        y_true = self._gather_store(store_name=f'{prefix}_y')

        if config.model.n_out == 1:
            # binary case 
            preds = torch.zeros_like(probs)
            preds[probs > 0.5] = 1
        else:
            preds = torch.argmax(
                probs,
                dim=1,
            ).squeeze(-1)

        if logits.size(0) != y_true.size(0):
            logger.error(f'Different n. examples in logits and y_true: logits shape: {logits.shape}, y_true shape:{y_true.shape}')
            return

        try:
            mae = mean_absolute_error(y_true, probs)
            self.log_metric(f'{prefix}_MAE', mae, preds.shape[0])
        except Exception as e:
            logger.error(e)

        try:
            balanced_accuracy = balanced_accuracy_score(y_true, preds)
            self.log_metric(f'{prefix}_balanced_accuracy', balanced_accuracy, preds.shape[0]) # pyright: ignore[reportArgumentType]

        except Exception as e:
            logger.error(e)
        try:
            roc_auc = roc_auc_score( # pyright: ignore[reportUnknownVariableType]
                y_true.long().numpy(), 
                probs.numpy(), 
                multi_class='ovo', 
                average='weighted')
            self.log_metric(f'{prefix}_roc_auc', roc_auc, probs.shape[0]) # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
        try:
            recall = recall_score(y_true.long().numpy(), preds.numpy())
            self.log_metric(f'{prefix}_recall', recall, probs.shape[0])
        except Exception as e:
            logger.error(e)
        try:
            precision = precision_score(y_true.long().numpy(), preds.numpy())
            self.log_metric(f'{prefix}_precision', precision, preds.shape[0])
        except Exception as e:
            logger.error(e)

        try: 
            precision, recall, _ = precision_recall_curve(
                y_true.long().numpy(),
                probs.numpy(),
                )
            precision = np.sort(precision)
            recall = np.flip(np.sort(recall))
            pr_auc = auc(precision, recall)
            self.log_metric(f'{prefix}_PR_AUC', pr_auc, probs.shape[0]) # pyright: ignore[reportArgumentType]
        except Exception as e:
            try:
                mssg = f'{e}\nrecall:{recall}\nprecision:{precision}'
            except:
                mssg = str(e)
            logger.error(mssg)

from .metric_tracker import MetricTracker 
from typing import NamedTuple, override
from dataclasses import dataclass
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_score, recall_score, mean_absolute_error # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs] 
import torch
import numpy as np
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
                     prefix: str,
                     step:  int
                     ) -> None:
        logits = self._gather_store(store_name=f'{prefix}_logits')
        probs = self._gather_store(store_name=f'{prefix}_probs')
        y_true = self._gather_store(store_name=f'{prefix}_y')
        self._export_plots(prefix, y_true, probs, step)
        if logits.size(0) != y_true.size(0):
            logger.error(f'Different n. examples in logits and y_true: logits shape: {logits.shape}, y_true shape:{y_true.shape}')
            return

        n_examples = probs.shape[0]
        try:
            mae = mean_absolute_error(y_true, probs)
            self.log_metric(f'{prefix}_MAE', mae, n_examples)
        except Exception as e:
            logger.error(e)
        try:
            roc_auc = roc_auc_score( # pyright: ignore[reportUnknownVariableType]
                y_true.long().numpy(), 
                probs.numpy(), 
                multi_class='ovo', 
                average='weighted')
            self.log_metric(f'{prefix}_roc_auc', roc_auc, n_examples) # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
 
        try: 
            pr_auc = average_precision_score(y_true.long().numpy(), probs.numpy())
            self.log_metric(f'{prefix}_PR_AUC', pr_auc, n_examples) # pyright: ignore[reportArgumentType]
        except Exception as e:
            try:
                mssg = f'{e}\nrecall:{recall}\nprecision:{precision}'
            except:
                mssg = str(e)
            logger.error(mssg)

        if config.model.n_out == 1:
            # binary case 
            preds = torch.zeros_like(probs)
            preds[probs > 0.5] = 1
        else:
            preds = torch.argmax(
                probs,
                dim=1,
            ).squeeze(-1)

        try:
            balanced_accuracy = balanced_accuracy_score(y_true, preds)
            self.log_metric(f'{prefix}_balanced_accuracy:50', balanced_accuracy, n_examples) # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
        try:
            recall = recall_score(y_true.long().numpy(), preds.numpy())
            self.log_metric(f'{prefix}_recall:50', recall, n_examples)
        except Exception as e:
            logger.error(e)
        try:
            precision = precision_score(y_true.long().numpy(), preds.numpy())
            self.log_metric(f'{prefix}_precision:50', precision, n_examples)
        except Exception as e:
            logger.error(e)
        
        if config.model.n_out != 1:
            return

        thetas = np.linspace(0, 1, num=100, endpoint=False)
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
                    
        try: 
            for theta in thetas:
                preds = torch.zeros_like(probs)
                preds[probs > theta] = 1
                preds = preds.numpy()
                
                balanced_accuracy = balanced_accuracy_score(y_true, preds)
                precision = precision_score(y_true.long().numpy(), preds)
                recall = recall_score(y_true.long().numpy(), preds)

                accuracy_scores.append(balanced_accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)

            accuracy_scores = np.array(accuracy_scores)

            high_accuracy_i = np.argmax(accuracy_scores)
            
            self.log_metric(f'{prefix}_best_accuracy', accuracy_scores[high_accuracy_i], n_examples) # pyright: ignore[reportArgumentType]
            self.log_metric(f'{prefix}_best_accuracy_theta', thetas[high_accuracy_i], n_examples) # pyright: ignore[reportArgumentType]
            self.log_metric(f'{prefix}_best_precision:best_acc', precision_scores[high_accuracy_i], n_examples) # pyright: ignore[reportArgumentType]
            self.log_metric(f'{prefix}_best_recall:best_acc', recall_scores[high_accuracy_i], n_examples) # pyright: ignore[reportArgumentType]
            
        except Exception as e:
            logger.error(f'theta: {theta} | error: {e}')

            

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import NamedTuple, override

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (  # pyright: ignore[reportUnknownVariableType, reportMissingTypeStubs]
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils.logging import setup_logger

from .metric_tracker import MetricTracker

logger = getLogger(__name__)
_ = setup_logger(logger)


class StoreParams(NamedTuple):
    name: str
    batch_shape: tuple[int, ...]
    buffer_size: int
    buffer_device: torch.device
    max_store: int
    n_examples: int


@dataclass
class Store:
    name: str
    buffer: torch.Tensor
    store: list[torch.Tensor]
    buffer_cursor: torch.Tensor
    store_cursor: int


class MetricTuple(NamedTuple):
    "Metric Tuple: stores named metric scores / weight values for dataframe concatenations"

    score: float
    weight: float


class BinaryClassificationTracker(MetricTracker):
    """
    Classification Metrics Tracker:

    Stores intermediate outputs on GPU, calcualtes results, displays them

    args:
        output_shape: shape out output by model
        output_buffer_size: n. of outputs that will be stored in 'fast' buffer
        output_max_size: n. of outputs that will be stored before moving to CPU

    """

    @override
    def _log_plots(
        self,
        prefix: str,
        y_true: np.ndarray,
        probs: np.ndarray,
        step: int,
    ) -> None:

        y_true = y_true.squeeze(-1)
        probs = probs.squeeze(-1)
        try:
            roc_plot = RocCurveDisplay.from_predictions(y_true, probs)
            mlflow.log_figure(
                roc_plot.figure_,
                f"{prefix}-plots/ROC/roc_curve-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(e)

        try:
            pr_plot = PrecisionRecallDisplay.from_predictions(
                y_true,
                probs,
                plot_chance_level=True,
            )
            mlflow.log_figure(
                pr_plot.figure_,
                f"{prefix}-plots/PR/pr_curve-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(e)

        # OutPut Probs histogram
        try:
            # 1. Calculate Proportions for the Legend
            total = len(y_true)
            pos_count = np.sum(y_true == 1)
            neg_count = np.sum(y_true == 0)

            pos_pct = (pos_count / total) * 100
            neg_pct = (neg_count / total) * 100

            # 2. Set the visual style
            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))

            # We pass the arrays directly here
            sns.histplot(
                x=probs.squeeze(),
                hue=y_true,
                multiple="stack",
                palette={0: "red", 1: "green"},
                bins=40,
                edgecolor="white",
                alpha=0.7,
                ax=ax,
            )

            # 4. Customizing the Legend with Proportions
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0], [0], color="green", lw=4, label=f"Positive (1): {pos_pct:.1f}%"
                ),
                Line2D(
                    [0], [0], color="red", lw=4, label=f"Negative (0): {neg_pct:.1f}%"
                ),
            ]

            ax.legend(
                handles=legend_elements, title="Label Distribution", loc="upper right"
            )

            # 5. Styling and Limits
            plt.xlim(0, 1)
            plt.xlabel("Classifier Output (Probability)", fontsize=11)
            plt.ylabel("Count", fontsize=11)
            plt.title(
                "Histogram of Output Probabilities with target composition",
                fontsize=13,
                pad=15,
            )
            plt.tight_layout()

            mlflow.log_figure(
                fig,
                f"{prefix}-plots/ProbHist/-histogram-step-{step}.png",
                save_kwargs={"dpi": 72},
            )
        except Exception as e:
            logger.error(str(e))

    @override
    def calc_metrics(self, prefix: str, step: int) -> None:
        logits = self._gather_store(store_name=f"{prefix}_logits")
        probs = self._gather_store(store_name=f"{prefix}_probs")
        y_true = self._gather_store(store_name=f"{prefix}_y")
        self._log_plots(prefix, y_true.long().numpy(), probs.numpy(), step)
        if logits.size(0) != y_true.size(0):
            logger.error(
                f"Different n. examples in logits and y_true: logits shape: {logits.shape}, y_true shape:{y_true.shape}"
            )
            return

        n_examples = probs.shape[0]
        try:
            mae = mean_absolute_error(y_true, probs)
            self.log_metric(f"{prefix}_MAE", mae, n_examples)
        except Exception as e:
            logger.error(e)
        try:
            roc_auc = roc_auc_score(  # pyright: ignore[reportUnknownVariableType]
                y_true.long().numpy(),
                probs.numpy(),
                multi_class="ovo",
                average="weighted",
            )
            self.log_metric(f"{prefix}_roc_auc", roc_auc, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)

        try:
            pr_auc = average_precision_score(y_true.long().numpy(), probs.numpy())
            self.log_metric(f"{prefix}_PR_AUC", pr_auc, n_examples)  # pyright: ignore[reportArgumentType]
        except Exception as e:
            try:
                mssg = f"{e}\nrecall:{recall}\nprecision:{precision}"
            except:
                mssg = str(e)
            logger.error(mssg)

        if self.config.model.n_out == 1:
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
            self.log_metric(
                f"{prefix}_balanced_accuracy:50", balanced_accuracy, n_examples
            )  # pyright: ignore[reportArgumentType]
        except Exception as e:
            logger.error(e)
        try:
            recall = recall_score(y_true.long().numpy(), preds.numpy())
            self.log_metric(f"{prefix}_recall:50", recall, n_examples)
        except Exception as e:
            logger.error(e)
        try:
            precision = precision_score(y_true.long().numpy(), preds.numpy())
            self.log_metric(f"{prefix}_precision:50", precision, n_examples)
        except Exception as e:
            logger.error(e)

        if self.config.model.n_out != 1:
            return

        thetas = np.linspace(0.25, 0.75, num=20, endpoint=False)
        accuracy_scores = []

        try:
            for theta in thetas:
                preds = torch.zeros_like(probs)
                preds[probs > theta] = 1
                preds = preds.numpy()

                balanced_accuracy = balanced_accuracy_score(y_true, preds)

                accuracy_scores.append(balanced_accuracy)

            accuracy_scores = np.array(accuracy_scores)
            high_accuracy_i = np.argmax(accuracy_scores)

            preds = torch.zeros_like(probs)
            preds[probs > thetas[high_accuracy_i]] = 1
            preds = preds.numpy()

            recall = recall_score(y_true, preds)
            precision = precision_score(y_true, preds)

            self.log_metric(
                f"{prefix}_best_accuracy", accuracy_scores[high_accuracy_i], n_examples
            )  # pyright: ignore[reportArgumentType]
            self.log_metric(
                f"{prefix}_best_accuracy_theta", thetas[high_accuracy_i], n_examples
            )  # pyright: ignore[reportArgumentType]
            self.log_metric(f"{prefix}_precision:best_acc", precision, n_examples)  # pyright: ignore[reportArgumentType]
            self.log_metric(f"{prefix}_recall:best_acc", recall, n_examples)  # pyright: ignore[reportArgumentType]

        except Exception as e:
            logger.error(f"theta: {theta} | error: {e}")

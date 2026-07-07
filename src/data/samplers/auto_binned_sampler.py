import numpy as np
import polars as pl
import torch
from torch.utils.data import WeightedRandomSampler


class AutoBinnedSampler(WeightedRandomSampler):
    """
    A custom WeightedRandomSampler that extracts a target column from a Polars DataFrame,
    bins its continuous values using NumPy, and calculates inverse-frequency weights.
    """

    def __init__(
        self, dataset, y_col_name, num_bins=100, replacement=True, num_samples=None
    ):
        # 1. Extract the column from the Polars DataFrame
        if hasattr(dataset, "df_y") and isinstance(dataset.df_y, pl.DataFrame):
            # Highly efficient zero-copy conversion to NumPy
            targets = dataset.df_y.get_column(y_col_name).to_numpy()
        elif hasattr(dataset, y_col_name):
            targets = getattr(dataset, y_col_name)
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu().numpy()
        else:
            raise AttributeError(
                f"Could not find Polars DataFrame '.df' or attribute '{y_col_name}' in dataset."
            )

        targets = dataset._format_y(torch.tensor(targets)).numpy()
        targets = np.asarray(targets, dtype=np.float32)

        # 2. Math-based uniform binning using NumPy
        min_val, max_val = targets.min(), targets.max()

        if max_val == min_val:
            # Handle edge case where all target values are identical
            bin_assignments = np.zeros_like(targets, dtype=np.int64)
        else:
            # Map values to integer bin indices: 0 to (num_bins - 1)
            bin_assignments = (
                (targets - min_val) / (max_val - min_val) * num_bins
            ).astype(np.int64)
            bin_assignments = np.clip(bin_assignments, 0, num_bins - 1)

        # 3. Calculate frequencies and inverse weights
        bin_counts = np.bincount(bin_assignments, minlength=num_bins)
        # Add epsilon to prevent division by zero on empty bins
        bin_weights = 1.0 / (bin_counts + 1e-6)

        # 4. Map bin weights back to every individual row sample
        sample_weights = bin_weights[bin_assignments]
        sample_weights_tensor = torch.from_numpy(sample_weights).double()

        # 5. Set default sample size if not provided
        if num_samples is None:
            num_samples = len(sample_weights_tensor)

        # 6. Initialize the parent PyTorch Sampler class
        super().__init__(
            weights=sample_weights_tensor,
            num_samples=num_samples,
            replacement=replacement,
        )

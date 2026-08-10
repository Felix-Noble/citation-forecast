"""Runtime context injected by the CLI apps.

Runtime values describe *how* a run executes (device, compile flags, subsample)
and are intentionally separated from experiment-relevant configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, kw_only=True)
class RunContext:
    """Runtime-only concerns passed to ``Experiment.build``."""

    device: torch.device
    dtype: torch.dtype
    compile_mode: str = ""
    fullgraph: bool = False
    subsample: int | None = None

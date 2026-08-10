"""Tracker contract re-export.

The concrete :class:`MetricTracker` implementation lives in
``metric_tracker.py``; this module preserves the import path used during
Phase 1 until the cleanup pass.
"""

from .metric_tracker import MetricTracker

__all__ = ["MetricTracker"]

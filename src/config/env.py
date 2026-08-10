"""Environment / machine settings.

Phase 0 introduces a typed :class:`Env` dataclass loaded from
``config/config.toml`` with optional CLI overrides.  A temporary PEP 562
``__getattr__`` shim keeps the legacy module-level names working for the
remaining old-code paths (``preprocess``, ``describe``, ``engineer`` and the
temporary train/eval wiring) until they are migrated in later phases.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.get_root_dir import get_root_dir


@dataclass(frozen=True, kw_only=True)
class Env:
    """Machine-specific settings that are not experiment-relevant."""

    tracking_uri: str
    raw_loc: Path
    staged_loc: Path
    artifact_loc: Path


_DEFAULT_ENV: Env | None = None


def _load_toml() -> dict[str, Any]:
    path = get_root_dir(markers=("pyproject.toml",)) / "config" / "config.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_env(overrides: dict[str, Any] | None = None) -> Env:
    """Load ``[env]`` from ``config/config.toml`` and apply CLI overrides.

    All four keys get CLI override flags (J5).  ``staged_loc`` is validated to
    exist at load time.
    """
    overrides = overrides or {}
    data = _load_toml()
    env_data = data.get("env", {})

    staged_loc = Path(
        overrides.get("staged_loc", env_data.get("staged_loc", "/tmp/staged"))
    )
    if not staged_loc.exists():
        raise FileNotFoundError(
            f"staged_loc does not exist: {staged_loc}"
        )

    return Env(
        tracking_uri=overrides.get(
            "tracking_uri", env_data.get("tracking_uri", "http://127.0.0.1:5000")
        ),
        raw_loc=Path(
            overrides.get("raw_loc", env_data.get("raw_loc", "/tmp/raw"))
        ),
        staged_loc=staged_loc,
        artifact_loc=Path(
            overrides.get("artifact_loc", env_data.get("artifact_loc", "/tmp/artifacts"))
        ),
    )


def _default_env() -> Env:
    global _DEFAULT_ENV
    if _DEFAULT_ENV is None:
        _DEFAULT_ENV = load_env()
    return _DEFAULT_ENV


def __getattr__(name: str) -> Any:
    """Temporary PEP 562 shim for legacy module-level env access."""
    env = _default_env()
    if name == "TRACKING_URI":
        return env.tracking_uri
    if name == "RAW_LOC":
        return env.raw_loc
    if name == "STAGED_LOC":
        return env.staged_loc
    if name == "ARTIFACT_LOC":
        return env.artifact_loc
    if name == "EXPERIMENT":
        # Still required by the temporary train/eval wiring.
        try:
            data = _load_toml()
            experiment = data.get("experiment")
            if experiment:
                return experiment
        except Exception:
            pass
        return "General-2"
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Configuration and path resolution.

Every path in this project is expressed relative to the repository root so the
code runs unchanged after cloning onto another machine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# src/utils/config.py -> src/utils -> src -> <project root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_config(name: str = "paths.yaml") -> dict[str, Any]:
    """Load a YAML config from ``configs/`` by file name."""
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve(relative_path: str | Path) -> Path:
    """Turn a project-relative path into an absolute one."""
    return (PROJECT_ROOT / Path(relative_path)).resolve()


def dataset_path(key: str, group: str = "raw", config: dict | None = None) -> Path:
    """Resolve a path declared in ``configs/paths.yaml``, e.g. ``dataset_path('irf_train')``."""
    cfg = config or load_config("paths.yaml")
    try:
        rel = cfg[group][key]
    except KeyError as exc:
        raise KeyError(f"No path '{group}.{key}' in configs/paths.yaml") from exc
    return resolve(rel)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

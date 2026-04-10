"""Centralized path resolution for the dwipada project.

All data directories and config paths derive from PROJECT_ROOT,
eliminating scattered Path(__file__).parent hacks.
"""

from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (contains pyproject.toml)."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("Could not find project root (no pyproject.toml found)")


PROJECT_ROOT = _find_project_root()

# Config
CONFIG_FILE = PROJECT_ROOT / "config.yaml"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = PROJECT_ROOT / "datasets"
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "synthetic_data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"

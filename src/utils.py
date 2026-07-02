"""Shared utility functions for the machine learning course design project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def get_data_path(filename: str) -> Path:
    """Return the path to a file in the data directory."""
    return DATA_DIR / filename

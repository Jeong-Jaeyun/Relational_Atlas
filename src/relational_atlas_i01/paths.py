"""Stable project paths, independent of the current working directory."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
NOTES_DIR = PROJECT_ROOT / "notes"


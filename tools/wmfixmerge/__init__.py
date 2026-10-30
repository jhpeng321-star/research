"""WMFIX tools use this worktree's implementation when invoked with python -m."""
from pathlib import Path
import sys

SRC = str(Path(__file__).resolve().parents[2] / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

"""Pytest fixtures for the WM-43B fixture-package tests (no SUMO needed).

v2: the r6/r7 dispatcher patches are already applied in the worktree
(traffic_shepherd/wall_group.py carries downstream_position_m and the
wall-v2.1 closure law), so the v1 patched_wg temp-copy fixture is gone; the
live module is imported directly.
"""
import sys
from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = FIXTURE_DIR.parents[2]

for path in (str(REPO_ROOT), str(FIXTURE_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_fixture  # noqa: E402,F401 - requires the sys.path setup above

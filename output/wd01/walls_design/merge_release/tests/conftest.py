"""Pytest bootstrap for the wall-law v2.2 'merge-release' tests.

Import surface (in priority order):
- tests/            -> these test modules + v22_fakes
- tests/shim/       -> campaign_walls_v1 + traffic_shepherd shims, so the
                       PATCHED post copies shadow the pre-v2.2 modules
- <workspace root>  -> the frozen traffic_shepherd fallback (config,
                       mixed_assignment) - never edited
- E:/research-dense/output -> the LIVE read-only campaign_stage1_v3 /
                       campaign_pilot_v1 dependencies the patched
                       wall_driver imports (imports never write there;
                       bytecode caching is disabled)

No D: paths anywhere (WD_FORBID_D=1); zero SUMO; no __pycache__.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True

TESTS_DIR = Path(__file__).resolve().parent
MERGE_DIR = TESTS_DIR.parent                       # .../merge_release
OUTPUT_DIR = MERGE_DIR.parents[1]                  # .../output
REPO_ROOT = OUTPUT_DIR.parent                      # workspace root
LIVE_OUTPUT = Path("E:/research-dense/output")     # READ-ONLY dependency tree

for _root in (REPO_ROOT, str(TESTS_DIR / "shim"), TESTS_DIR):
    # insert each at position 0, lowest priority FIRST: the final order
    # is [tests, tests/shim, repo, ...]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
if str(LIVE_OUTPUT) not in sys.path:
    sys.path.append(str(LIVE_OUTPUT))

import pytest  # noqa: E402

from campaign_walls_v1 import common  # noqa: E402


@pytest.fixture(autouse=True)
def _no_holdout_guard_file(monkeypatch):
    """An EMPTY holdout exclusion list: these tests never read the live
    protocols/holdout_exclusions.json (the guard module is only needed
    by plan/launch code, but the hook keeps every import inert)."""
    monkeypatch.setattr(
        common, "HOLDOUT_EXCLUSIONS_DATA",
        {"schema": "holdout_exclusions/1", "excluded": [],
         "kept_disclosed": []})


@pytest.fixture()
def post_copies() -> Path:
    return MERGE_DIR / "copies" / "post"

"""campaign_walls_v1 pytest fixtures (helpers live in wall_fakes)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = TESTS_DIR.parent.parent      # <repo>/output
REPO_ROOT = OUTPUT_ROOT.parent             # <repo>
for _root in (REPO_ROOT, OUTPUT_ROOT, TESTS_DIR):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from wall_fakes import (  # noqa: E402
    make_source_plan,
    make_source_plan_chunk3,
)


@pytest.fixture()
def source_bundle(tmp_path):
    """Both synthetic SOURCE plans with their fake chunk bases roots."""
    roots = {"chunk2": tmp_path / "runs" / "chunk2",
             "chunk3": tmp_path / "runs" / "chunk3"}
    plans = {"chunk2": make_source_plan(8477, roots["chunk2"]),
             "chunk3": make_source_plan_chunk3(8547, roots["chunk3"])}
    return {"plans": plans, "bases_roots": roots, "tmp_path": tmp_path}

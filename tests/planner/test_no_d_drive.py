"""PL-13: NO hard-coded D: drive literals anywhere in the live planner surface.

The D: junctions are scheduled for removal, so any surviving literal would
break the suite. Every former literal now resolves through
:mod:`traffic_shepherd.planner.paths` (env ``RESEARCH_DENSE_ROOT`` /
``RD_WT_ROOT``, E: defaults). This test re-runs the same grep the migration
used, over the same scope, and fails on ANY drive-D path outside the explicit
allow-list. The runtime half of the proof is the ``PLANNER_FORBID_D=1`` guard
in ``tests/planner/conftest.py``: with the env var set, an actual filesystem
access to D: raises.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

SCAN_ROOTS = (
    REPO_ROOT / "traffic_shepherd" / "planner",
    REPO_ROOT / "tests" / "planner",
    REPO_ROOT / "output" / "planner_v1",
    REPO_ROOT / "output" / "pl02a_offline_v1",
)
#: generated reports (RESULTS.json, *.md reports, caches, logs) are history,
#: not live code -- the brief excludes them from the grep
SKIP_SUFFIXES = {".md", ".json", ".csv", ".log"}
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache", "cache"}

#: D: drive PATH literal (either slash, case-insensitive). Plain-prose
#: mentions of "D:" without a slash are not paths and do not match.
D_DRIVE_PATH_RE = re.compile(r"[Dd]:[/\\]")

#: Historical-record DATA only: frozen files whose bytes are pinned
#: elsewhere (sha256-ledgered evidence) and must stay verbatim. Each entry:
#: (repo-relative path, justification). Anything not listed here FAILS.
ALLOW_LIST = {
    "output/planner_v1/PYTEST_FULL.txt": (
        "frozen PL01 closeout evidence log: its pytest output was captured in "
        "the D:-era pl01 workspace and its sha256 is pinned in "
        "LEDGER_DRAFT.json; generated evidence is never rewritten"
    ),
}


def test_no_d_drive_path_literals():
    offenders = []
    for root in SCAN_ROOTS:
        if not root.is_dir():  # pragma: no cover - the four dirs always ship
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
                continue
            if SKIP_DIR_NAMES.intersection(path.parts):
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue  # binary artifact, not a path-bearing source file
            for lineno, line in enumerate(text.splitlines(), start=1):
                if D_DRIVE_PATH_RE.search(line) and rel not in ALLOW_LIST:
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "hard-coded D: drive paths found (resolve via "
        "traffic_shepherd.planner.paths; env RESEARCH_DENSE_ROOT/RD_WT_ROOT):\n"
        + "\n".join(offenders)
    )


def test_allow_list_entries_still_exist():
    """The allow-list names REAL frozen files: an entry whose file is gone
    (or renamed) must be dropped, not carried silently."""
    for rel in ALLOW_LIST:
        assert (REPO_ROOT / rel).is_file(), f"stale allow-list entry: {rel}"

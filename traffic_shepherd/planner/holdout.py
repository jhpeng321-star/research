"""PL-12: HOLDOUT exclusion-guard adapter for the planner.

The single source of truth for the 13 exposed chunk-1 HOLDOUT worlds is the
reconstructor-owned ``tools/holdout_guard.py`` plus its data file
``protocols/holdout_exclusions.json``.  This adapter IMPORTS the guard BY
PATH from ``<RESEARCH_DENSE_ROOT>/tools/holdout_guard.py``
(``traffic_shepherd.planner.paths.research_dense_root()``; the env override
decides, so tests can point at a tmp fixture).  The exclusion list is NEVER
copied, re-listed or hard-coded here or in any caller.

Contract (owner ruling, PL-12):

- ``assert_holdout_eligible(..., holdout_requested=True)`` FAILS CLOSED: a
  missing or unparseable guard raises the named
  :class:`HoldoutGuardUnavailableError`, and any excluded world (per world,
  including every re-run arm of it) raises the guard's own
  ``HoldoutExclusionError`` (a ``ValueError`` subclass).
- With ``holdout_requested=False`` the call never fails on a missing guard
  (HOLDOUT was not requested); it stays a best-effort defence-in-depth
  refusal when the guard does load.

Who calls it, and refuse-vs-filter semantics:

- ``output/pl02a_offline_v1/report_chunk1_final.build_report`` and
  ``report_chunk_final.build_report``: REFUSE.  With HOLDOUT in
  ``--splits`` (even with ``--owner-allow-holdout``) the excluded worlds are
  refused by name BEFORE any outcome file is opened; the whole report is
  aborted.  ``kept_disclosed`` worlds are LISTED in the report head and in
  ``RESULTS.json['guard']`` whenever HOLDOUT is included.
- ``output/pl02a_offline_v1/run.enforce_train_only``: REFUSE.  The
  ``--allow-non-train`` owner override for scoped smoke runs still refuses
  the excluded HOLDOUT worlds.
- ``traffic_shepherd.planner.closed_loop`` (``--owner-allow-holdout``):
  REFUSE at config construction, before any run directory is created.
- ``output/planner_v1/plexp1_analysis/analyze_plexp1``: TRAIN-only path
  (non-TRAIN worlds are already refused); the guard is called defensively
  with ``holdout_requested=False`` so a sneaky excluded TRAIN-labelled
  world is still refused, while a missing guard cannot fail a TRAIN-only
  analysis.
- ``traffic_shepherd.planner.recorded`` has no split filter for HOLDOUT
  (the split scoping happens in the report modules above); nothing to wire.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Sequence

from traffic_shepherd.planner.paths import research_dense_root

__all__ = [
    "HoldoutGuardUnavailableError",
    "assert_holdout_eligible",
    "filter_holdout",
    "guard_path",
    "holdout_disclosures",
    "load_guard",
]

#: the reconstructor-owned guard, relative to the research-dense root
GUARD_SUBPATH = Path("tools") / "holdout_guard.py"

_MODULE_NAME = "traffic_shepherd.planner._holdout_guard_module"


class HoldoutGuardUnavailableError(RuntimeError):
    """HOLDOUT was requested but the guard is missing or unparseable.

    Fail closed: fix ``RESEARCH_DENSE_ROOT`` or restore
    ``<root>/tools/holdout_guard.py`` (owned by the reconstructor).
    """


def guard_path() -> Path:
    """Where the single-source-of-truth guard must live."""
    return research_dense_root() / GUARD_SUBPATH


def load_guard():
    """Import ``tools/holdout_guard.py`` BY PATH (never copied).

    Re-imported fresh on every call so a changed ``RESEARCH_DENSE_ROOT``
    (tests: ``monkeypatch.setenv``) always re-resolves both the module and
    its module-level ``EXCLUSIONS_PATH``.  Raises the named
    :class:`HoldoutGuardUnavailableError` when the file is missing or does
    not parse.
    """
    path = guard_path()
    if not path.is_file():
        raise HoldoutGuardUnavailableError(
            f"holdout guard missing: {path} does not exist "
            f"(expected <RESEARCH_DENSE_ROOT>/{GUARD_SUBPATH.as_posix()}); "
            "HOLDOUT evaluations fail closed without it"
        )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise HoldoutGuardUnavailableError(f"holdout guard not importable: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(_MODULE_NAME, None)
        raise HoldoutGuardUnavailableError(
            f"holdout guard failed to load from {path}: {exc}"
        ) from exc
    return module


def assert_holdout_eligible(
    worlds: Sequence[str], *, holdout_requested: bool, data=None
):
    """Refuse excluded roots (per world, run names accepted) when requested.

    ``holdout_requested=True``: fail closed (missing/unparseable guard ->
    :class:`HoldoutGuardUnavailableError`; excluded world -> the guard's own
    ``HoldoutExclusionError``).  ``holdout_requested=False``: best-effort
    only -- an excluded world is still refused when the guard loads, but a
    missing guard does NOT fail a non-HOLDOUT path.  Returns the loaded
    guard module (or ``None`` when skipped).
    """
    names = [str(w) for w in worlds]
    if not holdout_requested:
        try:
            guard = load_guard()
        except HoldoutGuardUnavailableError:
            return None
        guard.assert_final_holdout_eligible(names, data)
        return guard
    guard = load_guard()
    guard.assert_final_holdout_eligible(names, data)
    return guard


def filter_holdout(worlds: Sequence[str], *, holdout_requested: bool, data=None) -> list:
    """Guard-filtered world list (refuse-vs-filter entry points that FILTER
    may use this); fail closed exactly like :func:`assert_holdout_eligible`
    when ``holdout_requested`` is true, pass-through when false."""
    names = [str(w) for w in worlds]
    if not holdout_requested:
        try:
            guard = load_guard()
        except HoldoutGuardUnavailableError:
            return names
        return list(guard.filter_final_holdout(names, data))
    return list(load_guard().filter_final_holdout(names, data))


def holdout_disclosures(data=None) -> list:
    """``kept_disclosed`` roots: KEPT in HOLDOUT but LISTED in every final
    HOLDOUT report (owner ruling WM-70)."""
    return list(load_guard().disclosed_roots(data))

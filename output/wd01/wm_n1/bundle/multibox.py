"""P10 MULTIBOX: one config per GPU box, boxes in parallel, member-1 first.

The WM-N1 GPU queue from :mod:`bundle.queue` is sequential.  The held-out
strategy tests (amendment 3 A3.4) and the 5 ensemble members are independent
jobs, so the owner splits them across boxes: **each box runs exactly one
config**, boxes run in parallel, and every config is guarded by its own owner
``APPROVAL.json``.

Configs
-------

* ``member-1`` .. ``member-5`` -- the 5-member Idea A ensemble, member 1 FIRST
  (amendment 1 A1).  ``member-k`` (k>1) refuses to start until ``member-1``
  has completed, wherever that ran.
* ``n-member-1`` .. ``n-member-5`` -- the variant **N** comparator, the
  original label-conditioned pure-neural design (amendment 3 A3.3).  These are
  the v1 ``model/`` + ``train/`` stack trained with the v1 CLI
  ``python -m train --member k`` on the same packed corpus; they obey the SAME
  member-1-first rule inside their own family.  N cannot handle unseen
  strategies (that is the point of the comparator).
* ``heldout-i-j4b`` -- (i) train on J4 (wall law v2.1), evaluate on J4B
  (law v2.2); the J4B outcomes are read ONLY by this test.
* ``heldout-ii-leave-pi`` -- (ii) train with every PI arm removed, evaluate
  the PI arms of CAL.
* ``heldout-iii-leave-highest-wall-speed`` -- (iii) train with the wall-speed
  arms of the a-priori highest J4 speed factor removed, evaluate those arms.

Each held-out config is a 1-member model.  ``member-1-first`` is per model
family: ``member-k``/``n-member-k`` (k>1) each wait for their own family's
member 1.

The owner decision (2026-10-01) is that every rented GPU (3-4 boxes) runs these
configs: Idea A members 1-5 (member-1 first), the N comparator members, and the
three held-out tests.  :func:`plan_fleet` schedules them onto 3 or 4 physical
boxes; :mod:`bundle` ``GPU_PLAN.md`` records both tables.  ``member-1`` must be
the first job on box 1.

Approval binding (per config)
-----------------------------

``APPROVAL.json`` (schema ``wm_n1_multibox_approval/1``) must bind the
config's written ``TIMING_GATE.json`` sha256 AND all eight frozen records,
LF-normalised (CRLF-safe):

* ``registration_lf_sha256``  = ``666233aa...93246`` (REGISTRATION_WM_N1.md)
* ``amendment_1_lf_sha256``   = ``decb9656...16c57`` (member-1-first)
* ``amendment_2_lf_sha256``   = ``56fa99a0...c83ed`` (J4 disclosure)
* ``amendment_3_lf_sha256``   = ``a597d986...04eef6`` (Idea A)
* ``amendment_4_lf_sha256``   = ``d392b300...0822f`` (CAL-only selection)
* ``amendment_5_lf_sha256``   = ``7788c270...d13ae7`` (K_max format, v3=192)
* ``amendment_6_lf_sha256``   = ``28dbc184...c948``  (state global token, A6)
* ``cal_split_lf_sha256``     = ``bc4e32f2...20d46`` (CAL_SPLIT_A4.json)

The declared fields must equal the frozen constants AND, when the records are
present, the re-hashed files.  The timing gate is hashed on its raw bytes (JSON
is not line-ending normalised), exactly like :mod:`bundle.approval`.

Refusals
--------

* :class:`UnassignedConfigRefusal` -- a box was asked for a config that is not
  assigned to it (the default assignment is one config per box).
* :class:`BoxOrderError` -- a member-k (k>1) config was started before its
  own family's member 1 completed.
* :class:`ConfigApprovalMissing` -- the per-config owner approval is absent or
  does not bind the gate + frozen records.

State is per box (``<work_dir>/boxes/<box>/state.json``) plus a fleet ledger
(``<work_dir>/fleet_state.json``) so a re-run resumes and never silently
repeats a completed config.

CLI (run from ``output/wd01/wm_n1``)::

    python -m bundle.multibox plan
    python -m bundle.multibox plan-fleet --boxes 3      # GPU_PLAN table
    python -m bundle.multibox run-box  --box box-member-1 --work-dir WORK
    python -m bundle.multibox run-config --box box-member-1 --config member-1 \
        --work-dir WORK --approval A.json --timing-gate TIMING_GATE.json
    python -m bundle.multibox run-fleet --work-dir WORK --parallel
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from evalh import cal_split as _cal_split
from evalh.seals import LF_SHA_AMENDMENT_6, lf_sha256
from model import constants as C

#: The two A4.2 CAL halves (single source: `evalh.cal_split`).  Every
#: `evalh.run_eval --split cal` argv a box config emits names exactly one of
#: these, so the member-1/quick-tune SELECTION path reads CAL-tune only and the
#: calibration/blend/variant path reads CAL-cal only.
CAL_TUNE = _cal_split.CAL_TUNE
CAL_CAL = _cal_split.CAL_CAL

# ------------------------------------------------------------------ frozen ---
WM_N1_DIR = Path(__file__).resolve().parents[1]

LF_SHA_REGISTRATION = ("666233aa1d1e466750d6d944b9102c84a2f32850ebeb4a97c9f3c9b5"
                       "6dd93246")
LF_SHA_AMENDMENT_1 = ("decb96562ebfb549b56406d980be9c78bf4a377bd50c5eadeea"
                      "08aac16e16c57")
LF_SHA_AMENDMENT_2 = ("56fa99a04fdd38e37a518dfc65972bc5e2e6d227210372a75258b"
                      "f28249c83ed")
LF_SHA_AMENDMENT_3 = ("a597d9861dcc859cbcb9c10b65069a81031732516ec12a0296fd6"
                      "8aff604eef6")
# Amendment 4 (CAL-only selection; A4.2) + the materialised CAL split.  Every
# per-config APPROVAL.json must bind these in addition to the four records
# above, so a config cannot start against an unbound improvement-loop split.
LF_SHA_AMENDMENT_4 = ("d392b30030f2b04ddb5db25c681cf92bca80fe1fc4e77d2343"
                      "03f7bbb810822f")
LF_SHA_CAL_SPLIT_A4 = ("bc4e32f2b6efcf0e0165cb175a0b3ffaec0aaa6f419076463"
                       "f9242ad90820d46")
# Amendment 5 (K_max is a FORMAT parameter; K=192 v3) LF-sha.  Every per-config
# APPROVAL.json binds this too, so a config cannot start against an unbound
# K_max rule (the M1-only configs run on pack_v3).
LF_SHA_AMENDMENT_5 = ("7788c270f35a0620f0bfd71a61d871f741b6bbbffa144ba61438"
                      "e28b73d13ae7")
# Amendment 6 (state-only global token for the correction head) LF-sha is
# imported from evalh.seals (``LF_SHA_AMENDMENT_6``); every per-config
# APPROVAL.json binds it too, so a config cannot train with
# ``state_global_token`` ON against an unfrozen/tampered A6 record.

REGISTRATION_PATH = WM_N1_DIR / "REGISTRATION_WM_N1.md"
AMENDMENT_1_PATH = WM_N1_DIR / "REGISTRATION_WM_N1_AMENDMENT_1.md"
AMENDMENT_2_PATH = WM_N1_DIR / "REGISTRATION_WM_N1_AMENDMENT_2_J4_DISCLOSURE.md"
AMENDMENT_3_PATH = WM_N1_DIR / "REGISTRATION_WM_N1_AMENDMENT_3_IDEA_A.md"
AMENDMENT_4_PATH = WM_N1_DIR / "REGISTRATION_WM_N1_AMENDMENT_4_M1_TUNING.md"
AMENDMENT_5_PATH = WM_N1_DIR / "REGISTRATION_WM_N1_AMENDMENT_5_KMAX_RULE.md"
AMENDMENT_6_PATH = WM_N1_DIR / "REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md"
CAL_SPLIT_PATH = WM_N1_DIR / "CAL_SPLIT_A4.json"

# frozen-record key -> (path, frozen LF-sha256)
DEFAULT_RECORDS: Mapping[str, tuple] = {
    "registration": (REGISTRATION_PATH, LF_SHA_REGISTRATION),
    "amendment_1": (AMENDMENT_1_PATH, LF_SHA_AMENDMENT_1),
    "amendment_2": (AMENDMENT_2_PATH, LF_SHA_AMENDMENT_2),
    "amendment_3": (AMENDMENT_3_PATH, LF_SHA_AMENDMENT_3),
    "amendment_4": (AMENDMENT_4_PATH, LF_SHA_AMENDMENT_4),
    "amendment_5": (AMENDMENT_5_PATH, LF_SHA_AMENDMENT_5),
    "amendment_6": (AMENDMENT_6_PATH, LF_SHA_AMENDMENT_6),
    "cal_split": (CAL_SPLIT_PATH, LF_SHA_CAL_SPLIT_A4),
}

APPROVAL_SCHEMA = "wm_n1_multibox_approval/1"
APPROVAL_FILENAME = "APPROVAL.json"
TIMING_GATE_FILENAME = "TIMING_GATE.json"
STATE_FILENAME = "state.json"
FLEET_STATE_FILENAME = "fleet_state.json"
BOXES_DIRNAME = "boxes"
APPROVALS_DIRNAME = "approvals"

#: The agreed (parallel-piece) executor entry point, resolved when no executor
#: callable is injected.  ``bundle.e2e.run_config_job(box_id, config, work_dir)``.
DEFAULT_EXECUTOR_ENTRY = "bundle.e2e.run_config_job"

# Serialises per-box/fleet state read-modify-write across parallel boxes in
# one process; temp files are unique so concurrent writers never collide.
_STATE_LOCK = threading.RLock()


class MultiboxError(RuntimeError):
    """A multibox configuration or run failed."""


class UnassignedConfigRefusal(PermissionError):
    """A box was asked to run a config not assigned to it."""


class BoxOrderError(MultiboxError):
    """A member-k (k>1) config started before member 1 completed."""


class ConfigApprovalMissing(PermissionError):
    """The per-config owner approval is absent or does not bind the records."""


# ------------------------------------------------------------------ configs ---
IDEA_A_FAMILY = "idea_a"
N_FAMILY = "n"
FAMILY_ORDER: tuple = (IDEA_A_FAMILY, N_FAMILY)

#: Each family's member 1, which must complete before that family's other
#: members may start (amendment 1 A1; the N comparator follows the same rule).
FAMILY_MEMBER_1: Mapping[str, str] = {
    IDEA_A_FAMILY: "member-1",
    N_FAMILY: "n-member-1",
}
#: Backwards-compatible alias for the Idea A family head.
MEMBER_1 = FAMILY_MEMBER_1[IDEA_A_FAMILY]

_V1_TRAIN_CLI = "python -m train --member {member}"


@dataclass(frozen=True)
class ConfigSpec:
    """One queue config (a family member or a held-out-strategy test)."""

    config_id: str
    kind: str                       # "member" | "heldout"
    family: str = IDEA_A_FAMILY     # "idea_a" (headline) | "n" (comparator)
    member: int = 1                 # 1..5; held-out tests are 1-member models
    description: str = ""
    train: Optional[str] = None
    evaluate: Optional[str] = None
    excluded_arm: Optional[str] = None
    train_cli: Optional[str] = None  # exact training entry point for the box


CONFIG_SPECS: Mapping[str, ConfigSpec] = {
    "member-1": ConfigSpec(
        "member-1", "member", IDEA_A_FAMILY, 1,
        "Idea A ensemble member 1 (trained FIRST, amendment 1 A1)"),
    "member-2": ConfigSpec(
        "member-2", "member", IDEA_A_FAMILY, 2, "Idea A ensemble member 2"),
    "member-3": ConfigSpec(
        "member-3", "member", IDEA_A_FAMILY, 3, "Idea A ensemble member 3"),
    "member-4": ConfigSpec(
        "member-4", "member", IDEA_A_FAMILY, 4, "Idea A ensemble member 4"),
    "member-5": ConfigSpec(
        "member-5", "member", IDEA_A_FAMILY, 5, "Idea A ensemble member 5"),
    "n-member-1": ConfigSpec(
        "n-member-1", "member", N_FAMILY, 1,
        "Variant N comparator member 1 (v1 pure-neural; trained FIRST)",
        train="TRAIN packed corpus", train_cli=_V1_TRAIN_CLI.format(member=1)),
    "n-member-2": ConfigSpec(
        "n-member-2", "member", N_FAMILY, 2,
        "Variant N comparator member 2 (v1 pure-neural)",
        train="TRAIN packed corpus", train_cli=_V1_TRAIN_CLI.format(member=2)),
    "n-member-3": ConfigSpec(
        "n-member-3", "member", N_FAMILY, 3,
        "Variant N comparator member 3 (v1 pure-neural)",
        train="TRAIN packed corpus", train_cli=_V1_TRAIN_CLI.format(member=3)),
    "n-member-4": ConfigSpec(
        "n-member-4", "member", N_FAMILY, 4,
        "Variant N comparator member 4 (v1 pure-neural)",
        train="TRAIN packed corpus", train_cli=_V1_TRAIN_CLI.format(member=4)),
    "n-member-5": ConfigSpec(
        "n-member-5", "member", N_FAMILY, 5,
        "Variant N comparator member 5 (v1 pure-neural)",
        train="TRAIN packed corpus", train_cli=_V1_TRAIN_CLI.format(member=5)),
    "heldout-i-j4b": ConfigSpec(
        "heldout-i-j4b", "heldout", IDEA_A_FAMILY, 1,
        "(i) HEADLINE: train J4 (law v2.1) -> evaluate J4B (law v2.2)",
        train="J4", evaluate="J4B", excluded_arm="J4B"),
    "heldout-ii-leave-pi": ConfigSpec(
        "heldout-ii-leave-pi", "heldout", IDEA_A_FAMILY, 1,
        "(ii) train with every PI arm removed -> evaluate the PI arms of CAL",
        train="all-but-PI", evaluate="CAL-PI", excluded_arm="PI"),
    "heldout-iii-leave-highest-wall-speed": ConfigSpec(
        "heldout-iii-leave-highest-wall-speed", "heldout", IDEA_A_FAMILY, 1,
        "(iii) train without the highest J4 wall-speed factor -> evaluate it",
        train="all-but-highest-wall-speed",
        evaluate="CAL-highest-wall-speed",
        excluded_arm="highest J4 wall-speed factor"),
}

MEMBER_CONFIGS: tuple = tuple(c for c, s in CONFIG_SPECS.items()
                              if s.kind == "member"
                              and s.family == IDEA_A_FAMILY)
N_MEMBER_CONFIGS: tuple = tuple(c for c, s in CONFIG_SPECS.items()
                                if s.kind == "member" and s.family == N_FAMILY)
HELDOUT_CONFIGS: tuple = tuple(c for c, s in CONFIG_SPECS.items()
                               if s.kind == "heldout")
ALL_MEMBER_CONFIGS: tuple = MEMBER_CONFIGS + N_MEMBER_CONFIGS
ALL_CONFIGS: tuple = ALL_MEMBER_CONFIGS + HELDOUT_CONFIGS

#: Default fleet: exactly ONE config per box.  Boxes run in parallel except
#: that each family's member-k (k>1) box waits for that family's member-1.
DEFAULT_BOX_ASSIGNMENT: Mapping[str, tuple] = {
    **{f"box-{c}": (c,) for c in ALL_CONFIGS},
}


def assignment_receipt(assignment: Optional[Mapping] = None) -> dict:
    """JSON-able, ordered view of the fleet assignment."""
    assignment = assignment or DEFAULT_BOX_ASSIGNMENT
    return {
        "one_config_per_box": all(len(v) == 1 for v in assignment.values()),
        "n_boxes": len(assignment),
        "n_configs": sum(len(v) for v in assignment.values()),
        "boxes": {b: list(assignment[b]) for b in _ordered_boxes(assignment)},
    }


def configs_for_box(box_id: str,
                    assignment: Optional[Mapping] = None) -> tuple:
    assignment = assignment or DEFAULT_BOX_ASSIGNMENT
    if box_id not in assignment:
        raise MultiboxError(
            f"unknown box {box_id!r}; boxes are {list(assignment)}")
    return tuple(assignment[box_id])


def box_for_config(config: str,
                   assignment: Optional[Mapping] = None) -> str:
    assignment = assignment or DEFAULT_BOX_ASSIGNMENT
    for box_id, configs in assignment.items():
        if config in configs:
            return box_id
    raise MultiboxError(f"config {config!r} is not assigned to any box")


def member_index(config: str) -> Optional[int]:
    spec = CONFIG_SPECS.get(config)
    return None if spec is None or spec.kind != "member" else int(spec.member)


def _family_rank(family: str) -> int:
    return FAMILY_ORDER.index(family) if family in FAMILY_ORDER \
        else len(FAMILY_ORDER)


def planned_order(configs: Sequence[str]) -> tuple:
    """Within-box order: each family's member-1 first, then members, held-out.

    Member-1-first is enforced *inside each family* (Idea A, then variant N).
    """
    def key(config: str):
        spec = CONFIG_SPECS.get(config)
        if spec is not None and spec.kind == "member":
            return (_family_rank(spec.family), int(spec.member), 0, str(config))
        return (len(FAMILY_ORDER), 0, 1, str(config))

    return tuple(sorted(configs, key=key))


def _ordered_boxes(assignment: Mapping) -> list:
    """Box list: Idea A member-1's box first, then N member-1's box, then rest."""
    boxes = list(assignment)

    def rank(box_id: str) -> int:
        for i, family in enumerate(FAMILY_ORDER):
            if FAMILY_MEMBER_1[family] in assignment[box_id]:
                return i
        return len(FAMILY_ORDER)

    ordered = sorted(enumerate(boxes), key=lambda pair: (rank(pair[1]), pair[0]))
    return [box for _, box in ordered]


# --------------------------------------------------------------- planner ----
#: Design-phase GPU-hour envelopes per config, from amendment 3 A3.7.
IDEA_A_MEMBER_GPU_HOURS: tuple = (9.0, 17.0)   # per Idea A member
N_MEMBER_GPU_HOURS: tuple = (5.0, 10.0)        # per variant N comparator member
HELDOUT_GPU_HOURS: tuple = IDEA_A_MEMBER_GPU_HOURS  # 1-member model, as Idea A


def _hours_range(config: str) -> tuple:
    spec = CONFIG_SPECS[config]
    if spec.family == N_FAMILY:
        return N_MEMBER_GPU_HOURS
    if spec.kind == "heldout":
        return HELDOUT_GPU_HOURS
    return IDEA_A_MEMBER_GPU_HOURS


def design_gpu_hours(config: str) -> float:
    """Midpoint design estimate (GPU-h) for one config (amendment 3 A3.7)."""
    if config not in CONFIG_SPECS:
        raise MultiboxError(f"unknown config {config!r}")
    lo, hi = _hours_range(config)
    return (float(lo) + float(hi)) / 2.0


def hours_range(config: str) -> tuple:
    """Public per-config design GPU-hour envelope ``(low, high)`` (A3.7)."""
    if config not in CONFIG_SPECS:
        raise MultiboxError(f"unknown config {config!r}")
    return _hours_range(config)


# ------------------------------------------------------------- co-run model ---
#: DOCUMENTED design assumption for variant (c) (owner 2026-10-01): running
#: jobs concurrently on a >=48 GB card raises the box's AGGREGATE throughput
#: by 1.3x for 2 simultaneous jobs and 1.5x for 3.  These are placeholders to
#: be replaced by the 1-hour timing gate before any wall-clock claim is made.
CO_RUN_SPEEDUP: Mapping[int, float] = {1: 1.0, 2: 1.3, 3: 1.5}


def co_run_speedup(n_jobs: int) -> float:
    """Aggregate-throughput multiplier for ``n_jobs`` co-running on one box."""
    if n_jobs <= 1:
        return 1.0
    if n_jobs == 2:
        return CO_RUN_SPEEDUP[2]
    return CO_RUN_SPEEDUP[3]


def simulate_box_schedule(ordered: Sequence[str], *, concurrency: int = 1,
                          hours_fn: Optional[Callable[[str], float]] = None,
                          head_finish: Optional[Mapping[str, float]] = None) -> dict:
    """Batch-schedule one box's ordered queue, honouring member-1-first.

    Each batch is the next up-to-``concurrency`` jobs in ``ordered`` that are
    *released*; a member-k (k>1) is released only once its own family's
    member-1 has finished (``head_finish`` maps family -> finish hours).  A
    batch of ``n`` jobs takes ``sum(hours)/co_run_speedup(n)`` wall hours and
    every job in it reports the batch-completion time (the aggregate-throughput
    co-run assumption).  ``concurrency=1`` reduces to a sequential schedule.
    """
    size = hours_fn or design_gpu_hours
    head_finish = dict(head_finish or {})
    concurrency = max(1, min(int(concurrency), 3))
    remaining = list(ordered)
    unknown = [c for c in remaining if c not in CONFIG_SPECS]
    if unknown:
        raise MultiboxError(f"unknown config(s) in box schedule: {unknown}")
    if len(set(remaining)) != len(remaining):
        raise MultiboxError("box schedule contains a duplicate config")
    start: dict = {}
    finish: dict = {}
    t = 0.0

    def released(config: str) -> bool:
        spec = CONFIG_SPECS[config]
        if spec.kind == "member" and spec.member > 1:
            head = FAMILY_MEMBER_1[spec.family]
            if head in finish:                 # head ran (and finished) here
                return finish[head] <= t + 1e-9
            if head in remaining:              # head still queued on this box
                return False
            return head_finish.get(head, 0.0) <= t + 1e-9
        return True

    while remaining:
        ready = [c for c in remaining if released(c)][:concurrency]
        if not ready:
            waits = []
            for config in remaining:
                spec = CONFIG_SPECS[config]
                if spec.kind == "member" and spec.member > 1:
                    head = FAMILY_MEMBER_1[spec.family]
                    if head in finish:
                        waits.append(max(0.0, finish[head] - t))
                    elif head in head_finish:
                        waits.append(max(0.0, head_finish[head] - t))
            step = min(waits) if waits else 0.0
            if step <= 1e-12:
                raise MultiboxError(
                    f"deadlock scheduling {remaining}: family heads unknown")
            t += step
            continue
        total = float(sum(float(size(c)) for c in ready))
        end = t + total / co_run_speedup(len(ready))
        for c in ready:
            start[c] = t
            finish[c] = end
            remaining.remove(c)
        t = end

    return {
        "start_hours": start,
        "finish_hours": finish,
        "box_finish_hours": max(finish.values()) if finish else 0.0,
        "concurrency": concurrency,
    }


def plan_fleet(n_boxes: int,
               configs: Optional[Sequence[str]] = None,
               hours_fn: Optional[Callable[[str], float]] = None,
               concurrency: int = 1) -> dict:
    """Deterministic list schedule of configs onto ``n_boxes`` physical boxes.

    Honours both family gates: Idea A ``member-1`` is box 1's first job at
    t=0 and the N comparator's ``n-member-1`` is box 2's first job at t=0;
    every other member k>1 is released only when its own family's member-1
    finishes (amendment 1 A1).  Held-out 1-member tests are independent.
    The greedy (earliest-finish) pass minimises the estimated wall-clock.
    ``hours_fn`` overrides the per-config hours (default:
    :func:`design_gpu_hours`); pass a low/high sizer to bound the estimate.
    """
    if int(n_boxes) < 1:
        raise MultiboxError("n_boxes must be >= 1")
    n_boxes = int(n_boxes)
    configs = list(configs) if configs is not None else list(ALL_CONFIGS)
    unknown = [c for c in configs if c not in CONFIG_SPECS]
    if unknown:
        raise MultiboxError(f"unknown config(s) in plan: {unknown}")
    size = hours_fn or design_gpu_hours

    boxes = [f"box-{i + 1}" for i in range(n_boxes)]
    free = {b: 0.0 for b in boxes}
    order: dict = {b: [] for b in boxes}
    start: dict = {}

    def place(config: str, release: float, box_id: str) -> None:
        t = max(free[box_id], release)
        start[config] = t
        order[box_id].append(config)
        free[box_id] = t + size(config)

    # 1. Family heads first: Idea A member-1 on box 1, then n-member-1.
    heads = [FAMILY_MEMBER_1[f] for f in FAMILY_ORDER
             if FAMILY_MEMBER_1[f] in configs]
    head_finish: dict = {}
    for i, head in enumerate(heads):
        if i < n_boxes:
            box_id = boxes[i]
        else:
            box_id = min(boxes, key=lambda b: (free[b], b))
        place(head, 0.0, box_id)
        head_finish[CONFIG_SPECS[head].family] = free[box_id]

    # 2. Everything else: list-schedule by (release, -hours, config_id).
    def release_of(config: str) -> float:
        spec = CONFIG_SPECS[config]
        if spec.kind == "member":
            return head_finish.get(spec.family, 0.0)
        return 0.0

    rest = [c for c in configs if c not in heads]
    rest.sort(key=lambda c: (release_of(c), -size(c), c))
    for config in rest:
        release = release_of(config)
        box_id = min(boxes, key=lambda b: (max(free[b], release)
                                           + size(config), b))
        place(config, release, box_id)

    if int(concurrency) > 1:
        return _replan_fleet_concurrent(boxes, order, configs, size,
                                        int(concurrency))

    makespan = max(free.values()) if free else 0.0
    return {
        "schema": "wm_n1_gpu_plan/1",
        "n_boxes": n_boxes,
        "boxes": {b: list(order[b]) for b in boxes},
        "config_start_hours": {c: start[c] for c in configs},
        "config_hours": {c: size(c) for c in configs},
        "box_finish_hours": dict(free),
        "wall_clock_hours": makespan,
        "member_1_first_on_box_1": bool(order[boxes[0]]
                                        and order[boxes[0]][0] == MEMBER_1),
    }


def _head_box_rank(box_id: str, order: Mapping, boxes: Sequence[str]) -> int:
    configs = order[box_id]
    for i, family in enumerate(FAMILY_ORDER):
        if FAMILY_MEMBER_1[family] in configs:
            return i
    return len(FAMILY_ORDER)


def _replan_fleet_concurrent(boxes: Sequence[str], order: Mapping,
                             configs: Sequence[str],
                             size: Callable[[str], float],
                             concurrency: int) -> dict:
    """Re-time a greedy fleet assignment with co-run batches per box."""
    concurrency = max(1, min(int(concurrency), 3))
    head_finish: dict = {}
    start: dict = {}
    finish: dict = {}
    box_finish: dict = {}
    ordered_boxes = sorted(
        boxes, key=lambda b: (_head_box_rank(b, order, boxes), list(boxes).index(b)))
    for box_id in ordered_boxes:
        res = simulate_box_schedule(order[box_id], concurrency=concurrency,
                                    hours_fn=size, head_finish=head_finish)
        start.update(res["start_hours"])
        finish.update(res["finish_hours"])
        box_finish[box_id] = res["box_finish_hours"]
        for config in order[box_id]:
            spec = CONFIG_SPECS[config]
            if spec.kind == "member" and spec.member == 1:
                head_finish[spec.family] = res["finish_hours"][config]
    makespan = max(box_finish.values()) if box_finish else 0.0
    return {
        "schema": "wm_n1_gpu_plan/1",
        "n_boxes": len(boxes),
        "concurrency": concurrency,
        "co_run_speedup": dict(CO_RUN_SPEEDUP),
        "boxes": {b: list(order[b]) for b in boxes},
        "config_start_hours": {c: start[c] for c in configs},
        "config_hours": {c: size(c) for c in configs},
        "box_finish_hours": box_finish,
        "wall_clock_hours": makespan,
        "member_1_first_on_box_1": bool(order[boxes[0]]
                                        and order[boxes[0]][0] == MEMBER_1),
    }


def format_plan(plan: Mapping) -> str:
    """Human-readable GPU_PLAN table for one :func:`plan_fleet` result."""
    lines = [f"n_boxes={plan['n_boxes']}  "
             f"wall_clock_hours={plan['wall_clock_hours']:.1f}"]
    for box_id, configs in plan["boxes"].items():
        lines.append(f"{box_id}  (finish {plan['box_finish_hours'][box_id]:.1f} h)")
        for config in configs:
            lines.append(
                f"    {config}  start {plan['config_start_hours'][config]:.1f} h"
                f"  ({plan['config_hours'][config]:.1f} GPU-h)")
    return "\n".join(lines)


# ------------------------------------------------------- two-box owner plans ---
BOX_CONFIG_SCHEMA = "wm_n1_box_config/1"
BOX_CONFIGS_INDEX_SCHEMA = "wm_n1_box_configs_index/1"
TWO_BOX_PLAN_SCHEMA = "wm_n1_gpu_plan_2box/1"

BOX_A = "box-A"     # GPU-A MAIN
BOX_B = "box-B"     # GPU-B TESTS

#: The owner's three two-box variants (2026-10-01).  The N comparator's
#: ``n-member-1`` is a family head in every variant, so member-1-first holds
#: inside each family.
TWO_BOX_VARIANTS: Mapping[str, Mapping] = {
    "a": {
        "title": "2-box (a): all 13 configs (A = GPU-A MAIN, B = GPU-B TESTS)",
        "description": (
            "All 13 configs.  Box A runs the 5 Idea A members and the 5 variant "
            "N members (Idea A member-1 first, then N member-1 before its own "
            "members); box B runs the three held-out 1-member tests."),
        "co_run": False,
        "boxes": {
            BOX_A: ("member-1", "member-2", "member-3", "member-4", "member-5",
                    "n-member-1", "n-member-2", "n-member-3", "n-member-4",
                    "n-member-5"),
            BOX_B: ("heldout-i-j4b", "heldout-ii-leave-pi",
                    "heldout-iii-leave-highest-wall-speed"),
        },
    },
    "b": {
        "title": "2-box (b) LEAN: Idea A on A; held-out + N member-1 on B",
        "description": (
            "Lean 9-config plan.  Box A = Idea A members 1-5 (member-1 first); "
            "box B = held-out (i) FIRST (headline), then (ii), (iii), and the "
            "single-member N comparator point n-member-1 (disclosed as a "
            "single-member point comparator, not the full N ensemble)."),
        "co_run": False,
        "boxes": {
            BOX_A: ("member-1", "member-2", "member-3", "member-4", "member-5"),
            BOX_B: ("heldout-i-j4b", "heldout-ii-leave-pi",
                    "heldout-iii-leave-highest-wall-speed", "n-member-1"),
        },
    },
    "c": {
        "title": "2-box (c) LEAN co-run on >=48 GB cards",
        "description": (
            "Same 9 configs as (b) but each box may run 2-3 jobs concurrently "
            "on a >=48 GB card.  member-1-first still gates Idea A members 2-5; "
            "throughput uses the documented aggregate speedup assumption."),
        "co_run": True,
        "boxes": {
            BOX_A: ("member-1", "member-2", "member-3", "member-4", "member-5"),
            BOX_B: ("heldout-i-j4b", "heldout-ii-leave-pi",
                    "heldout-iii-leave-highest-wall-speed", "n-member-1"),
        },
    },
}


def plan_two_box(variant: str, *, concurrency: int = 1,
                 hours_fn: Optional[Callable[[str], float]] = None) -> dict:
    """Deterministic 2-box owner schedule for variant ``a``/``b``/``c``.

    Returns per-box ordered jobs, start/finish hours, the wall-clock and the
    landing times of Idea A ``member-1`` and of the headline held-out test
    ``heldout-i-j4b``.  ``concurrency`` is 1 for (a)/(b) and 2-3 for (c).
    """
    if variant not in TWO_BOX_VARIANTS:
        raise MultiboxError(
            f"unknown 2-box variant {variant!r}; expected one of "
            f"{list(TWO_BOX_VARIANTS)}")
    variant_spec = TWO_BOX_VARIANTS[variant]
    concurrency = int(concurrency)
    if concurrency < 1 or concurrency > 3:
        raise MultiboxError("concurrency must be 1, 2 or 3")
    if not variant_spec["co_run"] and concurrency != 1:
        raise MultiboxError(
            f"variant {variant!r} is sequential; concurrency must be 1")
    size = hours_fn or design_gpu_hours

    head_finish: dict = {}
    boxes: dict = {}
    for box_id, ordered in variant_spec["boxes"].items():
        res = simulate_box_schedule(ordered, concurrency=concurrency,
                                    hours_fn=size, head_finish=head_finish)
        boxes[box_id] = {
            "box_id": box_id,
            "assigned_configs": list(ordered),
            "start_hours": dict(res["start_hours"]),
            "finish_hours": dict(res["finish_hours"]),
            "box_finish_hours": res["box_finish_hours"],
        }
        for config in ordered:
            spec = CONFIG_SPECS[config]
            if spec.kind == "member" and spec.member == 1:
                head_finish[spec.family] = res["finish_hours"][config]

    wall = max((b["box_finish_hours"] for b in boxes.values()), default=0.0)
    m1_landing = next((b["finish_hours"][MEMBER_1] for b in boxes.values()
                       if MEMBER_1 in b["finish_hours"]), None)
    heldout_i_landing = next(
        (b["finish_hours"]["heldout-i-j4b"] for b in boxes.values()
         if "heldout-i-j4b" in b["finish_hours"]), None)
    return {
        "schema": TWO_BOX_PLAN_SCHEMA,
        "variant": variant,
        "title": variant_spec["title"],
        "description": variant_spec["description"],
        "co_run": bool(variant_spec["co_run"]),
        "concurrency": concurrency,
        "co_run_speedup": dict(CO_RUN_SPEEDUP),
        "n_boxes": len(boxes),
        "boxes": boxes,
        "box_finish_hours": {b: boxes[b]["box_finish_hours"] for b in boxes},
        "wall_clock_hours": wall,
        "member_1_landing_hours": m1_landing,
        "heldout_i_landing_hours": heldout_i_landing,
        "config_start_hours": {c: s for b in boxes.values()
                               for c, s in b["start_hours"].items()},
    }


def plan_two_box_envelope(variant: str, *, concurrency: int = 1) -> dict:
    """Low/mid/high envelope plans for a 2-box variant (A3.7 ranges)."""
    if variant not in TWO_BOX_VARIANTS:
        raise MultiboxError(f"unknown 2-box variant {variant!r}")
    return {
        "variant": variant,
        "concurrency": int(concurrency),
        "low": plan_two_box(variant, concurrency=concurrency,
                            hours_fn=lambda c: hours_range(c)[0]),
        "mid": plan_two_box(variant, concurrency=concurrency),
        "high": plan_two_box(variant, concurrency=concurrency,
                             hours_fn=lambda c: hours_range(c)[1]),
    }


# ------------------------------------------------------ box-config loading ----
def load_box_config(path) -> dict:
    """Load and validate one ``box_configs`` ``boxN.json`` file.

    Validates the schema, that every assigned config is known, and that each
    family's member-1 precedes its other members.
    """
    path = Path(path)
    if not path.is_file():
        raise MultiboxError(f"box config not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise MultiboxError(f"box config {path} is not JSON: {e}") from e
    if not isinstance(data, dict):
        raise MultiboxError(f"box config {path} is not a JSON object")
    if data.get("schema") != BOX_CONFIG_SCHEMA:
        raise MultiboxError(
            f"box config {path} schema {data.get('schema')!r} != "
            f"{BOX_CONFIG_SCHEMA!r}")
    if not data.get("box_id"):
        raise MultiboxError(f"box config {path} has no box_id")
    assigned = list(data.get("assigned_configs") or [])
    if not assigned:
        raise MultiboxError(f"box config {path} has no assigned_configs")
    unknown = [c for c in assigned if c not in CONFIG_SPECS]
    if unknown:
        raise MultiboxError(f"box config {path} unknown config(s): {unknown}")
    stop_after = data.get("stop_after")
    if stop_after is not None and stop_after not in assigned:
        raise MultiboxError(
            f"box config {path} stop_after {stop_after!r} is not one of its "
            f"assigned_configs {assigned}")
    for family, head in FAMILY_MEMBER_1.items():
        if head not in assigned:
            continue
        members = [c for c in assigned
                   if CONFIG_SPECS[c].kind == "member"
                   and CONFIG_SPECS[c].family == family]
        if members[0] != head:
            raise MultiboxError(
                f"box config {path} violates member-1-first: {head} is not the "
                f"first {family} member ({members[0]})")
    return data


def load_box_configs_dir(directory) -> dict:
    """Load a plan directory (``index.json`` + its box files) via multibox."""
    directory = Path(directory)
    index_path = directory if directory.is_file() else directory / "index.json"
    if not index_path.is_file():
        raise MultiboxError(f"box configs index not found: {index_path}")
    directory = index_path.parent
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise MultiboxError(f"index {index_path} is not JSON: {e}") from e
    if index.get("schema") != BOX_CONFIGS_INDEX_SCHEMA:
        raise MultiboxError(
            f"index {index_path} schema {index.get('schema')!r} != "
            f"{BOX_CONFIGS_INDEX_SCHEMA!r}")
    boxes_meta = index.get("boxes")
    if not isinstance(boxes_meta, dict) or not boxes_meta:
        raise MultiboxError(f"index {index_path} has no boxes")
    boxes: dict = {}
    for box_id, meta in boxes_meta.items():
        rel = (meta or {}).get("file")
        if not rel:
            raise MultiboxError(f"index {index_path} entry {box_id} has no file")
        cfg = load_box_config(directory / Path(rel).name)
        if cfg["box_id"] != box_id:
            raise MultiboxError(
                f"index {index_path} box {box_id} != file box_id "
                f"{cfg['box_id']}")
        boxes[box_id] = cfg
    return {
        "schema": "wm_n1_box_configs/1",
        "plan": index.get("plan"),
        "n_boxes": index.get("n_boxes", len(boxes)),
        "wall_clock_hours": index.get("wall_clock_hours"),
        "co_run": index.get("co_run"),
        "concurrency": index.get("concurrency"),
        "index": index,
        "boxes": boxes,
    }


def assignment_from_box_configs(directory) -> dict:
    """Return a ``{box_id: (config, ...)}`` assignment usable by run_config."""
    loaded = load_box_configs_dir(directory)
    return {box_id: tuple(cfg["assigned_configs"])
            for box_id, cfg in loaded["boxes"].items()}


# ------------------------------------------------- box-config generation -----
_SEED_BY_MEMBER: Mapping[int, int] = {1: 101, 2: 202, 3: 303, 4: 404, 5: 505}
_GPU_ENV: Mapping[str, str] = {
    "python": "3.10/3.11 (>=3.10,<3.12)",
    "torch": ">=2.7,<2.13 (cu128)",
    "install": ("python -m pip install -r "
                "output/wd01/wm_n1/bundle/requirements-wmn1.txt"),
    "note": ("GPU box runs CUDA (cu128, Blackwell sm_120); the CPU box "
             "(2.12.1+cpu) is the parity reference; verify with "
             "python -m bundle.envcheck."),
}
_CORPUS_SOURCE = "D:/rd-staging/wmn1_feat/pack_v2"

# Amendment 5 ("K = 192"): the M1-only box configs (Idea A member-1 on box A,
# held-out (i) J4->J4B on box B) train on the NEW pack_v3 and their timing gate
# / envcheck are v3 (K_max = 192).  Every other plan keeps the frozen pack_v2.
M1_ONLY_CORPUS_SOURCE = "D:/rd-staging/wmn1_feat/pack_v3"
M1_ONLY_FORMAT = C.FORMAT_V3
M1_ONLY_K_MAX = int(C.K_MAX_V3)


def _config_seed(config: str) -> int:
    spec = CONFIG_SPECS[config]
    if spec.kind == "member":
        return _SEED_BY_MEMBER.get(int(spec.member), 101)
    return 101


def _config_entry(config: str, box_id: str, start_hours: float,
                  gpu_hours: float) -> dict:
    """Per-config argv/seed/output metadata (mirrors box_configs/3box)."""
    spec = CONFIG_SPECS[config]
    seed = _config_seed(config)
    calibration_argv = None
    if spec.family == N_FAMILY:
        out = f"$WORK/members/n_member_{spec.member}"
        train_argv = ["python", "-m", "train", "--member", str(spec.member),
                      "--seed", str(seed), "--corpus", "$WORK/pack",
                      "--out", out, "--device", "cuda"]
        # N is a post-train comparator scored for the variant choice, which
        # A4.2 pins to CAL-cal (NOT the selection half).
        eval_argv = ["python", "-m", "evalh.run_eval", "--split", "cal",
                     "--cal-half", CAL_CAL,
                     "--out", f"$WORK/eval/n_member_{spec.member}_cal",
                     "--predictor", "adapter.predictor:WmN1Predictor",
                     "--reference", "evalh.climatology:ClimatologyPredictor"]
        score_argv = ["python", "-m", "evalh.run_eval", "--split", "train_oof",
                      "--out", f"$WORK/score/n_member_{spec.member}_oof",
                      "--predictor", "adapter.predictor:WmN1Predictor",
                      "--reference", "evalh.climatology:ClimatologyPredictor"]
        expected = [f"{out}/member_{spec.member}.pt", f"{out}/SEEDS.json",
                    f"{out}/TRAIN_RECEIPT.json",
                    f"$WORK/eval/n_member_{spec.member}_cal/REPORT.json"]
        trainer = "train.trainer.Trainer (v1 pure-neural comparator)"
        notes = ("variant N comparator (v1 CLI); member-1-first inside its own "
                 "family; cannot handle unseen strategies by design.")
    elif spec.kind == "member":                            # Idea A member
        out = f"$WORK/members/member_{spec.member}"
        train_argv = ["python", "-m", "train.idea_a", "--member",
                      str(spec.member), "--seed", str(seed), "--corpus",
                      "$WORK/pack", "--out", out, "--device", "cuda",
                      "--ensemble-k", "5"]
        # Selection metric reads ONLY CAL-tune (A4.2); the separate CAL-cal
        # argv carries the calibration/blend rows.
        eval_argv = ["python", "-m", "evalh.run_eval", "--split", "cal",
                     "--cal-half", CAL_TUNE,
                     "--out", f"$WORK/eval/member_{spec.member}_cal",
                     "--predictor", "adapter.predictor:IdeaABundlePredictor",
                     "--reference", "evalh.climatology:ClimatologyPredictor"]
        calibration_argv = [
            "python", "-m", "evalh.run_eval", "--split", "cal",
            "--cal-half", CAL_CAL,
            "--out", f"$WORK/eval/member_{spec.member}_cal_cal",
            "--predictor", "adapter.predictor:IdeaABundlePredictor",
            "--reference", "evalh.climatology:ClimatologyPredictor"]
        score_argv = ["python", "-m", "evalh.run_eval", "--split", "train_oof",
                      "--out", f"$WORK/score/member_{spec.member}_oof",
                      "--predictor", "adapter.predictor:IdeaABundlePredictor",
                      "--reference", "evalh.climatology:ClimatologyPredictor"]
        expected = [f"{out}/member_{spec.member}.pt",
                    f"{out}/member_{spec.member}.json",
                    f"{out}/TRAIN_RECEIPT.json",
                    f"$WORK/eval/member_{spec.member}_cal/rows.csv",
                    f"$WORK/eval/member_{spec.member}_cal/REPORT.json",
                    f"$WORK/eval/member_{spec.member}_cal_cal/rows.csv",
                    f"$WORK/eval/member_{spec.member}_cal_cal/REPORT.json"]
        trainer = "train.idea_a.IdeaATrainer"
        notes = ("member-1-first: member_k (k>1) refuses until member-1 "
                 "completes; Idea A trainer, NOT the v1 trainer.")
    else:                                                  # held-out 1-member
        out = f"$WORK/members/{config}"
        train_argv = ["python", "-m", "train.idea_a", "--member", "1",
                      "--seed", str(seed), "--corpus", "$WORK/pack",
                      "--out", out, "--device", "cuda", "--ensemble-k", "1"]
        trainer = "train.idea_a.IdeaATrainer"
        if config == "heldout-i-j4b":
            eval_argv = ["python", "-m", "evalh.idea_a", "heldout-config",
                         "--test", "i", "--out", "$WORK/eval/heldout-i-j4b",
                         "--predictor",
                         "adapter.predictor:IdeaABundlePredictor",
                         "--reference",
                         "evalh.climatology:ClimatologyPredictor"]
            score_argv = list(eval_argv)
            expected = [f"{out}/member_1.pt", f"{out}/TRAIN_RECEIPT.json",
                        "$WORK/eval/heldout-i-j4b/HELDOUT_CONFIG.json"]
            notes = ("held-out test (i): train J4 (law v2.1), evaluate J4B (law "
                     "v2.2). J4B is the J4 wall panel RE-RUN under wall law "
                     "v2.2 (REGISTRATION_J4B.md, LF 2463f5be), read ONLY by "
                     "test (i). It is NOT the chunk-5 sealed final TEST "
                     "(f8c7d760); eval_argv is the registered A3.4 held-out "
                     "config check and consumes no seal.")
        else:
            # Held-out (ii)/(iii) evaluation reads CAL-cal (not the selection
            # half): they are registered held-out evaluations, not selection.
            eval_argv = ["python", "-m", "evalh.run_eval", "--split", "cal",
                         "--cal-half", CAL_CAL,
                         "--out", f"$WORK/eval/{config}", "--predictor",
                         "adapter.predictor:IdeaABundlePredictor",
                         "--reference",
                         "evalh.climatology:ClimatologyPredictor"]
            score_argv = list(eval_argv)
            expected = [f"{out}/member_1.pt", f"{out}/TRAIN_RECEIPT.json",
                        f"$WORK/eval/{config}/rows.csv",
                        f"$WORK/eval/{config}/REPORT.json"]
            notes = ""
    return {
        "config": config, "kind": spec.kind, "family": spec.family,
        "member": int(spec.member), "seed": seed, "trainer": trainer,
        "train_argv": train_argv, "eval_argv": eval_argv,
        "calibration_argv": calibration_argv, "score_argv": score_argv,
        "expected_outputs": expected, "notes": notes,
        "corpus_source": _CORPUS_SOURCE,
        "executor": "bundle.e2e.run_config_job",
        "approval": f"$WORK/approvals/{config}/APPROVAL.json",
        "timing_gate": f"$WORK/approvals/{config}/TIMING_GATE.json",
        "box_cli": ["python", "-m", "bundle.multibox", "run-config",
                    "--box", box_id, "--config", config, "--work-dir", "$WORK",
                    "--approval", f"$WORK/approvals/{config}/APPROVAL.json",
                    "--timing-gate",
                    f"$WORK/approvals/{config}/TIMING_GATE.json"],
        "start_hours": float(start_hours), "gpu_hours": float(gpu_hours),
    }


def build_two_box_configs(variant: str, *, concurrency: int = 1) -> dict:
    """Build the ``box{A,B}.json`` payloads + index for a 2-box variant."""
    plan = plan_two_box(variant, concurrency=concurrency)
    boxes: dict = {}
    for box_id, bplan in plan["boxes"].items():
        entries = [_config_entry(c, box_id, bplan["start_hours"][c],
                                 design_gpu_hours(c))
                   for c in bplan["assigned_configs"]]
        boxes[box_id] = {
            "schema": BOX_CONFIG_SCHEMA,
            "plan": f"2-box-{variant}",
            "n_boxes": 2,
            "box_id": box_id,
            "wall_clock_hours": plan["wall_clock_hours"],
            "box_finish_hours": bplan["box_finish_hours"],
            "assigned_configs": list(bplan["assigned_configs"]),
            "member_1_first": True,
            "gpu_env": dict(_GPU_ENV),
            "corpus_source": _CORPUS_SOURCE,
            "configs": entries,
        }
    index = {
        "schema": BOX_CONFIGS_INDEX_SCHEMA,
        "plan": f"2-box-{variant}",
        "n_boxes": 2,
        "wall_clock_hours": plan["wall_clock_hours"],
        "co_run": plan["co_run"],
        "concurrency": plan["concurrency"],
        "co_run_assumption": (
            "aggregate throughput speedup 1.3x (2 jobs) / 1.5x (3 jobs); "
            "placeholder to be replaced by the 1-h timing gate"),
        "generated_by": "bundle.multibox.plan_two_box",
        "boxes": {},
    }
    for box_id, bplan in plan["boxes"].items():
        fname = "boxA.json" if box_id == BOX_A else "boxB.json"
        index["boxes"][box_id] = {
            "file": fname,
            "configs": list(bplan["assigned_configs"]),
            "finish_hours": bplan["box_finish_hours"],
            "start_hours": dict(bplan["start_hours"]),
        }
    return {"plan": plan, "boxes": boxes, "index": index}


def write_two_box_configs(root, *, variants: Sequence[str] = ("a", "b", "c"),
                          c_concurrency: int = 3) -> list:
    """Write ``root/<variant>/box{A,B}.json`` + ``index.json``; return paths."""
    root = Path(root)
    written: list = []
    for variant in variants:
        conc = int(c_concurrency) if variant == "c" else 1
        built = build_two_box_configs(variant, concurrency=conc)
        vdir = root / str(variant)
        vdir.mkdir(parents=True, exist_ok=True)
        for box_id, cfg in built["boxes"].items():
            fname = "boxA.json" if box_id == BOX_A else "boxB.json"
            written.append(_write_json(vdir / fname, cfg))
        written.append(_write_json(vdir / "index.json", built["index"]))
    return written


# --------------------------------------------- N-box plan_fleet box configs ---
def build_plan_fleet_configs(n_boxes: int) -> dict:
    """Build ``box{1..N}.json`` payloads + index for the ``plan_fleet`` plans.

    Reproduces the 3-box / 4-box ``bundle/box_configs`` from
    :func:`plan_fleet` + :func:`_config_entry` (the single source of every
    argv), so the on-disk JSONs can be regenerated instead of hand-edited.
    """
    plan = plan_fleet(int(n_boxes))
    plan_name = f"{int(n_boxes)}-box"
    boxes: dict = {}
    for box_id, configs in plan["boxes"].items():
        entries = [_config_entry(c, box_for_config(c),
                                 plan["config_start_hours"][c],
                                 design_gpu_hours(c))
                   for c in configs]
        boxes[box_id] = {
            "schema": BOX_CONFIG_SCHEMA,
            "plan": plan_name,
            "n_boxes": int(n_boxes),
            "box_id": box_id,
            "wall_clock_hours": plan["wall_clock_hours"],
            "box_finish_hours": plan["box_finish_hours"][box_id],
            "assigned_configs": list(configs),
            "member_1_first": True,
            "gpu_env": dict(_GPU_ENV),
            "corpus_source": _CORPUS_SOURCE,
            "configs": entries,
        }
    index = {
        "schema": BOX_CONFIGS_INDEX_SCHEMA,
        "plan": plan_name,
        "n_boxes": int(n_boxes),
        "wall_clock_hours": plan["wall_clock_hours"],
        "generated_by": "bundle.multibox.plan_fleet",
        "boxes": {},
    }
    for i, (box_id, configs) in enumerate(plan["boxes"].items(), start=1):
        index["boxes"][box_id] = {
            "file": f"{int(n_boxes)}box/box{i}.json",
            "configs": list(configs),
            "finish_hours": plan["box_finish_hours"][box_id],
        }
    return {"plan": plan, "boxes": boxes, "index": index}


def write_plan_fleet_configs(root, n_boxes: int) -> list:
    """Write ``root/box{1..N}.json`` + ``index.json``; return written paths."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    built = build_plan_fleet_configs(n_boxes)
    written: list = []
    for i, (box_id, cfg) in enumerate(built["boxes"].items(), start=1):
        written.append(_write_json(root / f"box{i}.json", cfg))
    written.append(_write_json(root / "index.json", built["index"]))
    return written


# --------------------------------------------------- M1-FIRST owner variant ---
#: Owner "one model first" variant (2026-10-01): box A trains Idea A member 1
#: ALONE and stops; box B runs ONLY held-out (i) J4->J4B and stops.  Members
#: 2-5, held-out (ii)/(iii) and the N comparator are NOT queued here.
M1_ONLY_PLAN = "2-box-c_m1only"
M1_ONLY_BOXES: Mapping[str, tuple] = {
    BOX_A: ("member-1",),
    BOX_B: ("heldout-i-j4b",),
}
M1_ONLY_STOP_AFTER: Mapping[str, str] = {
    BOX_A: "member-1",
    BOX_B: "heldout-i-j4b",
}
#: Quick-tune is a CAL-only improvement loop, NOT a box config: a small
#: world-stratified TRAIN subset + a fraction of the epochs, same seed.
QUICK_TUNE_PLAN = {
    "schema": "wm_n1_quick_tune/1",
    "subset_frac": 0.25,
    "epoch_frac": 0.35,
    "seed": "same as the member seed (101 for member-1)",
    "cli": ["python", "-m", "train.idea_a", "--member", "1",
            "--quick-tune", "--subset-frac", "0.25", "--epoch-frac", "0.35"],
    "note": ("CAL-only improvement loop; rank-fidelity is checked by running "
             "the same 3 configs quick vs full and comparing their rank order "
             "(see M1_FIRST_PLAN.md)"),
}


def build_m1_only_configs() -> dict:
    """Build the M1-FIRST ``box{A,B}.json`` payloads + index (no plan_fleet)."""
    boxes: dict = {}
    stop_by_box: dict = {}
    for box_id, configs in M1_ONLY_BOXES.items():
        entries = [_config_entry(c, box_id, 0.0, design_gpu_hours(c))
                   for c in configs]
        for entry in entries:
            # Amendment 5: the M1-only configs run on pack_v3 with a v3 timing
            # gate / envcheck (K_max = 192), NOT the frozen pack_v2.
            entry["corpus_source"] = M1_ONLY_CORPUS_SOURCE
            entry["format"] = M1_ONLY_FORMAT
            entry["k_max"] = M1_ONLY_K_MAX
            entry["envcheck_argv"] = [
                "python", "-m", "bundle.envcheck", "--device", "cuda",
                "--format", M1_ONLY_FORMAT]
        finish = float(sum(design_gpu_hours(c) for c in configs))
        stop_by_box[box_id] = M1_ONLY_STOP_AFTER[box_id]
        boxes[box_id] = {
            "schema": BOX_CONFIG_SCHEMA,
            "plan": M1_ONLY_PLAN,
            "n_boxes": 2,
            "box_id": box_id,
            "wall_clock_hours": finish,
            "box_finish_hours": finish,
            "assigned_configs": list(configs),
            "member_1_first": True,
            "m1_only": True,
            "stop_after": M1_ONLY_STOP_AFTER[box_id],
            "gpu_env": dict(_GPU_ENV),
            "corpus_source": M1_ONLY_CORPUS_SOURCE,
            "format": M1_ONLY_FORMAT,
            "k_max": M1_ONLY_K_MAX,
            "envcheck_argv": [
                "python", "-m", "bundle.envcheck", "--device", "cuda",
                "--format", M1_ONLY_FORMAT],
            "configs": entries,
        }
    index = {
        "schema": BOX_CONFIGS_INDEX_SCHEMA,
        "plan": M1_ONLY_PLAN,
        "n_boxes": 2,
        "wall_clock_hours": max(b["box_finish_hours"]
                                for b in boxes.values()),
        "co_run": False,
        "concurrency": 1,
        "m1_only": True,
        "generated_by": "bundle.multibox.build_m1_only_configs",
        "boxes": {},
    }
    for box_id, cfg in boxes.items():
        fname = "boxA.json" if box_id == BOX_A else "boxB.json"
        index["boxes"][box_id] = {
            "file": fname,
            "configs": list(cfg["assigned_configs"]),
            "finish_hours": cfg["box_finish_hours"],
            "stop_after": stop_by_box[box_id],
        }
    return {"boxes": boxes, "index": index}


def write_m1_only_configs(root) -> list:
    """Write ``root/{boxA,boxB,index}.json`` for the M1-FIRST variant."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    built = build_m1_only_configs()
    written: list = []
    for box_id, cfg in built["boxes"].items():
        fname = "boxA.json" if box_id == BOX_A else "boxB.json"
        written.append(_write_json(root / fname, cfg))
    written.append(_write_json(root / "index.json", built["index"]))
    return written


def _h(value) -> str:
    return "-" if value is None else f"{float(value):.1f}"


def _variant_table(variant: str, concurrency: int) -> list:
    env = plan_two_box_envelope(variant, concurrency=concurrency)
    jobs = {b: " \u2192 ".join(env["mid"]["boxes"][b]["assigned_configs"])
            for b in env["mid"]["boxes"]}
    rows = ["| box | ordered jobs | low (h) | mid (h) | high (h) |",
            "| --- | --- | --- | --- | --- |"]
    for box_id in env["mid"]["boxes"]:
        rows.append(
            f"| {box_id} | {jobs[box_id]} | "
            f"{_h(env['low']['box_finish_hours'][box_id])} | "
            f"{_h(env['mid']['box_finish_hours'][box_id])} | "
            f"{_h(env['high']['box_finish_hours'][box_id])} |")
    rows.append(
        f"| **wall-clock** | | **{_h(env['low']['wall_clock_hours'])}** | "
        f"**{_h(env['mid']['wall_clock_hours'])}** | "
        f"**{_h(env['high']['wall_clock_hours'])}** |")
    land = (f"Idea A member-1 lands at {_h(env['low']['member_1_landing_hours'])}"
            f" / {_h(env['mid']['member_1_landing_hours'])} / "
            f"{_h(env['high']['member_1_landing_hours'])} h "
            "(low / mid / high); held-out (i) lands at "
            f"{_h(env['low']['heldout_i_landing_hours'])} / "
            f"{_h(env['mid']['heldout_i_landing_hours'])} / "
            f"{_h(env['high']['heldout_i_landing_hours'])} h.")
    return rows + ["", land]


def two_box_plan_markdown() -> str:
    """Render ``bundle/GPU_PLAN_2BOX.md`` from the planner (single source)."""
    out = [
        "# WM-N1 two-GPU-box plan (`bundle/GPU_PLAN_2BOX.md`)",
        "",
        "**Owner decision (2026-10-01):** pack the WM-N1 configs onto the two "
        "rented GPU boxes with three variants: (a) all 13 configs, (b) a LEAN "
        "9-config plan, and (c) the LEAN plan with co-run on >=48 GB cards.  "
        "Generated by `bundle.multibox`; the JSON lives in "
        "`bundle/box_configs/2box/{a,b,c}/` and loads through "
        "`bundle.multibox.load_box_configs_dir`.",
        "",
        "## Design GPU-hour estimates (amendment 3 A3.7)",
        "",
        "| config family | envelope (GPU-h) | point (midpoint) |",
        "| --- | --- | --- |",
        "| Idea A member (`member-k`) | 9-17 | 13.0 |",
        "| variant N member (`n-member-k`) | 5-10 | 7.5 |",
        "| held-out test (1 member, as Idea A) | 9-17 | 13.0 |",
        "",
        "`low`/`high` apply the per-config envelope to every job; `mid` uses "
        "the midpoint.  All are design estimates to be replaced by the timing "
        "gate.",
        "",
        "## Scheduling rules",
        "",
        "1. Idea A `member-1` is the first job on box A.",
        "2. `member-k` (k>1) is released only once `member-1` finishes "
        "(member-1-first, amendment 1 A1); the N family head `n-member-1` "
        "obeys the same rule inside its own family.",
        "3. In variant (b)/(c) the held-out headline test (i) is first on box B, "
        "then (ii), (iii), then the single-member N point `n-member-1`.",
        "",
        "## Co-run assumption (variant c only)",
        "",
        "Variant (c) models 2-3 jobs per box on a >=48 GB card using an "
        "**explicit, documented aggregate-throughput assumption**: 2 "
        "simultaneous jobs -> **1.3x** aggregate, 3 -> **1.5x**.  A batch of n "
        "jobs therefore takes `sum(GPU-h)/speedup(n)` wall hours and each job "
        "in a batch reports the batch-completion time.  `member-1` still runs "
        "alone first; members 2-5 start only after `member-1` completes.  "
        "**This assumption is a placeholder and must be replaced by the 1-hour "
        "timing gate before any wall-clock claim.**",
        "",
        "## Variant (a): all 13 configs",
        "",
        "Box A = GPU-A **MAIN** (Idea A members 1-5 then N members 1-5, both "
        "family heads first); box B = GPU-B **TESTS** (the three held-out "
        "1-member tests).  Sequential.",
        "",
    ]
    out += _variant_table("a", 1)
    out += [
        "",
        "## Variant (b) LEAN: Idea A on A; held-out + N member-1 on B",
        "",
        "9 configs.  Box A = Idea A members 1-5 (member-1 first).  Box B = "
        "held-out (i) FIRST (headline), then (ii), (iii), then `n-member-1` as "
        "a **single-member point comparator** (disclosed: this is one N "
        "member, not the full 5-member N ensemble).",
        "",
    ]
    out += _variant_table("b", 1)
    out += [
        "",
        "## Variant (c): LEAN co-run on >=48 GB cards",
        "",
        "Same 9 configs as (b), 2-3 jobs per box.  `member-1` runs alone "
        "first; members 2-5 start only after it completes.  Both the 2-job and "
        "3-job modes are shown because the choice depends on the timing gate.",
        "",
        "### (c) 2 jobs per box (1.3x aggregate)",
        "",
    ]
    out += _variant_table("c", 2)
    out += ["", "### (c) 3 jobs per box (1.5x aggregate)", ""]
    out += _variant_table("c", 3)
    out += [
        "",
        "## Loading",
        "",
        "```",
        "python -c \"from bundle.multibox import load_box_configs_dir, "
        "assignment_from_box_configs; "
        "print(assignment_from_box_configs('bundle/box_configs/2box/a'))\"",
        "python -m bundle.multibox plan-2box --variant c --concurrency 3",
        "```",
        "",
        "## Approval and safety (unchanged)",
        "",
        "Each config keeps its own owner `APPROVAL.json` binding that config's "
        "written `TIMING_GATE.json` sha256 and the four frozen records.  No "
        "member trains before its approval exists; `member-k` (k>1) refuses "
        "until its family's `member-1` has completed.",
        "",
    ]
    return "\n".join(out)


# -------------------------------------------------------------------- util ---
def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def box_dir(work_dir, box_id: str) -> Path:
    return Path(work_dir) / BOXES_DIRNAME / str(box_id)


def _default_approval_path(work_dir, config: str) -> Path:
    return Path(work_dir) / APPROVALS_DIRNAME / str(config) / APPROVAL_FILENAME


def _default_gate_path(work_dir, config: str) -> Path:
    return Path(work_dir) / APPROVALS_DIRNAME / str(config) / TIMING_GATE_FILENAME


def _write_json(path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(
        f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(json.dumps(payload, indent=1).encode("utf-8"))
    os.replace(tmp, path)
    return path


def load_box_state(work_dir, box_id: str) -> dict:
    path = box_dir(work_dir, box_id) / STATE_FILENAME
    if not path.is_file():
        return {"schema": "wm_n1_multibox_box_state/1", "box_id": str(box_id),
                "completed": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise MultiboxError(f"corrupt box state {path}: {e}") from e
    data.setdefault("schema", "wm_n1_multibox_box_state/1")
    data.setdefault("box_id", str(box_id))
    data.setdefault("completed", {})
    return data


def save_box_state(work_dir, box_id: str, state: dict) -> Path:
    state["updated_utc"] = _utc()
    return _write_json(box_dir(work_dir, box_id) / STATE_FILENAME, state)


def load_fleet_state(work_dir) -> dict:
    path = Path(work_dir) / FLEET_STATE_FILENAME
    if not path.is_file():
        return {"schema": "wm_n1_multibox_fleet_state/1", "completed": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise MultiboxError(f"corrupt fleet state {path}: {e}") from e
    data.setdefault("schema", "wm_n1_multibox_fleet_state/1")
    data.setdefault("completed", {})
    return data


def save_fleet_state(work_dir, state: dict) -> Path:
    state["updated_utc"] = _utc()
    return _write_json(Path(work_dir) / FLEET_STATE_FILENAME, state)


def family_member1_completed(work_dir, family: str) -> bool:
    """Has this family's member-1 completed anywhere in the fleet?"""
    head = FAMILY_MEMBER_1.get(family)
    if head is None:
        return True
    return head in (load_fleet_state(work_dir).get("completed") or {})


def member1_completed(work_dir) -> bool:
    """Idea A member-1 completed (backwards-compatible helper)."""
    return family_member1_completed(work_dir, IDEA_A_FAMILY)


def _mark_completed(work_dir, box_id: str, config: str,
                    receipt: dict) -> None:
    with _STATE_LOCK:
        state = load_box_state(work_dir, box_id)
        state["completed"][config] = receipt
        save_box_state(work_dir, box_id, state)
        fleet = load_fleet_state(work_dir)
        fleet["completed"][config] = {"box_id": str(box_id),
                                      "completed_utc": receipt.get("completed_utc")}
        save_fleet_state(work_dir, fleet)


# ---------------------------------------------------------------- approval ---
def _records(records: Optional[Mapping]) -> Mapping:
    return records if records is not None else DEFAULT_RECORDS


def require_config_approval(approval_path, *, config: str,
                            timing_gate_path=None,
                            records: Optional[Mapping] = None) -> dict:
    """Validate the per-config owner approval, or raise.

    Binds: the written ``TIMING_GATE.json`` raw-bytes sha256, the registration
    LF-sha256, ALL FIVE amendment LF-sha256 values (A1-A5 plus the A6 state
    global token) and the CAL split LF-sha256 (CRLF-safe).  This gate governs
    EVERY ``APPROVAL.json``, including the ``2box/c_m1only`` boxA/boxB configs.
    """
    path = Path(approval_path)
    if not path.is_file():
        raise ConfigApprovalMissing(f"owner approval file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                                # noqa: BLE001
        raise ConfigApprovalMissing(
            f"owner approval file is not JSON: {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigApprovalMissing(f"owner approval is not an object: {path}")
    if not data.get("approved"):
        raise ConfigApprovalMissing(f"approval {path} is not an owner approval")
    schema = data.get("schema")
    if schema is not None and schema != APPROVAL_SCHEMA:
        raise ConfigApprovalMissing(
            f"approval schema {schema!r} != {APPROVAL_SCHEMA!r}")

    declared = {
        "registration_lf_sha256": LF_SHA_REGISTRATION,
        "amendment_1_lf_sha256": LF_SHA_AMENDMENT_1,
        "amendment_2_lf_sha256": LF_SHA_AMENDMENT_2,
        "amendment_3_lf_sha256": LF_SHA_AMENDMENT_3,
        "amendment_4_lf_sha256": LF_SHA_AMENDMENT_4,
        "amendment_5_lf_sha256": LF_SHA_AMENDMENT_5,
        "amendment_6_lf_sha256": LF_SHA_AMENDMENT_6,
        "cal_split_lf_sha256": LF_SHA_CAL_SPLIT_A4,
    }
    missing = [k for k in declared if not data.get(k)]
    if missing:
        raise ConfigApprovalMissing(
            f"approval {path} is missing binding fields: {missing} for "
            f"config {config!r}")
    for key, want in declared.items():
        if data[key] != want:
            raise ConfigApprovalMissing(
                f"approval {key} {data.get(key)!r} != frozen {want!r}")

    # Re-hash the actual frozen records when present (CRLF-safe).
    rec = _records(records)
    for key, frozen in (("registration", LF_SHA_REGISTRATION),
                        ("amendment_1", LF_SHA_AMENDMENT_1),
                        ("amendment_2", LF_SHA_AMENDMENT_2),
                        ("amendment_3", LF_SHA_AMENDMENT_3),
                        ("amendment_4", LF_SHA_AMENDMENT_4),
                        ("amendment_5", LF_SHA_AMENDMENT_5),
                        ("amendment_6", LF_SHA_AMENDMENT_6),
                        ("cal_split", LF_SHA_CAL_SPLIT_A4)):
        entry = rec.get(key)
        if entry is None:
            continue
        if isinstance(entry, (tuple, list)):
            rec_path, rec_sha = entry[0], entry[1]
        else:
            rec_path, rec_sha = entry, frozen
        rec_path = Path(rec_path)
        if rec_sha != frozen:
            raise ConfigApprovalMissing(
                f"frozen record {key} configured sha {rec_sha!r} != {frozen!r}")
        if not rec_path.is_file():
            raise ConfigApprovalMissing(
                f"required frozen record not found: {rec_path}")
        got = lf_sha256(rec_path)
        if got != frozen:
            raise ConfigApprovalMissing(
                f"{rec_path.name} LF-sha256 {got} != {frozen}")

    # Timing gate: declared field must equal the actual written file.
    gate_path = timing_gate_path
    if gate_path is None:
        base = path.parent
        rel = data.get("timing_gate_path") or TIMING_GATE_FILENAME
        gate_path = (Path(rel) if Path(rel).is_absolute() else base / Path(rel))
    gate_path = Path(gate_path)
    if not gate_path.is_file():
        raise ConfigApprovalMissing(f"timing gate file not found: {gate_path}")
    got_gate = _sha256_file(gate_path)
    if data.get("timing_gate_sha256") != got_gate:
        raise ConfigApprovalMissing(
            f"approval timing_gate_sha256 {data.get('timing_gate_sha256')!r} "
            f"!= {gate_path} sha256 {got_gate}")

    declared_config = data.get("config")
    if declared_config is not None and declared_config != config:
        raise ConfigApprovalMissing(
            f"approval is for config {declared_config!r}, not {config!r}")
    return dict(data, approval_path=str(path), config=config,
                timing_gate_resolved=str(gate_path))


def write_config_approval(path, *, config: str, timing_gate_path=None,
                          owner: str = "owner", approved_utc: str = None,
                          records: Optional[Mapping] = None) -> dict:
    """Write one per-config ``APPROVAL.json`` binding all eight frozen records.

    The canonical way to produce an APPROVAL.json for ANY config (including
    the ``2box/c_m1only`` boxA/boxB configs).  The amendment LF-shas (A1-A5
    plus the A6 state global token), the CAL split LF-sha and the config's
    written ``TIMING_GATE.json`` sha are all recorded, so
    :func:`require_config_approval` accepts it only while the actual bytes
    still match.  ``records`` is accepted for symmetry with the validator;
    the writer emits the frozen constants.
    """
    gate = timing_gate_path or TIMING_GATE_FILENAME
    body = {
        "schema": APPROVAL_SCHEMA,
        "approved": True,
        "owner": owner,
        "approved_utc": approved_utc or _utc(),
        "config": str(config),
        "timing_gate_path": str(gate),
        "timing_gate_sha256": _sha256_file(gate),
        "registration_lf_sha256": LF_SHA_REGISTRATION,
        "amendment_1_lf_sha256": LF_SHA_AMENDMENT_1,
        "amendment_2_lf_sha256": LF_SHA_AMENDMENT_2,
        "amendment_3_lf_sha256": LF_SHA_AMENDMENT_3,
        "amendment_4_lf_sha256": LF_SHA_AMENDMENT_4,
        "amendment_5_lf_sha256": LF_SHA_AMENDMENT_5,
        "amendment_6_lf_sha256": LF_SHA_AMENDMENT_6,
        "cal_split_lf_sha256": LF_SHA_CAL_SPLIT_A4,
    }
    _write_json(path, body)
    return body


# ----------------------------------------------------------------- runtime ---
def _resolve_default_executor() -> Callable:
    if "." not in DEFAULT_EXECUTOR_ENTRY:
        raise MultiboxError("bad DEFAULT_EXECUTOR_ENTRY")
    module, attr = DEFAULT_EXECUTOR_ENTRY.rsplit(".", 1)
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError as e:
        raise MultiboxError(
            f"executor piece missing: {DEFAULT_EXECUTOR_ENTRY} "
            f"(import {module} failed: {e.name}); pass an explicit executor "
            f"or wait for the parallel bundle e2e piece") from e
    fn = getattr(mod, attr, None)
    if fn is None:
        raise MultiboxError(
            f"executor entry point missing: {DEFAULT_EXECUTOR_ENTRY}")
    return fn


def run_config(box_id: str, config: str, *, work_dir,
               executor: Optional[Callable] = None,
               approval_path=None, timing_gate_path=None,
               records: Optional[Mapping] = None,
               assignment: Optional[Mapping] = None) -> dict:
    """Run one config on one box; refuse wrong boxes and out-of-order members.

    Idempotent per box: a config already recorded complete is returned as-is.
    """
    if config not in CONFIG_SPECS:
        raise MultiboxError(
            f"unknown config {config!r}; expected one of {list(CONFIG_SPECS)}")
    configs = configs_for_box(box_id, assignment)
    if config not in configs:
        raise UnassignedConfigRefusal(
            f"box {box_id!r} is assigned {list(configs)}; refusing unassigned "
            f"config {config!r}")

    work_dir = Path(work_dir)
    state = load_box_state(work_dir, box_id)
    if config in state["completed"]:
        return dict(state["completed"][config], skipped=True)

    idx = member_index(config)
    spec = CONFIG_SPECS[config]
    if idx is not None and idx > 1 and not family_member1_completed(
            work_dir, spec.family):
        raise BoxOrderError(
            f"box {box_id!r}: {config} refused; {FAMILY_MEMBER_1[spec.family]} "
            "has not completed (amendment 1 A1: member-1-first per family)")

    if approval_path is None:
        approval_path = _default_approval_path(work_dir, config)
    if timing_gate_path is None:
        timing_gate_path = _default_gate_path(work_dir, config)
    approval = require_config_approval(
        approval_path, config=config, timing_gate_path=timing_gate_path,
        records=records)

    fn = executor or _resolve_default_executor()
    bdir = box_dir(work_dir, box_id)
    bdir.mkdir(parents=True, exist_ok=True)
    out = fn(box_id, config, str(bdir))
    if out is None:
        out = {}
    if not isinstance(out, dict):
        raise MultiboxError(
            f"executor for {config} returned {type(out).__name__}, expected "
            "a receipt dict")
    receipt = dict(out)
    receipt.update(config=config, box_id=str(box_id),
                   kind=CONFIG_SPECS[config].kind, member=idx,
                   work_dir=str(bdir), approval_path=str(approval_path),
                   timing_gate_sha256=approval["timing_gate_sha256"],
                   completed_utc=_utc())
    _mark_completed(work_dir, box_id, config, receipt)
    return receipt


def run_box(box_id: str, *, work_dir,
            executor: Optional[Callable] = None,
            approval_paths: Optional[Mapping] = None,
            timing_gate_paths: Optional[Mapping] = None,
            records: Optional[Mapping] = None,
            assignment: Optional[Mapping] = None,
            stop_after: Optional[str] = None) -> dict:
    """Run every config assigned to ``box_id`` in member-1-first order.

    ``stop_after`` (a config id assigned to the box) stops the box after that
    config; later assigned configs are not run.  The M1-FIRST variant uses
    ``stop_after="member-1"`` on box A and ``stop_after="heldout-i-j4b"`` on
    box B, so a box never advances past the one model it is approved to run.
    """
    configs = configs_for_box(box_id, assignment)
    order = planned_order(configs)
    if stop_after is not None:
        if stop_after not in order:
            raise MultiboxError(
                f"stop_after {stop_after!r} is not in box {box_id!r} order "
                f"{list(order)}")
        order = order[:order.index(stop_after) + 1]
    approval_paths = approval_paths or {}
    timing_gate_paths = timing_gate_paths or {}
    receipts = {}
    for config in order:
        receipts[config] = run_config(
            box_id, config, work_dir=work_dir, executor=executor,
            approval_path=approval_paths.get(config),
            timing_gate_path=timing_gate_paths.get(config),
            records=records, assignment=assignment)
    return dict(box_id=str(box_id), configs=list(order), receipts=receipts,
                stop_after=stop_after)


def run_box_config(box_config_path, *, work_dir,
                   executor: Optional[Callable] = None,
                   approval_paths: Optional[Mapping] = None,
                   timing_gate_paths: Optional[Mapping] = None,
                   records: Optional[Mapping] = None) -> dict:
    """Run one ``box_configs`` file, honouring its ``stop_after`` field."""
    cfg = load_box_config(box_config_path)
    assignment = {cfg["box_id"]: tuple(cfg["assigned_configs"])}
    return run_box(cfg["box_id"], work_dir=work_dir, executor=executor,
                   approval_paths=approval_paths,
                   timing_gate_paths=timing_gate_paths, records=records,
                   assignment=assignment, stop_after=cfg.get("stop_after"))


def run_fleet(*, work_dir, executor: Optional[Callable] = None,
              approval_paths: Optional[Mapping] = None,
              timing_gate_paths: Optional[Mapping] = None,
              records: Optional[Mapping] = None,
              assignment: Optional[Mapping] = None,
              parallel: bool = False, max_workers: Optional[int] = None) -> dict:
    """Run all boxes; each family's member-1 box first, the rest in parallel."""
    assignment = assignment or DEFAULT_BOX_ASSIGNMENT
    boxes = _ordered_boxes(assignment)
    heads = tuple(FAMILY_MEMBER_1[f] for f in FAMILY_ORDER)
    phase1 = [b for b in boxes if any(h in assignment[b] for h in heads)]
    rest = [b for b in boxes if b not in phase1]

    def _one(box_id: str) -> tuple:
        return box_id, run_box(
            box_id, work_dir=work_dir, executor=executor,
            approval_paths=approval_paths, timing_gate_paths=timing_gate_paths,
            records=records, assignment=assignment)

    results = {}
    for box_id in phase1:
        results[box_id] = _one(box_id)[1]
    if rest:
        if parallel:
            workers = int(max_workers or min(len(rest), 8))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for box_id, res in pool.map(_one, rest):
                    results[box_id] = res
        else:
            for box_id in rest:
                results[box_id] = _one(box_id)[1]
    return dict(schema="wm_n1_multibox_fleet_run/1",
                parallel=bool(parallel),
                one_config_per_box=all(len(v) == 1
                                       for v in assignment.values()),
                boxes={b: results[b] for b in boxes})


# --------------------------------------------------------------------- cli ---
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bundle.multibox")
    sub = p.add_subparsers(dest="command")

    sub.add_parser("plan", help="print the box -> config assignment")

    pf = sub.add_parser(
        "plan-fleet",
        help="schedule all configs onto N physical boxes (GPU_PLAN table)")
    pf.add_argument("--boxes", type=int, default=3)
    pf.add_argument("--json", action="store_true",
                    help="emit the machine-readable plan instead of the table")

    p2 = sub.add_parser(
        "plan-2box",
        help="owner 2-box variants a/b/c (co-run for c; GPU_PLAN_2BOX.md)")
    p2.add_argument("--variant", choices=("a", "b", "c"), default="a")
    p2.add_argument("--concurrency", type=int, default=1,
                    help="1 (a/b) or 2-3 (c co-run)")
    p2.add_argument("--json", action="store_true",
                    help="emit the machine-readable plan")
    p2.add_argument("--markdown", action="store_true",
                    help="emit bundle/GPU_PLAN_2BOX.md content")

    rb = sub.add_parser("run-box", help="run one box's assigned configs")
    rb.add_argument("--box", required=True)
    rb.add_argument("--work-dir", required=True)
    rb.add_argument("--stop-after", default=None,
                    help="stop after this assigned config (M1-FIRST mode)")
    rb.add_argument("--executor", default=None,
                    help="dotted callable (default bundle.e2e.run_config_job)")

    rbc = sub.add_parser(
        "run-box-config",
        help="run one box_configs JSON file (honours its stop_after field)")
    rbc.add_argument("--config", required=True, help="path to boxA.json")
    rbc.add_argument("--work-dir", required=True)
    rbc.add_argument("--executor", default=None)

    rc = sub.add_parser("run-config", help="run one config on one box")
    rc.add_argument("--box", required=True)
    rc.add_argument("--config", required=True)
    rc.add_argument("--work-dir", required=True)
    rc.add_argument("--approval", default=None)
    rc.add_argument("--timing-gate", default=None)
    rc.add_argument("--executor", default=None)

    rf = sub.add_parser("run-fleet", help="run every box (member-1 first)")
    rf.add_argument("--work-dir", required=True)
    rf.add_argument("--executor", default=None)
    rf.add_argument("--parallel", action="store_true")
    rf.add_argument("--max-workers", type=int, default=None)
    return p


def _resolve_cli_executor(dotted: Optional[str]) -> Optional[Callable]:
    if not dotted:
        return None
    module, attr = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in (None, "plan"):
        print(json.dumps(assignment_receipt(), indent=1))
        return 0
    if args.command == "plan-fleet":
        plan = plan_fleet(args.boxes)
        print(json.dumps(plan, indent=1) if args.json else format_plan(plan))
        return 0
    if args.command == "plan-2box":
        if args.markdown:
            print(two_box_plan_markdown())
            return 0
        plan = plan_two_box(args.variant, concurrency=args.concurrency)
        if args.json:
            print(json.dumps(plan, indent=1))
            return 0
        view = {
            "n_boxes": plan["n_boxes"],
            "wall_clock_hours": plan["wall_clock_hours"],
            "boxes": {b: d["assigned_configs"]
                      for b, d in plan["boxes"].items()},
            "box_finish_hours": plan["box_finish_hours"],
            "config_start_hours": plan["config_start_hours"],
            "config_hours": {c: design_gpu_hours(c)
                             for c in plan["config_start_hours"]},
        }
        print(format_plan(view))
        return 0
    executor = _resolve_cli_executor(getattr(args, "executor", None))
    if args.command == "run-config":
        out = run_config(args.box, args.config, work_dir=args.work_dir,
                         executor=executor, approval_path=args.approval,
                         timing_gate_path=args.timing_gate)
    elif args.command == "run-box":
        out = run_box(args.box, work_dir=args.work_dir, executor=executor,
                      stop_after=args.stop_after)
    elif args.command == "run-box-config":
        out = run_box_config(args.config, work_dir=args.work_dir,
                             executor=executor)
    elif args.command == "run-fleet":
        out = run_fleet(args.work_dir, executor=executor,
                        parallel=args.parallel, max_workers=args.max_workers)
    else:                                                 # pragma: no cover
        raise MultiboxError(f"unknown command {args.command!r}")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

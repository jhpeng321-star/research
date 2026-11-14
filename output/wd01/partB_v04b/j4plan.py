"""J4 wall-pilot plan access: split FIRST, by name (registration s.2).

The J4 plan (schema wm59_walls_pilot_world_plan/1, plan_seed 20261101,
44 worlds x 4 wall branches) is the ONLY split authority.  The
registered TRAIN/CAL name lists are asserted against it, so a drifted
or superseded plan cannot silently re-split the data.  HOLDOUT: the J4
plan contains none (the source chunk-4 plan's 10 holdout worlds were
excluded at campaign design); chunk-5 is never enumerated here.

Every run directory this module returns lives under the E:/research-dense
runs roots (wdpaths.remap; no D: paths, WD_FORBID_D=1).
"""
from __future__ import annotations

import json
from pathlib import Path

from . import _paths  # noqa: F401  (sys.path shims)
from . import reg

import wdpaths  # noqa: E402  (read-only reference module)

# Registered split (registration s.2, counted BY NAME from the plan).
# NOTE (single-name drift, disclosed everywhere): the registration text
# lists "w8662m2700p10t300" but the J4 plan -- the registration's own
# split authority -- holds w8662m2600p10t300 (loading 2600, TRAIN,
# same seed/pen/t0 cell).  split_by_name() follows the PLAN and reports
# the drift via SPLIT_NAME_DRIFT; it is never silently dropped.
SPLIT_NAME_DRIFT = {"w8662m2700p10t300": "w8662m2600p10t300"}
TRAIN_WORLDS = (
    "w8659m2700p10t420", "w8660m2700p10t300", "w8661m2600p10t240",
    "w8662m2700p10t300", "w8663m2700p10t480", "w8664m2600p10t360",
    "w8665m2700p10t240", "w8666m2700p10t480", "w8667m2700p10t420",
    "w8668m2600p10t240", "w8669m2700p10t360", "w8670m2700p10t300",
    "w8676m2700p20t360", "w8677m2700p20t360", "w8678m2700p20t300",
    "w8679m2400p20t540", "w8680m2600p20t300", "w8681m2700p20t420",
    "w8682m2400p20t240", "w8683m2400p20t540", "w8684m2700p20t360",
    "w8685m2700p20t480", "w8686m2600p20t240", "w8687m2600p20t480",
    "w8688m2700p20t420", "w8693m2600p40t420", "w8694m2600p40t240",
    "w8695m2400p40t480", "w8696m2600p40t420", "w8697m2700p40t240",
    "w8698m2600p40t300", "w8699m2700p40t240", "w8700m2600p40t420",
    "w8701m2700p40t540", "w8702m2700p40t300", "w8703m2600p40t540",
    "w8704m2700p40t480", "w8705m2700p40t300", "w8706m2400p40t360")
CAL_WORLDS = (
    "w8657m2700p10t300", "w8658m2600p10t360", "w8674m2600p20t300",
    "w8675m2700p20t300", "w8692m2600p40t300")


class PlanMismatch(RuntimeError):
    """The J4 plan disagrees with the registered split."""


def j4_plan_path(data_root: str | Path | None = None) -> Path:
    import wdpaths as W
    root = Path(data_root) if data_root else Path(W.RESEARCH_DENSE_ROOT)
    return Path(wdpaths.remap(root / reg.J4_PLAN_REL))


def load_plan(data_root: str | Path | None = None) -> dict:
    """The raw J4 plan dict; the schema/seed are asserted."""
    p = j4_plan_path(data_root)
    plan = json.loads(p.read_text(encoding="utf-8"))
    if plan.get("schema") != reg.J4_PLAN_SCHEMA:
        raise PlanMismatch(f"{p}: schema {plan.get('schema')!r} != "
                           f"{reg.J4_PLAN_SCHEMA!r}")
    if int(plan.get("plan_seed", -1)) != reg.J4_PLAN_SEED:
        raise PlanMismatch(f"{p}: plan_seed {plan.get('plan_seed')!r} != "
                           f"{reg.J4_PLAN_SEED}")
    return plan


def plan_worlds(data_root: str | Path | None = None) -> dict:
    """{world: world record} of the J4 plan (44 entries)."""
    plan = load_plan(data_root)
    return {str(w["world"]): w for w in plan["worlds"]}


def split_by_name(data_root: str | Path | None = None) -> dict[str, dict]:
    """{world: record} for the TRAIN worlds, counted BY NAME from the
    plan (the registration's own split authority).

    Assertions: exactly 39 TRAIN + 5 CAL plan worlds, every registered
    name present with the right plan split label, and every plan TRAIN
    world accounted for by the registered list EXCEPT the recorded
    single-name drift (SPLIT_NAME_DRIFT, reported, never silent).
    CAL worlds are returned by cal_worlds() only -- a TRAIN build never
    touches them.
    """
    ws = plan_worlds(data_root)
    names = set(ws)
    if len(names) != 44:
        raise PlanMismatch(f"J4 plan holds {len(names)} worlds != 44")
    plan_train = {w for w, rec in ws.items()
                  if rec.get("split") == "TRAIN"}
    plan_cal = {w for w, rec in ws.items()
                if rec.get("split") == "CALIBRATION"}
    if len(plan_train) != len(TRAIN_WORLDS) or \
            len(plan_cal) != len(CAL_WORLDS):
        raise PlanMismatch(
            f"J4 plan split counts {len(plan_train)} TRAIN / "
            f"{len(plan_cal)} CAL != registered "
            f"{len(TRAIN_WORLDS)}/{len(CAL_WORLDS)}")
    drifted = plan_train ^ set(TRAIN_WORLDS)
    if drifted and drifted != set(SPLIT_NAME_DRIFT.values()) | \
            set(SPLIT_NAME_DRIFT):
        raise PlanMismatch(
            f"J4 plan TRAIN names differ from the registration beyond "
            f"the recorded drift {SPLIT_NAME_DRIFT}: {sorted(drifted)}")
    if plan_cal != set(CAL_WORLDS):
        raise PlanMismatch(
            f"J4 plan CAL names differ from the registration: "
            f"{sorted(plan_cal ^ set(CAL_WORLDS))}")
    for w, rec in ws.items():
        if w in TRAIN_WORLDS and rec.get("split") != "TRAIN":
            raise PlanMismatch(
                f"J4 plan split of {w} = {rec.get('split')!r}, "
                "registered TRAIN")
        if w in CAL_WORLDS and rec.get("split") != "CALIBRATION":
            raise PlanMismatch(
                f"J4 plan split of {w} = {rec.get('split')!r}, "
                "registered CALIBRATION")
    return {w: ws[w] for w in sorted(plan_train)}


def cal_worlds(data_root: str | Path | None = None) -> dict[str, dict]:
    """{world: record} for the 5 registered CAL worlds (for the ONE
    guarded CAL read only; split assertions as split_by_name)."""
    split_by_name(data_root)          # full-plan integrity first
    ws = plan_worlds(data_root)
    return {w: ws[w] for w in CAL_WORLDS}


def wall_branches(rec: dict) -> list[dict]:
    """The 4 wall branch records of one plan world (asserted names)."""
    brs = list(rec.get("branches", ()))
    got = tuple(b["arm"] for b in brs)
    if got != reg.WALL_ARMS:
        raise PlanMismatch(f"{rec.get('world')}: wall arms {got} != "
                           f"{reg.WALL_ARMS}")
    return brs


def runs_root(data_root: str | Path | None = None) -> Path:
    import wdpaths as W
    root = Path(data_root) if data_root else Path(W.RESEARCH_DENSE_ROOT)
    return Path(wdpaths.remap(root / reg.J4_RUNS_ROOT))


def bases_root(data_root: str | Path | None = None) -> Path:
    import wdpaths as W
    root = Path(data_root) if data_root else Path(W.RESEARCH_DENSE_ROOT)
    return Path(wdpaths.remap(root / reg.J4_BASES_ROOT))

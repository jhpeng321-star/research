"""WD-01-PB-V04A source registry (registration section 2).

Enumerates every (chunk, world, source) of the v0.4a dataset root from
each chunk's WORLD_PLAN.json.  The split of a world is read from the
plan BEFORE any run file is touched; HOLDOUT worlds are never
enumerated (they cannot reach a spec), tools/holdout_guard.py +
protocols/holdout_exclusions.json are applied on top, and an audit hook
records every path this process opens so a close-out assertion can show
that no opened path names a HOLDOUT world.

Source naming inside the v0.4a root (``<world>__<source>_*.npz``):

  chunk-4        base, native_base, native, pi, fixed1, fixed2,
                 switching1..3               (all PRIMARY)
  chunk-2/3      base (damping fleet, PRIMARY);
                 native, pi  = the J3 TRUE fleet-native / fleet-PI arms
                 from campaign_stage1_chunk{2,3}_truearms (PRIMARY);
                 swnative, swpi = the campaign ``__native``/``__pi``
                 random per-CAV SWITCHING arms (AUXILIARY, training only)
  chunk1 / chunk1_fix27
                 base (damping, AUXILIARY: the registration lists only
                 the chunk-2/3 bases as primary), swnative, swpi
                 (AUXILIARY)

Canonical policy classes (POLICY_CLASSES):
  damping_fleet native_fleet pi_fleet fixed_mix switching_mix wall

WM-N1 label coverage extension (owner-approved 2026-10-04, WM-N1-LABELCOV):
  chunk1 / chunk1_fix27 / chunk2 / chunk3 additionally enumerate the
  campaign branch arms fixed1, fixed2 (fixed_mix) and switching1..3
  (switching_mix) from ``runs/campaign_stage1_chunk*/<world>__<arm>``,
  and chunk4 additionally enumerates the four wall arms of the walls
  pilot (policy ``wall``) from
  ``runs/campaign_walls_pilot_chunk4/<world>__<arm>`` (their base run
  stays the shared ``campaign_stage1_chunk4`` damping base).  All new
  sources are branch-type; HOLDOUT is still never enumerated and every
  spec still carries its plan split.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # output/wd01 (wdpaths)
import wdpaths  # noqa: E402

RD = Path(wdpaths.RESEARCH_DENSE_ROOT)

POLICY_CLASSES = ("switching_mix", "damping_fleet", "native_fleet",
                  "pi_fleet", "fixed_mix", "wall")

# WM-N1 label coverage (2026-10-04): the walls-pilot branch arms.  Their
# runs live in their own root; the shared damping base stays in
# campaign_stage1_chunk4 (the walls plan records bases_root).
WALLS_PLAN = RD / "output/campaign_walls_v1/results_plan/WORLD_PLAN.json"
WALLS_ROOT = RD / "runs/campaign_walls_pilot_chunk4"
WALL_ARMS = ("w-lo-early", "w-lo-late", "w-mid-early", "w-mid-late")
# campaign fixed / switching branch arm names added for coverage
FIXED_ARMS = ("fixed1", "fixed2")
SWITCH_ARMS = ("switching1", "switching2", "switching3")

PLAN = {
    "chunk4": RD / "output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json",
    "chunk3": RD / "output/campaign_stage1_v3/results_plan_chunk3/WORLD_PLAN.json",
    "chunk2": RD / "output/campaign_stage1_v2/results_plan_chunk2/WORLD_PLAN.json",
    "chunk1": RD / "output/campaign_stage1_chunk1/results_plan_wm44/WORLD_PLAN.json",
    "chunk1_fix27": RD / "output/campaign_stage1_v3/results_plan_fix27/WORLD_PLAN.json",
}
RUNS = {
    "chunk4": RD / "runs/campaign_stage1_chunk4",
    "chunk3": RD / "runs/campaign_stage1_chunk3",
    "chunk2": RD / "runs/campaign_stage1_chunk2",
    "chunk1": RD / "runs/campaign_stage1_chunk1",
    "chunk1_fix27": RD / "runs/campaign_stage1_chunk1_fix27",
}
TRUEARMS = {
    "chunk2": RD / "runs/campaign_stage1_chunk2_truearms",
    "chunk3": RD / "runs/campaign_stage1_chunk3_truearms",
}
BASES_ROOT = {"chunk1_fix27": RD / "runs/campaign_stage1_verify27"}
DEFAULT_SPEED_MODE = {"chunk4": 27, "chunk3": 27, "chunk2": 27,
                      "chunk1": 31, "chunk1_fix27": 27}
V032_MANIFEST = RD / "output/wd01_partB_v0data_v032/MANIFEST.json"
ALL_CHUNKS = ("chunk4", "chunk3", "chunk2", "chunk1", "chunk1_fix27")
KEEP_SPLITS = ("TRAIN", "CALIBRATION")


@dataclass
class SrcSpec:
    chunk: str
    world: str
    source: str
    split: str
    role: str                 # primary | aux
    policy: str               # POLICY_CLASSES
    verify_kind: str          # actions_v04.ARM_KINDS
    type: str                 # base_like | branch
    arm_dir: str
    base_dir: str
    t0_s: float
    demand_veh_h: int
    penetration: float
    runs_root: str
    default_speed_mode: int
    note: str = ""

    def key(self) -> str:
        return f"{self.chunk}/{self.world}__{self.source}"


def chunk_worlds(chunk: str) -> tuple[list[dict], list[str]]:
    """(worlds with split in KEEP_SPLITS sorted by name, HOLDOUT names).
    The split comes from the plan; nothing else is opened here."""
    plan = json.loads(Path(wdpaths.remap(PLAN[chunk]))
                      .read_text(encoding="utf-8"))
    keep = sorted((w for w in plan["worlds"] if w["split"] in KEEP_SPLITS),
                  key=lambda w: w["world"])
    hold = sorted(w["world"] for w in plan["worlds"]
                  if w["split"] not in KEEP_SPLITS)
    return keep, hold


def walls_plan_worlds() -> tuple[list[dict], list[str]]:
    """(TRAIN/CAL worlds of the walls pilot sorted by name, HOLDOUT names).

    Read from the frozen walls-pilot plan; the same KEEP_SPLITS and
    never-HOLDOUT convention as :func:`chunk_worlds` applies."""
    plan = json.loads(Path(wdpaths.remap(WALLS_PLAN))
                      .read_text(encoding="utf-8"))
    keep = sorted((w for w in plan["worlds"] if w["split"] in KEEP_SPLITS),
                  key=lambda w: w["world"])
    hold = sorted(w["world"] for w in plan["worlds"]
                  if w["split"] not in KEEP_SPLITS)
    return keep, hold


def superseded_chunk1_worlds() -> set[str]:
    """chunk-1 worlds superseded by chunk1_fix27 (p40 re-runs), taken from
    the frozen v0.3.2 MANIFEST (its superseded_by flags)."""
    man = json.loads(Path(wdpaths.remap(V032_MANIFEST))
                     .read_text(encoding="utf-8"))
    return {e["world"] for e in man["worlds"].values()
            if e.get("chunk_id") == "chunk1" and e.get("superseded_by")}


def _spec(chunk, w, source, role, policy, kind, typ, arm, base, note=""):
    return SrcSpec(chunk=chunk, world=w["world"], source=source,
                   split=w["split"], role=role, policy=policy,
                   verify_kind=kind, type=typ, arm_dir=str(arm),
                   base_dir=str(base), t0_s=float(w["t0_s"]),
                   demand_veh_h=int(w["loading_veh_h"]),
                   penetration=float(w["penetration"]),
                   runs_root=str(RUNS[chunk]),
                   default_speed_mode=DEFAULT_SPEED_MODE[chunk], note=note)


def _wall_spec(w, arm: str) -> SrcSpec:
    """One walls-pilot branch source (policy ``wall``, runs in WALLS_ROOT;
    the shared base stays the campaign_stage1_chunk4 damping base)."""
    name = w["world"]
    return SrcSpec(
        chunk="chunk4", world=name, source=arm, split=w["split"],
        role="primary", policy="wall", verify_kind="wall", type="branch",
        arm_dir=str(WALLS_ROOT / f"{name}__{arm}"),
        base_dir=str(RUNS["chunk4"] / f"{name}__base"),
        t0_s=float(w["t0_s"]), demand_veh_h=int(w["loading_veh_h"]),
        penetration=float(w["penetration"]), runs_root=str(WALLS_ROOT),
        default_speed_mode=DEFAULT_SPEED_MODE["chunk4"],
        note="walls pilot branch arm (WM-N1 label coverage 2026-10-04)")


def enumerate_specs(chunks=ALL_CHUNKS, worlds: set | None = None,
                    *, exclusions: dict | None = None,
                    include_superseded: bool = False) -> list[SrcSpec]:
    """All source specs for the requested chunks (TRAIN/CAL worlds only).

    ``include_superseded`` (WM-N1 label coverage 2026-10-04): also
    enumerate chunk1 worlds superseded by chunk1_fix27.  The frozen v0.4a
    root skipped them; the pack_v3 TRAIN corpus keeps their items, so the
    coverage build must be able to resolve their raw runs.  Default False
    keeps the frozen-root behaviour byte-for-byte.
    """
    sys.path.insert(0, str(RD / "tools"))
    import holdout_guard as HG
    excl = HG.excluded_roots(exclusions)
    sup = superseded_chunk1_worlds() if "chunk1" in chunks else set()
    out: list[SrcSpec] = []
    for chunk in chunks:
        keep, hold = chunk_worlds(chunk)
        runs = RUNS[chunk]
        for w in keep:
            name = w["world"]
            if name in excl:
                raise AssertionError(
                    f"holdout_exclusions world {name} has split "
                    f"{w['split']!r} in {chunk}: refusing")
            if worlds and name not in worlds:
                continue
            if chunk == "chunk1" and name in sup and not include_superseded:
                continue                      # superseded by fix27
            base = (BASES_ROOT.get(chunk, runs)) / f"{name}__base"
            if chunk == "chunk4":
                out.append(_spec(chunk, w, "base", "primary", "damping_fleet",
                                 "damping_fleet", "base_like",
                                 runs / f"{name}__base", base))
                out.append(_spec(chunk, w, "native_base", "primary",
                                 "native_fleet", "native_fleet", "base_like",
                                 runs / f"{name}__native_base", base))
                for src, pol, kind in (
                        ("native", "native_fleet", "native_fleet"),
                        ("pi", "pi_fleet", "pi_fleet"),
                        ("fixed1", "fixed_mix", "fixed"),
                        ("fixed2", "fixed_mix", "fixed"),
                        ("switching1", "switching_mix", "switching"),
                        ("switching2", "switching_mix", "switching"),
                        ("switching3", "switching_mix", "switching")):
                    out.append(_spec(chunk, w, src, "primary", pol, kind,
                                     "branch", runs / f"{name}__{src}", base))
                continue
            base_role = "primary" if chunk in ("chunk2", "chunk3") else "aux"
            out.append(_spec(chunk, w, "base", base_role, "damping_fleet",
                             "damping_fleet", "base_like", base, base))
            if chunk in TRUEARMS:
                tr = TRUEARMS[chunk]
                out.append(_spec(chunk, w, "native", "primary",
                                 "native_fleet", "native_fleet", "branch",
                                 tr / f"{name}__native", base,
                                 "J3 true fleet-native arm"))
                out.append(_spec(chunk, w, "pi", "primary", "pi_fleet",
                                 "pi_fleet", "branch",
                                 tr / f"{name}__pi", base,
                                 "J3 true fleet-PI arm"))
            for src, sw in (("swnative", "native"), ("swpi", "pi")):
                out.append(_spec(chunk, w, src, "aux", "switching_mix",
                                 "switching", "branch",
                                 runs / f"{name}__{sw}", base,
                                 "campaign per-CAV switching arm"))
            # WM-N1 label coverage (2026-10-04): the campaign fixed /
            # switching branch arms were never enumerated before; their
            # raw runs sit beside the campaign arms in the same runs root.
            for src in FIXED_ARMS:
                out.append(_spec(chunk, w, src, "primary", "fixed_mix",
                                 "fixed", "branch", runs / f"{name}__{src}",
                                 base, "campaign fixed-mix arm"))
            for src in SWITCH_ARMS:
                out.append(_spec(chunk, w, src, "primary", "switching_mix",
                                 "switching", "branch",
                                 runs / f"{name}__{src}", base,
                                 "campaign switching arm"))
        if chunk == "chunk4":
            # WM-N1 label coverage (2026-10-04): the walls-pilot branch
            # arms; same HOLDOUT/exclusion guard as every other spec.
            wkeep, whold = walls_plan_worlds()
            for w in wkeep:
                name = w["world"]
                if name in excl:
                    raise AssertionError(
                        f"holdout_exclusions world {name} has split "
                        f"{w['split']!r} in the walls pilot: refusing")
                if worlds and name not in worlds:
                    continue
                for arm in WALL_ARMS:
                    out.append(_wall_spec(w, arm))
    return out


# --------------------------------------------------------- open audit ----

class OpenAudit:
    """sys.addaudithook recorder of every path this process opens (audit
    event 'open').  Hooks cannot be removed, so it is a process-global
    singleton that only records while active."""

    _installed = False
    paths: list = []
    active = False

    @classmethod
    def start(cls):
        if not cls._installed:
            import sys as _sys

            def hook(event, args):
                if cls.active and event == "open":
                    try:
                        cls.paths.append(str(args[0]))
                    except Exception:            # noqa: BLE001
                        pass
            _sys.addaudithook(hook)
            cls._installed = True
        cls.active = True

    @classmethod
    def stop(cls):
        cls.active = False

    @classmethod
    def assert_no_holdout(cls, holdout_worlds) -> int:
        """Raise if any recorded opened path names a HOLDOUT world; return
        the number of recorded paths."""
        hw = sorted(set(holdout_worlds))
        bad = sorted({p for p in cls.paths
                      for w in hw if w in p.replace("\\", "/")})
        if bad:
            raise AssertionError(
                f"HOLDOUT world path opened ({len(bad)}): {bad[:5]}")
        return len(cls.paths)

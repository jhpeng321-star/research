"""Stage 1b-prep-4 Part A analysis: legacy-vs-current fidelity.

A3 compares the recovered legacy-driver rows (no_op, w10_t30_h025 on the
long-map IDM scene) with our Arm C and HRULE on the same seeds. A4 intercepts
every TraCI vehicle command both paths issue for one seed (6001) over steps
100-400 and lists every difference in KIND (not just value).

The legacy wall path is reproduced with the recovered driver's actuator
semantics (WallHoldPolicy with trigger_h=0.25/release_h=0.45, band 200-400,
10 m/s, tau 3.0) driven on the harness; the only semantic difference in this
reproduction is the prompt's explicit t = 100 engagement gate (the legacy
driver's own variant had no gate and could arm from the first step).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from experiments import REPO_ROOT

LEGACY_ROOT = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_legacy_probe"
HRULE_ROOT = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep3_hrule_m2200_p20_s6001_6030"
HOLD_ROOT = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030"
SEEDS = tuple(range(6001, 6031))


def part_a3() -> Dict[str, Any]:
    rows = []
    for seed in SEEDS:
        legacy_noop = json.loads(
            (LEGACY_ROOT / "noop_cache" / f"no_op.seed{seed}.json").read_text(encoding="utf-8")
        )
        legacy_wall = json.loads(
            (LEGACY_ROOT / "rows" / f"w10_t30_h025.seed{seed}.json").read_text(encoding="utf-8")
        )
        hold = json.loads(
            (HOLD_ROOT / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        hrule = json.loads(
            (HRULE_ROOT / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        c_inc = bool(hold["arms"]["C"]["health_collapse"]["incidence"])
        our_hrule_inc = bool(hrule["arms"]["HRULE"]["health_collapse"]["incidence"])
        legacy_collapse = bool(legacy_noop["sustained_collapse"])
        legacy_wall_collapse = bool(legacy_wall["sustained_collapse"])
        rows.append(
            {
                "seed": seed,
                "C_incidence": c_inc,
                "our_Hrule_incidence": our_hrule_inc,
                "legacy_noop_sustained_collapse": legacy_collapse,
                "legacy_wall_sustained_collapse": legacy_wall_collapse,
                "noop_agreement": bool(c_inc == legacy_collapse),
                "wall_prevented": bool(c_inc and not legacy_wall_collapse and legacy_noop["sustained_collapse"]),
                "our_hrule_prevented": bool(c_inc and not our_hrule_inc),
                "legacy_noop_loss": legacy_noop["loss"],
                "legacy_noop_tt": legacy_noop["tt"],
                "legacy_wall_loss": legacy_wall["loss"],
                "legacy_wall_tt": legacy_wall["tt"],
                "legacy_wall_first_collapse_step": legacy_wall["first_collapse_step"],
                "legacy_noop_first_collapse_step": legacy_noop["first_collapse_step"],
                "legacy_wall_dv_mean": legacy_wall["dv_mean"],
                "legacy_noop_dv_mean": legacy_noop["dv_mean"],
                "wall_agreement_class": _agreement_class(
                    c_inc, legacy_wall_collapse, our_hrule_inc
                ),
            }
        )
    noop_agreement = sum(1 for r in rows if r["noop_agreement"]) / len(rows)
    class_counts: Dict[str, int] = {}
    for label in ("both_prevent", "only_legacy_prevent", "only_ours_prevent", "neither", "c_no_collapse", "both_collapse"):
        class_counts[label] = sum(1 for r in rows if r["wall_agreement_class"] == label)
    legacy_collapsing = sum(1 for r in rows if r["legacy_noop_sustained_collapse"])
    our_collapsing = sum(1 for r in rows if r["C_incidence"])
    return {
        "schema": "sustained_wall_v1_stage1b_prep4_part_a3/1",
        "n": len(rows),
        "noop_agreement_fraction": noop_agreement,
        "noop_agreement_fraction_explicit": {
            "legacy_noop_collapse_count": legacy_collapsing,
            "our_C_collapse_count": our_collapsing,
            "disagreeing_seeds": [r["seed"] for r in rows if not r["noop_agreement"]],
        },
        "wall_agreement_counts": class_counts,
        "per_seed": rows,
    }


def _agreement_class(c_inc: bool, legacy_wall: bool, ours: bool) -> str:
    if not c_inc:
        return "c_no_collapse"
    prevented_legacy = not legacy_wall
    prevented_ours = not ours
    if prevented_legacy and prevented_ours:
        return "both_prevent"
    if prevented_legacy:
        return "only_legacy_prevent"
    if prevented_ours:
        return "only_ours_prevent"
    return "both_collapse"


def _install_interceptor(log: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wrap traci vehicle write methods with a call log."""
    import traci

    original = {name: getattr(traci.vehicle, name) for name in (
        "setSpeed", "setTau", "setSpeedMode", "setLaneChangeMode",
        "slowDown", "setType",
    )}

    def make(name: str):
        def wrapper(vehicle_id, *args, **kwargs):
            entry: Dict[str, Any] = {
                "step": getattr(_Current, "step", None),
                "cmd": name,
                "vehicle_id": str(vehicle_id),
                "args": [repr(a) for a in args],
            }
            try:
                entry["speedMode"] = int(traci.vehicle.getSpeedMode(vehicle_id))
            except Exception:
                entry["speedMode"] = None
            try:
                entry["tau"] = float(traci.vehicle.getTau(vehicle_id))
            except Exception:
                entry["tau"] = None
            log.append(entry)
            return original[name](vehicle_id, *args, **kwargs)

        return wrapper

    for name in original:
        setattr(traci.vehicle, name, make(name))
    return original


def _restore_interceptor(original: Dict[str, Any]) -> None:
    import traci

    for name, fn in original.items():
        setattr(traci.vehicle, name, fn)


class _Current:
    step: int = -1


def _run_traced(seed: int, policy_factory, run_steps: int = 400) -> List[Dict[str, Any]]:
    from experiments.idm import sustained_wall_v1_contract as C
    from experiments.idm.stage1b_runner_core import replay_branch

    log: List[Dict[str, Any]] = []
    original = _install_interceptor(log)

    class _WithStep:
        def __init__(self, policy):
            self.policy = policy

        def reset(self, env, seed=None):
            self.policy.reset(env, seed)
            _Current.step = 0

        def act(self, obs):
            _Current.step += 1
            return self.policy.act(obs)

        @property
        def last_audit(self):
            return getattr(self.policy, "last_audit", {"wall_active": False})

        def __getattr__(self, name):
            return getattr(self.policy, name)

    trace = replay_branch(
        seed=seed,
        arm="x",
        decision_step=None,
        scenario_file=C.SCENARIO_FILE,
        scenario_root=C.SCENARIO_ROOT_PATH,
        sumo_binary=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe",
        total_steps=run_steps,
        policy_factory=lambda: _WithStep(policy_factory()),
        cav_penetration=C.CAV_PENETRATION,
    )
    _restore_interceptor(original)
    return log


def part_a4(seed: int = 6001) -> Dict[str, Any]:
    from experiments.idm import sustained_wall_v1_contract as C
    from policies.sustained_wall import SustainedWallPolicy
    from policies.inflow_control import WallHoldPolicy
    from policies.metastable_rule import HealthRecordingWrapper

    # Our actuator path: policies/sustained_wall.py SustainedWallPolicy, band
    # 200-400, 10 m/s, tau 3.0, 0.3 clip, engaged for a 180 s pulse.
    our_factory = lambda: SustainedWallPolicy(
        decision_step=200,
        wall_params=dict(C.WALL_PARAMS),
        recovery_h=0.45,
        recovery_persist_s=1.0e6,
        max_wall_duration_s=180.0,
        wall_cap_s=900.0,
        taper_duration_s=5.0,
    )
    # Legacy reproduction: WallHoldPolicy with the trigger itself, driven from
    # the first step (no 100 s gate), exactly the recovered driver's path.
    legacy_factory = lambda: HealthRecordingWrapper(
        WallHoldPolicy(
            d_lo=200.0, d_hi=400.0, wall_speed=10.0, wall_tau=3.0,
            a_max_dec=0.3, trigger_h=0.25, release_h=0.45,
        )
    )
    log_ours = _run_traced(seed, our_factory, run_steps=400)
    log_legacy = _run_traced(seed, legacy_factory, run_steps=400)

    def summarize(log: List[Dict[str, Any]], start: int, end: int) -> Dict[str, Any]:
        window = [e for e in log if e["step"] is not None and start <= e["step"] <= end]
        kinds: Dict[str, int] = {}
        for entry in window:
            kinds[entry["cmd"]] = kinds.get(entry["cmd"], 0) + 1
        first_by_kind: Dict[str, Dict[str, Any]] = {}
        for entry in window:
            first_by_kind.setdefault(entry["cmd"], entry)
        return {
            "total_commands": len(window),
            "kinds": kinds,
            "first_command_by_kind": first_by_kind,
        }

    return {
        "schema": "sustained_wall_v1_stage1b_prep4_part_a4/1",
        "seed": seed,
        "window": [100, 400],
        "ours": summarize(log_ours, 100, 400),
        "legacy": summarize(log_legacy, 100, 400),
        "ours_first_commands": log_ours[:3],
        "legacy_first_commands": log_legacy[:3],
    }


def main() -> None:
    a3 = part_a3()
    print(json.dumps(
        {
            "noop_agreement": a3["noop_agreement_fraction"],
            "legacy_noop_collapse_count": a3["noop_agreement_fraction_explicit"]["legacy_noop_collapse_count"],
            "our_C_collapse_count": a3["noop_agreement_fraction_explicit"]["our_C_collapse_count"],
            "disagreeing_seeds": a3["noop_agreement_fraction_explicit"]["disagreeing_seeds"],
            "wall_agreement_counts": a3["wall_agreement_counts"],
        },
        indent=1,
    ))
    out_dir = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep4_legacy_probe"
    from experiments.io import atomic_write_json as _atomic_json

    _atomic_json(out_dir / "part_a3.json", a3)
    a4 = part_a4(6001)
    _atomic_json(out_dir / "part_a4.json", a4)
    print(json.dumps(a4, indent=1)[:3000])


if __name__ == "__main__":
    main()

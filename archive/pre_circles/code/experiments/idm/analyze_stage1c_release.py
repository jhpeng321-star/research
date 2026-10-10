"""Stage 1c release analysis: inventory, integrity, release timing, axes vs D,
oracle recovery, safety, flow.

Per cell (2200/2400/2600/2700, seeds 7401-7520 at p = 0.20): checks the
five-arm seed JSONs (D/W_ORACLE/W_R1/W_R2/W_HOLD) for inventory and prefix
integrity, replays the online release rules from the recorded 1 Hz rule
inputs to get the exact taper-start step (the seed JSONs do not store the
audit stream), compares each wall arm against D on delay/discharge/collapse,
computes the oracle-recovery fraction (of D-collapsed seeds, the fraction
the oracle arm finishes without a sustained health collapse), and screens
every arm pair for safety excess against the predeclared HeadroomMargins
budgets.

Not a gate: 1c is an information-pass (release rules vs the reference
hold); no go/no-go criterion is attached here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon

from counterfactual.headroom import HeadroomMargins
from experiments import REPO_ROOT
from experiments.idm.stage1c_release_contract import (
    CELLS,
    DECISION_STEP,
    HARD_MAX_HOLD_S,
    MIN_HOLD_S,
    R1_ARRIVAL_RATIO,
    R1_SPEED_FRACTION,
    RELEASE_WINDOW_S,
    SCENARIO_ROOT,
    TAIL_END_S,
    TAIL_DEMAND_VPH,
)
from experiments.io import atomic_write_json as _atomic_json

PROBE = REPO_ROOT / "runs" / "probe"
OUTPUT = PROBE / "stage1c_release"
WALL_ARMS = ("W_ORACLE", "W_R1", "W_R2", "W_HOLD")
FREE_FLOW_FALLBACK_MPS = 31.29


def _load_cell(cell: int):
    dirs = sorted(PROBE.glob(f"stage1c_release_m{cell}_p20_s*/seeds"))
    if not dirs:
        return [], []
    expected = set(CELLS[cell])
    payloads, paths = [], []
    for seed in CELLS[cell]:
        path = dirs[0] / f"seed{seed}.json"
        if not path.exists():
            continue
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
        paths.append(path)
    missing = [s for s in CELLS[cell] if s not in {int(p["seed"]) for p in payloads}]
    return payloads, missing


def free_flow_mps() -> float:
    """The policy's R1 free-flow reference: lane maxspeed of approach lane 0."""
    try:
        sys.path.insert(0, r"C:\Program Files (x86)\Eclipse\Sumo\tools")
        import sumolib

        scene = (SCENARIO_ROOT / "osm.dense.idm_m2200r400.plateau_decline.seed7401.sumocfg")
        if not scene.exists():
            return FREE_FLOW_FALLBACK_MPS
        text = scene.read_text(encoding="utf-8")
        import re

        m = re.search(r'net-file value="([^"]+)"', text)
        net_path = (scene.parent / m.group(1)).resolve() if m else None
        if net_path is None or not str(net_path).endswith((".net.xml", ".xml.gz")):
            return FREE_FLOW_FALLBACK_MPS
        net = sumolib.net.readNet(str(net_path))
        from config.constants import PRE_MERGE_EDGE

        return float(net.getLane(PRE_MERGE_EDGE + "_0").getSpeed())
    except Exception:
        return FREE_FLOW_FALLBACK_MPS


def replay_release_step(
    rule_inputs: Mapping[str, Any],
    *,
    rule: str,
    free_flow: float,
    r2_threshold: Optional[float],
) -> Optional[int]:
    """Exact taper-start step of the ReleaseRuleDampingWallPolicy state machine.

    Wall engages at t = DECISION_STEP + 1 (= 251). The rule may release from
    the first step with elapsed >= min hold (t >= 371) once the condition
    holds for RELEASE_WINDOW_S consecutive steps. The hard max fires first at
    the first absolute step t > hard max (= 901). Returns the taper-start
    step (901 on hard max).
    """
    steps = sorted(int(k) for k in rule_inputs)
    window = 0
    for t in steps:
        if t > HARD_MAX_HOLD_S:
            return t
        if t < DECISION_STEP + 1 + MIN_HOLD_S:
            continue
        now = rule_inputs[str(t)]
        if rule == "R1":
            cond = (
                float(now["arrival_rate_veh_h"])
                <= R1_ARRIVAL_RATIO * float(now["merge_discharge_veh_h"])
                and float(now["approach_mean_speed_mps"]) >= R1_SPEED_FRACTION * free_flow
            )
        else:
            cond = float(now["approach_density_veh_km_lane"]) < float(r2_threshold)
        window = window + 1 if cond else 0
        if window >= RELEASE_WINDOW_S:
            return t
    return None


def _pct(values: Sequence[float], fraction: float) -> Optional[float]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size == 0:
        return None
    return float(np.quantile(arr, fraction))


def _wilcoxon(values: Sequence[float]) -> Optional[float]:
    arr = np.asarray([float(v) for v in values], dtype=float)
    if arr.size < 6 or np.allclose(arr, 0.0):
        return None
    nonzero = arr[np.abs(arr) > 1e-12]
    if nonzero.size < 6:
        return None
    try:
        return float(wilcoxon(nonzero, zero_method="wilcox").pvalue)
    except Exception:
        return None


def _release_timing(payloads: Sequence[Mapping[str, Any]], free_flow: float) -> Dict[str, Any]:
    per_seed: Dict[str, Dict[str, Optional[int]]] = {}
    per_cell: Dict[str, Any] = {}
    for cell in sorted(CELLS):
        cell_rows = [p for p in payloads if int(p["mainline_demand_vph"]) == cell]
        cell_timing: Dict[str, Any] = {}
        for arm in WALL_ARMS:
            steps: list[int] = []
            for p in cell_rows:
                if arm == "W_ORACLE":
                    step = DECISION_STEP + 1 + 950  # cap 1200 -> taper at 1201
                elif arm == "W_HOLD":
                    step = DECISION_STEP + 1 + 650  # cap 900 -> taper at 901
                else:
                    per_seed[str((cell, int(p["seed"])))] = {}
                    step = replay_release_step(
                        p["rule_inputs"][arm],
                        rule=arm[-2:],
                        free_flow=free_flow,
                        r2_threshold=float(p["r2_threshold_veh_km_lane"]),
                    )
                    per_seed[str((cell, int(p["seed"])))][arm] = step
                    if step is None:
                        step = DECISION_STEP + 1 + 650  # hard max -> taper at 901
                steps.append(int(step))
            steps_sorted = sorted(steps)
            n = len(steps_sorted)
            hold = [s - (DECISION_STEP + 1) for s in steps]
            cell_timing[arm] = {
                "n": n,
                "release_step_min": steps_sorted[0] if n else None,
                "release_step_p25": _pct(steps_sorted, 0.25),
                "release_step_median": median(steps_sorted) if n else None,
                "release_step_p75": _pct(steps_sorted, 0.75),
                "release_step_max": steps_sorted[-1] if n else None,
                "hold_duration_s_min": min(hold) if hold else None,
                "hold_duration_s_median": median(hold) if hold else None,
                "hold_duration_s_max": max(hold) if hold else None,
                "fraction_released_by_rule": (
                    float(sum(1 for s in steps if s <= HARD_MAX_HOLD_S)) / n
                    if n
                    else None
                ),
                "fraction_hard_max": (
                    float(sum(1 for s in steps if s > HARD_MAX_HOLD_S)) / n
                    if n
                    else None
                ),
                "release_phase_counts": {
                    "plateau": sum(1 for s in steps if s <= 1200),
                    "decline": sum(1 for s in steps if 1200 < s <= 1500),
                    "tail": sum(1 for s in steps if 1500 < s <= 1800),
                    "post_demand": sum(1 for s in steps if s > 1800),
                },
            }
        per_cell[str(cell)] = cell_timing
    return {"per_cell": per_cell, "per_seed": per_seed}


def _axes_vs_d(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    per_cell: Dict[str, Any] = {}
    ordered = sorted({int(p["mainline_demand_vph"]) for p in payloads})
    for cell in ordered:
        cell_rows = [p for p in payloads if int(p["mainline_demand_vph"]) == cell]
        cell_sum: Dict[str, Any] = {}
        for arm in WALL_ARMS:
            d_delay, d_disc, a_delay, a_disc, hb, ltc = [], [], [], [], [], []
            for p in cell_rows:
                d = p["arms"]["D"]["delay"]
                a = p["arms"][arm]["delay"]
                d_delay.append(float(d["incremental_delay_s"]))
                a_delay.append(float(a["incremental_delay_s"]))
                d_disc.append(float(d["completed_vehicles"]))
                a_disc.append(float(a["completed_vehicles"]))
                out_d = p["arms"]["D"]["outcomes"]
                out_a = p["arms"][arm]["outcomes"]
                hb.append(int(out_a["hard_brake_vehicle_steps"]) - int(out_d["hard_brake_vehicle_steps"]))
                ltc.append(int(out_a["low_ttc_exposure_steps"]) - int(out_d["low_ttc_exposure_steps"]))
            n = len(cell_rows)
            delta_delay = [a - d for d, a in zip(d_delay, a_delay)]
            delta_disc = [a - d for d, a in zip(d_disc, a_disc)]
            cell_sum[arm] = {
                "n": n,
                "median_delay_delayed_s": median(d_delay) if n else None,
                "median_arm_delay_s": median(a_delay) if n else None,
                "median_delta_delay_s": median(delta_delay) if n else None,
                "delta_delay_p25": _pct(delta_delay, 0.25),
                "delta_delay_p75": _pct(delta_delay, 0.75),
                "fraction_delay_loss": float(sum(1 for x in delta_delay if x > 0)) / n if n else None,
                "fraction_delay_gain_gt_200s": (
                    float(sum(1 for x in delta_delay if x < -200.0)) / n if n else None
                ),
                "wilcoxon_p_delta_delay": _wilcoxon(delta_delay),
                "median_discharge_d": median(d_disc) if n else None,
                "median_discharge_arm": median(a_disc) if n else None,
                "median_delta_discharge": median(delta_disc) if n else None,
                "fraction_discharge_gain": (
                    float(sum(1 for x in delta_disc if x > 0)) / n if n else None
                ),
                "fraction_discharge_gain_ge3": (
                    float(sum(1 for x in delta_disc if x >= 3)) / n if n else None
                ),
                "wilcoxon_p_delta_discharge": _wilcoxon(delta_disc),
                "median_hard_brake_excess_steps": median(hb) if n else None,
                "median_low_ttc_excess_steps": median(ltc) if n else None,
            }
        per_cell[str(cell)] = cell_sum
    return per_cell


def _collapse_and_recovery(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    per_cell: Dict[str, Any] = {}
    for cell in sorted({int(p["mainline_demand_vph"]) for p in payloads}):
        cell_rows = [p for p in payloads if int(p["mainline_demand_vph"]) == cell]
        n = len(cell_rows)
        entry: Dict[str, Any] = {
            "n": n,
            "D_health_incidence": sum(
                1 for p in cell_rows if p["arms"]["D"]["health_collapse"]["incidence"]
            ),
            "D_physical_incidence": sum(
                1 for p in cell_rows if p["arms"]["D"]["physical_collapse"]["incidence"]
            ),
            "D_health_first_onset_median": median(
                [p["arms"]["D"]["health_collapse"]["first_onset_step"] or 0
                 for p in cell_rows if p["arms"]["D"]["health_collapse"]["incidence"]]
            ) if any(p["arms"]["D"]["health_collapse"]["incidence"] for p in cell_rows) else None,
            "per_arm": {},
        }
        collapsed_seeds = [
            p for p in cell_rows if p["arms"]["D"]["health_collapse"]["incidence"]
        ]
        n_collapsed = len(collapsed_seeds)
        entry["D_collapsed_seed_count"] = n_collapsed
        d_dur = [float(p["arms"]["D"]["health_collapse"]["total_duration_s"]) for p in cell_rows]
        d_onset = [
            p["arms"]["D"]["health_collapse"]["first_onset_step"] or 0
            for p in cell_rows if p["arms"]["D"]["health_collapse"]["incidence"]
        ]
        entry["D_health_duration_median_s"] = median(d_dur) if d_dur else None
        entry["D_health_onset_median_step"] = median(d_onset) if d_onset else None
        entry["D_health_onset_before_engagement_count"] = sum(
            1 for p in cell_rows
            if p["arms"]["D"]["health_collapse"]["incidence"]
            and (p["arms"]["D"]["health_collapse"]["first_onset_step"] or 0) <= DECISION_STEP
        )
        for arm in WALL_ARMS:
            arm_rows = [p for p in cell_rows if p["arms"][arm]["health_collapse"]["incidence"]]
            arm_dur = [float(p["arms"][arm]["health_collapse"]["total_duration_s"]) for p in cell_rows]
            dur_delta = [
                float(p["arms"][arm]["health_collapse"]["total_duration_s"])
                - float(p["arms"]["D"]["health_collapse"]["total_duration_s"])
                for p in cell_rows
            ]
            entry["per_arm"][arm] = {
                "health_incidence": len(arm_rows),
                "health_incidence_fraction": len(arm_rows) / n if n else None,
                "physical_incidence": sum(
                    1 for p in cell_rows if p["arms"][arm]["physical_collapse"]["incidence"]
                ),
                "recovered_fraction_of_D_collapsed": (
                    sum(1 for p in collapsed_seeds
                        if not p["arms"][arm]["health_collapse"]["incidence"]) / n_collapsed
                    if n_collapsed
                    else None
                ),
                "recovered_count": sum(
                    1 for p in collapsed_seeds
                    if not p["arms"][arm]["health_collapse"]["incidence"]
                ),
                "median_health_duration_s": median(arm_dur) if arm_dur else None,
                "median_delta_health_duration_s": median(dur_delta) if dur_delta else None,
            }
        per_cell[str(cell)] = entry
    return per_cell


def _safety_flags(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    margins = HeadroomMargins()
    per_cell: Dict[str, Any] = {}
    for cell in sorted({int(p["mainline_demand_vph"]) for p in payloads}):
        cell_rows = [p for p in payloads if int(p["mainline_demand_vph"]) == cell]
        per_cell[str(cell)] = _cell_safety(cell_rows, margins)
    return {"per_cell": per_cell, "pooled": _cell_safety(list(payloads), margins)}


def _cell_safety(rows: Sequence[Mapping[str, Any]], margins: HeadroomMargins) -> Dict[str, Any]:
    n = len(rows)
    out: Dict[str, Any] = {"n": n}
    for arm in WALL_ARMS:
        flagged: Dict[str, Any] = {
            "any": 0,
            "hard_brake": 0,
            "low_ttc": 0,
            "collision": 0,
            "teleport": 0,
        }
        for p in rows:
            d = p["arms"]["D"]
            a = p["arms"][arm]
            d_out = d["outcomes"]
            a_out = a["outcomes"]
            d_hb = int(d_out["hard_brake_vehicle_steps"])
            hb_excess = int(a_out["hard_brake_vehicle_steps"]) - d_hb
            hb_budget = max(margins.max_hard_brake_excess, math.ceil(margins.max_hard_brake_excess_fraction * d_hb))
            ltc_excess = int(a_out["low_ttc_exposure_steps"]) - int(d_out["low_ttc_exposure_steps"])
            coll_excess = int(a["events"]["collision_unique_vehicle_count"]) - int(d["events"]["collision_unique_vehicle_count"])
            tel_excess = int(a["events"]["teleport_unique_vehicle_count"]) - int(d["events"]["teleport_unique_vehicle_count"])
            if hb_excess > hb_budget:
                flagged["hard_brake"] += 1
            if ltc_excess > margins.max_low_ttc_excess_steps:
                flagged["low_ttc"] += 1
            if coll_excess > 0:
                flagged["collision"] += 1
            if tel_excess > 0:
                flagged["teleport"] += 1
            if (
                flagged["hard_brake"] > 0
                or flagged["low_ttc"] > 0
                or flagged["collision"] > 0
                or flagged["teleport"] > 0
            ):
                flagged["any"] += 1
        out[arm] = flagged
    return out


def _flow(payloads: Sequence[Mapping[str, Any]], free_flow: float) -> Dict[str, Any]:
    per_cell: Dict[str, Any] = {}
    for cell in sorted({int(p["mainline_demand_vph"]) for p in payloads}):
        cell_rows = [p for p in payloads if int(p["mainline_demand_vph"]) == cell]
        n = len(cell_rows)
        disc = [float(p["arms"]["D"]["delay"]["completed_vehicles"]) for p in cell_rows]
        d_delay = [float(p["arms"]["D"]["delay"]["incremental_delay_s"]) for p in cell_rows]
        r2 = [float(p["r2_threshold_veh_km_lane"]) for p in cell_rows]
        # R1 arrival/discharge at the seed's actual release step (by rule only).
        release_pairs = []
        for p in cell_rows:
            step = replay_release_step(
                p["rule_inputs"]["W_R1"], rule="R1", free_flow=free_flow,
                r2_threshold=float(p["r2_threshold_veh_km_lane"]),
            )
            if step is None or step > HARD_MAX_HOLD_S:
                continue
            now = p["rule_inputs"]["W_R1"][str(step)]
            release_pairs.append((float(now["arrival_rate_veh_h"]), float(now["merge_discharge_veh_h"])))
        per_cell[str(cell)] = {
            "n": n,
            "D_discharge_median": median(disc) if n else None,
            "D_discharge_min": min(disc) if n else None,
            "D_discharge_max": max(disc) if n else None,
            "D_delay_median_s": median(d_delay) if n else None,
            "r2_threshold_median_veh_km_lane": median(r2) if n else None,
            "R1_by_rule_count": len(release_pairs),
            "R1_release_median_arrival_veh_h": median([x for x, _ in release_pairs]) if release_pairs else None,
            "R1_release_median_discharge_veh_h": median([y for _, y in release_pairs]) if release_pairs else None,
        }
    return {"per_cell": per_cell, "tail_veh_per_h": TAIL_DEMAND_VPH, "tail_end_s": TAIL_END_S}


def analyze() -> Dict[str, Any]:
    payloads: list = []
    missing: Dict[str, list[int]] = {}
    for cell in sorted(CELLS):
        rows, cell_missing = _load_cell(cell)
        payloads.extend(rows)
        missing[str(cell)] = cell_missing
    payloads.sort(key=lambda p: (int(p["mainline_demand_vph"]), int(p["seed"])))
    free_flow = free_flow_mps()

    integrity_fail: list[Any] = []
    repeat_fail: list[Any] = []
    divergence: Dict[str, Dict[str, Any]] = {}
    for p in payloads:
        for arm in WALL_ARMS:
            chk = p["integrity"][arm]
            if not chk["prefix_identical_through_t_d"]:
                integrity_fail.append((int(p["seed"]), arm, chk))
        if p.get("repeat_check") is not None:
            if not p["repeat_check"]["exact_identity"]:
                repeat_fail.append((int(p["seed"]), p["repeat_check"]["second_digest"]))
    # First-divergence step distribution: the wall engages at t_d+1 = 251 but
    # only changes TraCI when an in-band CAV exists, so first divergence can
    # legitimately be later than 251 (still prefix-identical through t_d).
    for cell in sorted(CELLS):
        counts: Dict[str, int] = {}
        for p in payloads:
            if int(p["mainline_demand_vph"]) != cell:
                continue
            for arm in WALL_ARMS:
                step = p["integrity"][arm]["first_divergence_step"]
                counts[str(step)] = int(counts.get(str(step), 0)) + 1
        divergence[str(cell)] = counts

    timing = _release_timing(payloads, free_flow)
    axes = _axes_vs_d(payloads)
    recovery = _collapse_and_recovery(payloads)
    safety = _safety_flags(payloads)
    flow = _flow(payloads, free_flow)
    summary: Dict[str, Any] = {
        "schema": "stage1c_release_summary/1",
        "cells": {str(c): list(CELLS[c]) for c in sorted(CELLS)},
        "n_seeds": len(payloads),
        "missing_seeds": missing,
        "free_flow_speed_mps": free_flow,
        "integrity_failures": integrity_fail,
        "repeat_identity_pass_count": sum(
            1 for p in payloads if p.get("repeat_check") and p["repeat_check"]["exact_identity"]
        ),
        "repeat_identity_failures": repeat_fail,
        "first_divergence_step_counts": divergence,
        "release_timing": timing,
        "axes_vs_D": axes,
        "collapse_and_recovery": recovery,
        "safety_flags": safety,
        "flow": flow,
    }
    return summary


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--head", type=int, default=40)
    args = parser.parse_args()
    summary = analyze()
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "summary.json", summary)
    print(f"seeds: {summary['n_seeds']}/{sum(len(v) for v in CELLS.values())}")
    print(f"missing: {summary['missing_seeds']}")
    print(f"integrity failures: {len(summary['integrity_failures'])}")
    print(f"repeat identity: {summary['repeat_identity_pass_count']} passed, "
          f"{len(summary['repeat_identity_failures'])} failed")
    print(f"free-flow speed (R1 reference): {summary['free_flow_speed_mps']:.2f} m/s")
    print()
    for cell, t in summary["release_timing"]["per_cell"].items():
        print(f"== cell {cell}: release timing (taper start, wall engages t=251)")
        for arm in WALL_ARMS:
            ct = t[arm]
            print(
                f"  {arm:8s} n={ct['n']} release med={_fmt(ct['release_step_median'],1)} "
                f"[{_fmt(ct['release_step_min'],1)}, {_fmt(ct['release_step_max'],1)}] "
                f"hold med={_fmt(ct['hold_duration_s_median'],1)} s "
                f"by_rule={_fmt(ct['fraction_released_by_rule'],2)} "
                f"hardmax={_fmt(ct['fraction_hard_max'],2)} phases={ct['release_phase_counts']}"
            )
    print()
    for cell, axes_cell in summary["axes_vs_D"].items():
        print(f"== cell {cell}: axes vs D")
        for arm in WALL_ARMS:
            a = axes_cell[arm]
            print(
                f"  {arm:8s} n={a['n']} dDelta_delay med={_fmt(a['median_delta_delay_s'],1)}s "
                f"(p25 {_fmt(a['delta_delay_p25'],1)}, p75 {_fmt(a['delta_delay_p75'],1)}) "
                f"frac_loss={_fmt(a['fraction_delay_loss'],2)} "
                f"wilcoxon_p={_fmt(a['wilcoxon_p_delta_delay'],4)} "
                f"dDischarge med={_fmt(a['median_delta_discharge'],1)} "
                f"frac_gain={_fmt(a['fraction_discharge_gain'],2)} "
                f"wilk_p={_fmt(a['wilcoxon_p_delta_discharge'],4)} "
                f"med HB excess={_fmt(a['median_hard_brake_excess_steps'],1)} "
                f"med LTC excess={_fmt(a['median_low_ttc_excess_steps'],1)}"
            )
    print()
    for cell, r in summary["collapse_and_recovery"].items():
        print(
            f"== cell {cell}: D health incidence={r['D_health_incidence']}/{r['n']} "
            f"(physical {r['D_physical_incidence']}), collapsed seeds={r['D_collapsed_seed_count']} "
            f"D onset med={_fmt(r['D_health_onset_median_step'],0)} "
            f"onset<=250: {r['D_health_onset_before_engagement_count']} "
            f"D duration med={_fmt(r['D_health_duration_median_s'],1)}s"
        )
        for arm in WALL_ARMS:
            pa = r["per_arm"][arm]
            print(
                f"  {arm:8s} health={pa['health_incidence']}/{r['n']} "
                f"physical={pa['physical_incidence']} "
                f"recovered_of_D_collapsed={_fmt(pa['recovered_fraction_of_D_collapsed'],2)} "
                f"({pa['recovered_count']}/{r['D_collapsed_seed_count'] if r['D_collapsed_seed_count'] else 0}) "
                f"duration med={_fmt(pa['median_health_duration_s'],1)}s "
                f"dDuration med={_fmt(pa['median_delta_health_duration_s'],1)}s"
            )
    print()
    for cell, s in summary["safety_flags"]["per_cell"].items():
        line = "  ".join(f"{arm}=any{s[arm]['any']},hb{s[arm]['hard_brake']},ltc{s[arm]['low_ttc']},col{s[arm]['collision']},tel{s[arm]['teleport']}" for arm in WALL_ARMS)
        print(f"== cell {cell} safety flags (n={s['n']}): {line}")
    pooled = summary["safety_flags"]["pooled"]
    line = "  ".join(f"{arm}=any{pooled[arm]['any']}" for arm in WALL_ARMS)
    print(f"== pooled safety flags (n={pooled['n']}): {line}")
    print()
    for cell, f in summary["flow"]["per_cell"].items():
        print(
            f"== cell {cell}: D discharge med={_fmt(f['D_discharge_median'],1)} "
            f"range [{_fmt(f['D_discharge_min'],1)}, {_fmt(f['D_discharge_max'],1)}] "
            f"D delay med={_fmt(f['D_delay_median_s'],1)}s "
            f"r2 thresh med={_fmt(f['r2_threshold_median_veh_km_lane'],2)} "
            f"R1 by-rule {f['R1_by_rule_count']} arr/disch med={_fmt(f['R1_release_median_arrival_veh_h'],0)}/"
            f"{_fmt(f['R1_release_median_discharge_veh_h'],0)} veh/h"
        )
    print(f"\nsummary written: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()

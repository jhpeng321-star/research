"""Part A: artifact-only hard-brake / low-TTC / onset analysis of Stage 1a.

Reads ``runs/probe/sustained_wall_v1_stage1a_m2200_p20_s6001_6030/`` only and
writes ``runs/probe/sustained_wall_v1_stage1b_prep/part_a.json`` plus a
text summary. No SUMO and no model fitting.

Granularity contract (stated in the results): the three-phase split of
Delta_W hard-brake steps is computed per step (network total) from the stored
per-step series; the location split of of all hard-brake steps is available
only at road granularity (approach edge total) from the stored attribution
marginals because per-step lane positions were not persisted; the full
location x phase x leader-member split is computed on stored event starts
(limited to 200 per branch, truncation flag reported).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from experiments import REPO_ROOT
from experiments.idm.stage1b_common import (
    LOCATION_ORDER,
    event_phase_bucket,
    location_bucket,
    location_from_road_only,
    phase_bucket,
    series_in_window,
)
from experiments.io import atomic_write_json as _atomic_json

STAGE1A_ROOT = (
    REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030"
)
OUTPUT_DIR = REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep"
SEEDS = tuple(range(6001, 6031))


def _phase_series_stats(
    arm: Mapping[str, Any], t_d: int, release_step: int
) -> Dict[str, float]:
    series = arm["series"]
    first = t_d + 1
    return {
        "prefix": series_in_window(series, "hard_brake_count", 0, t_d),
        "during_wall": series_in_window(series, "hard_brake_count", first, release_step - 1),
        "release_60": series_in_window(series, "hard_brake_count", release_step, release_step + 60),
        "after_60": series_in_window(
            series, "hard_brake_count", release_step + 61, 10 ** 6
        ),
    }


def _location_marginals(arm: Mapping[str, Any]) -> Dict[str, float]:
    exposure = arm["outcomes"]["hard_brake_attribution"]["exposure_counts"]["road_id"]
    buckets: Dict[str, float] = {}
    for road_id, count in exposure.items():
        bucket = location_from_road_only(str(road_id))
        buckets[bucket] = buckets.get(bucket, 0.0) + float(count)
    return buckets


def _event_split(
    arm: Mapping[str, Any], release_step: int
) -> Tuple[Dict[Tuple[str, str, str, int, str], int], Dict[str, Any]]:
    events = arm["outcomes"]["hard_brake_attribution"]
    table: Dict[Tuple[str, str, str, int, str], int] = {}
    speeds_upstream: list[float] = []
    for event in events["event_starts"]:
        phase = event_phase_bucket(event, release_step)
        location = location_bucket(str(event.get("road_id")), event.get("lane_position_m"))
        leader_member = bool(event.get("leader_controlled_wall_cav"))
        member = bool(event.get("controlled_wall_cav"))
        vehicle_class = str(event.get("vehicle_class") or "unknown")
        key = (phase, location, vehicle_class, int(leader_member), int(member))
        table[key] = table.get(key, 0) + 1
        if location == "pre_edge_upstream":
            try:
                speeds_upstream.append(float(event.get("speed_mps")))
            except (TypeError, ValueError):
                continue
    summary = {
        "event_start_count": int(events["event_start_count"]),
        "event_sample_truncated": bool(events["event_sample_truncated"]),
        "unique_vehicle_count": int(events["unique_vehicle_count"]),
        "upstream_event_start_speed_mps": {
            "n": len(speeds_upstream),
            "p10": float(np.percentile(speeds_upstream, 10)) if speeds_upstream else None,
            "median": float(np.percentile(speeds_upstream, 50)) if speeds_upstream else None,
            "p90": float(np.percentile(speeds_upstream, 90)) if speeds_upstream else None,
            "frac_below_wall_speed_10": (
                float(np.mean(np.asarray(speeds_upstream) < 10.0)) if speeds_upstream else None
            ),
        },
    }
    return table, summary


def analyze() -> Dict[str, Any]:
    seed_rows = {}
    per_seed = []
    for seed in SEEDS:
        payload = json.loads(
            (STAGE1A_ROOT / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        arms = payload["arms"]
        c = arms["C"]
        w = arms["W"]
        t_d = int(payload["decision_landmark"]["step"])
        release_step = int((payload.get("wall_dose") or {}).get("release_step") or t_d + 1)
        hb_c = float(c["outcomes"]["hard_brake_vehicle_steps"])
        hb_w = float(w["outcomes"]["hard_brake_vehicle_steps"])
        delta = hb_w - hb_c
        phase_c = _phase_series_stats(c, t_d, release_step)
        phase_w = _phase_series_stats(w, t_d, release_step)
        loc_c = _location_marginals(c)
        loc_w = _location_marginals(w)
        events_w, events_w_summary = _event_split(w, release_step)
        events_c, events_c_summary = _event_split(c, release_step)
        h_series = c["series"]["h"]
        row = {
            "seed": seed,
            "t_d": t_d,
            "release_step": release_step,
            "C_hard_brake_steps": hb_c,
            "W_hard_brake_steps": hb_w,
            "Delta_W_hard_brake_steps": delta,
            "ratio_W_over_C": (hb_w / hb_c) if hb_c else None,
            "phase_split_steps": {
                phase: phase_w[phase] - phase_c[phase]
                for phase in ("prefix", "during_wall", "release_60", "after_60")
            },
            "phase_split_steps_components": {"C": phase_c, "W": phase_w},
            "location_marginal_delta_steps": {
                bucket: loc_w.get(bucket, 0.0) - loc_c.get(bucket, 0.0)
                for bucket in sorted(set(loc_w) | set(loc_c))
            },
            "location_marginal_steps": {"C": loc_c, "W": loc_w},
            "exposure_context_counts_delta": {
                key: float(w["outcomes"]["hard_brake_attribution"].get(
                    "exposure_context_counts", {}
                ).get(key, 0))
                - float(c["outcomes"]["hard_brake_attribution"].get(
                    "exposure_context_counts", {}
                ).get(key, 0))
                for key in (
                    "controlled_wall_cav",
                    "leader_controlled_wall_cav",
                    "within_50m_controlled_cav_same_road",
                    "within_50m_controlled_cav_same_lane",
                    "lane_changed",
                )
            },
            "event_start_table_W": {
                "|".join(str(part) for part in key): count for key, count in events_w.items()
            },
            "event_start_table_C": {
                "|".join(str(part) for part in key): count for key, count in events_c.items()
            },
            "event_start_summary_W": events_w_summary,
            "event_start_summary_C": events_c_summary,
            "leader_member_event_starts_W": sum(
                count for key, count in events_w.items() if key[3] == 1
            ),
            "leader_member_event_starts_C": sum(
                count for key, count in events_c.items() if key[3] == 1
            ),
            "h_at_100_150_200_C": {
                "h100": float(h_series[100]),
                "h150": float(h_series[150]),
                "h200": float(h_series[200]),
            },
            "C_onset": c["health_collapse"]["first_onset_step"],
            "onset_minus_t_d": (
                None
                if c["health_collapse"]["first_onset_step"] is None
                else float(c["health_collapse"]["first_onset_step"] - t_d)
            ),
            "Delta_W_low_ttc_steps": (
                float(sum(w["series"]["low_ttc_step_flag"]))
                - float(sum(c["series"]["low_ttc_step_flag"]))
            ),
            "Delta_P_low_ttc_steps": (
                float(sum(arms["P"]["series"]["low_ttc_step_flag"]))
                - float(sum(c["series"]["low_ttc_step_flag"]))
            ),
        }
        per_seed.append(row)
        seed_rows[seed] = row

    deltas = np.asarray([row["Delta_W_hard_brake_steps"] for row in per_seed])
    median_delta = float(np.median(deltas))
    median_seed = min(per_seed, key=lambda r: abs(r["Delta_W_hard_brake_steps"] - median_delta))
    worst_seed = max(per_seed, key=lambda r: r["Delta_W_hard_brake_steps"])

    # A2: low-TTC direction outside the P band.
    p_deltas = np.asarray([row["Delta_P_low_ttc_steps"] for row in per_seed])
    w_deltas = np.asarray([row["Delta_W_low_ttc_steps"] for row in per_seed])
    p5, p95 = np.percentile(p_deltas, 5), np.percentile(p_deltas, 95)
    outside = [row for row in per_seed if row["Delta_W_low_ttc_steps"] < p5 or row["Delta_W_low_ttc_steps"] > p95]
    outside_detail = []
    for row in sorted(outside, key=lambda r: r["seed"]):
        payload = json.loads(
            (STAGE1A_ROOT / "seeds" / f"seed{row['seed']}.json").read_text(encoding="utf-8")
        )
        arms = payload["arms"]
        loc_c = arms["C"]["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
        loc_w = arms["W"]["outcomes"]["low_ttc_attribution"]["exposure_counts"]["road_id"]
        outside_detail.append(
            {
                "seed": row["seed"],
                "Delta_W_low_ttc_steps": row["Delta_W_low_ttc_steps"],
                "direction": "increase" if row["Delta_W_low_ttc_steps"] > 0 else "decrease",
                "location_delta_steps": {
                    str(road): float(loc_w.get(road, 0)) - float(loc_c.get(road, 0))
                    for road in sorted(set(loc_c) | set(loc_w))
                },
            }
        )

    # A3: onset vs decision.
    bucketed = {"onset_before_t_d": [], "onset_t_d_to_60": [], "onset_at_65_plus": [], "no_onset": []}
    for row in per_seed:
        d = row["onset_minus_t_d"]
        if d is None:
            bucketed["no_onset"].append(row["seed"])
        elif d < 0:
            bucketed["onset_before_t_d"].append(row["seed"])
        elif d < 60:
            bucketed["onset_t_d_to_60"].append(row["seed"])
        else:
            bucketed["onset_at_65_plus"].append(row["seed"])

    # A4: trajectory and event listings for the prevented/induced pair.
    a4_cases = {}
    for seed in (6012, 6025):
        payload = json.loads(
            (STAGE1A_ROOT / "seeds" / f"seed{seed}.json").read_text(encoding="utf-8")
        )
        c = payload["arms"]["C"]
        w = payload["arms"]["W"]
        a4_cases[seed] = {
            "t_d": int(payload["decision_landmark"]["step"]),
            "release_step": int((payload.get("wall_dose") or {}).get("release_step")),
            "h_10s": {
                "step": list(c["series"]["step"][100::10]),
                "C": [round(float(v), 3) for v in c["series"]["h"][100::10]],
                "W": [round(float(v), 3) for v in w["series"]["h"][100::10]],
            },
            "wall_members_10s": [int(v) for v in w["series"]["member_count"][100::10]],
            "C_hard_brake_event_start_steps": [
                int(e["relative_step"])
                for e in c["outcomes"]["hard_brake_attribution"]["event_starts"]
            ],
            "W_hard_brake_event_start_steps": [
                int(e["relative_step"])
                for e in w["outcomes"]["hard_brake_attribution"]["event_starts"]
            ],
            "C_low_ttc_event_steps": [
                int(e["relative_step"])
                for e in c["outcomes"]["low_ttc_attribution"]["events"]
            ],
            "W_low_ttc_event_steps": [
                int(e["relative_step"])
                for e in w["outcomes"]["low_ttc_attribution"]["events"]
            ],
        }

    result = {
        "schema": "sustained_wall_v1_stage1b_part_a/1",
        "source": str(STAGE1A_ROOT),
        "median_seed_row": median_seed,
        "worst_seed_row": worst_seed,
        "low_ttc_outside_p_band": {
            "p_band": {"p5": float(p5), "p95": float(p95)},
            "count": len(outside),
            "direction_counts": {
                "increase": sum(1 for r in outside_detail if r["direction"] == "increase"),
                "decrease": sum(1 for r in outside_detail if r["direction"] == "decrease"),
                "zero": sum(1 for r in outside if r["Delta_W_low_ttc_steps"] == 0),
            },
            "seeds": outside_detail,
        },
        "onset_vs_decision": bucketed,
        "h_at_100_150_200": {
            str(seed): row["h_at_100_150_200_C"] for seed, row in seed_rows.items()
        },
        "a4_case_trajectories": a4_cases,
        "per_seed": per_seed,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_DIR / "part_a.json", result)
    return result


def _text_report(result: Dict[str, Any]) -> str:
    lines = []
    for row in result["per_seed"]:
        phase = row["phase_split_steps"]
        loc = row["location_marginal_delta_steps"]
        lines.append(
            f"seed {row['seed']:4d} t_d={row['t_d']:3d} rel={row['release_step']:3d} "
            f"C_hb={row['C_hard_brake_steps']:4.0f} W={row['W_hard_brake_steps']:4.0f} "
            f"dW={row['Delta_W_hard_brake_steps']:+4.0f} ratio={row['ratio_W_over_C']:.2f} | "
            f"during={phase['during_wall']:+4.0f} rel60={phase['release_60']:+4.0f} "
            f"after={phase['after_60']:+4.0f} | "
            + " ".join(f"{k}:{v:+3.0f}" for k, v in loc.items())
        )
    med = result["median_seed_row"]
    worst = result["worst_seed_row"]
    lines.append("")
    lines.append("MEDIAN seed:")
    lines.append(json.dumps(med, indent=1, default=str))
    lines.append("WORST seed:")
    lines.append(json.dumps(worst, indent=1, default=str))
    lines.append("")
    lines.append("A2 low-TTC outside P band:")
    lines.append(json.dumps(result["low_ttc_outside_p_band"], indent=1))
    lines.append("")
    lines.append("A3 onset vs decision:")
    lines.append(json.dumps(result["onset_vs_decision"], indent=1))
    return "\n".join(lines)


def main() -> None:
    result = analyze()
    print(_text_report(result))


if __name__ == "__main__":
    main()

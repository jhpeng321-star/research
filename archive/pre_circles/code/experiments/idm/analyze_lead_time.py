"""Stage 1c-2 Part D: lead-time analysis (D-arm onset minus engagement).

For every W decision with a stored branch this script computes
lead = onset - engagement, where onset is the treatment-blind first step of
the D branch with h < 0.15 for 10 consecutive seconds (uncapped search;
"no onset" if the D branch never qualifies) and engagement is the decision
step t_d of the arm (150/250/350 in p20; 250 in 1c; 250 for the 1c-2 fixed
arms and the onset step itself for WON arms, whose lead is 0 by
construction). Bins: no onset, lead > +100 (engaged early), 0..+100,
-100..0 (engaged after onset began), < -100 (engaged long after onset).

Reported per bin, pooled and by demand bin, and by hold duration where
available: mean/median Delta delay (s/veh, arm - D) and the prevention rate
(among branches whose D sustains a collapse, the fraction whose arm does
not; the collapse-free fraction is reported alongside).

Sources: p20 generation (arms W150/W250/W350, hidden demand), Stage 1c
(W_ORACLE/W_R1/W_R2/W_HOLD, cells 2200-2700), Stage 1c-2 (W250_H300/H450,
WON_H300/H450, W250_R1b, cells 2400/2600/2700 and the 2200 WON run).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional

from experiments import REPO_ROOT
from experiments.idm.rsu_feature_capture import first_onset_step
from experiments.io import atomic_write_json as _atomic_json

PROBE = REPO_ROOT / "runs" / "probe"
OUTPUT = PROBE / "stage1c2_partD"
LEAD_BINS = ("no_onset", "gt_plus100", "0_to_plus100", "minus100_to_0", "lt_minus100")


def _onset_of(h_series: List[float]) -> Optional[int]:
    return first_onset_step([float(v) for v in h_series], sustain_s=10.0,
                            max_step=10 ** 9)


def _lead_bin(lead: Optional[int]) -> str:
    if lead is None:
        return "no_onset"
    if lead > 100:
        return "gt_plus100"
    if lead >= 0:
        return "0_to_plus100"
    if lead >= -100:
        return "minus100_to_0"
    return "lt_minus100"


def _demand_bin_p20(demand: int) -> str:
    return f"{(demand // 100) * 100}"


def _delta_s_per_veh(arm: Dict[str, Any], d_arm: Dict[str, Any]) -> Optional[float]:
    a = arm["delay"]["mean_per_vehicle_delay_s"]
    d = d_arm["delay"]["mean_per_vehicle_delay_s"]
    return float(a) - float(d)


def _prevented(arm: Dict[str, Any], d_arm: Dict[str, Any]) -> bool:
    return bool(d_arm["health_collapse"]["incidence"]) and not bool(
        arm["health_collapse"]["incidence"])


def _bucket(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    deltas = [e["delta_s_per_veh"] for e in entries]
    d_collapsed = [e for e in entries if e["d_collapsed"]]
    return {
        "n": len(entries),
        "mean_delta_delay_s_per_veh": mean(deltas) if deltas else None,
        "median_delta_delay_s_per_veh": median(deltas) if deltas else None,
        "d_collapsed_n": len(d_collapsed),
        "prevention_rate": (sum(1 for e in d_collapsed if e["prevented"]) / len(d_collapsed))
        if d_collapsed else None,
        "collapse_free_rate": (sum(1 for e in entries if not e["collapsed"]) / len(entries))
        if entries else None,
    }


def collect_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []

    # --- p20 generation ---
    gen_dirs = sorted(PROBE.glob("stage1b_generation_p20_s*/seeds"))
    if gen_dirs:
        for path in sorted(gen_dirs[0].glob("seed*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            h = payload["arms"]["D"]["series"]["h"]
            onset = _onset_of(h)
            d_arm = payload["arms"]["D"]
            demand_bin = _demand_bin_p20(int(payload["hidden_demand_vph"]))
            for td in (150, 250, 350):
                arm = payload["arms"].get(f"W{td}")
                if arm is None:
                    continue
                entries.append({
                    "study": "p20", "cell": demand_bin, "arm": f"W{td}",
                    "hold_s": 650, "t_d": td,
                    "lead": None if onset is None else int(onset - td),
                    "lead_bin": _lead_bin(None if onset is None else int(onset - td)),
                    "delta_s_per_veh": _delta_s_per_veh(arm, d_arm),
                    "collapsed": bool(arm["health_collapse"]["incidence"]),
                    "d_collapsed": bool(d_arm["health_collapse"]["incidence"]),
                    "prevented": _prevented(arm, d_arm),
                })

    # --- Stage 1c reference arms ---
    for cell, seed_range in ((2200, range(7401, 7431)), (2400, range(7431, 7461)),
                             (2600, range(7461, 7491)), (2700, range(7491, 7521))):
        dirs = sorted(PROBE.glob(f"stage1c_release_m{cell}_p20_s*/seeds"))
        if not dirs:
            continue
        for seed in seed_range:
            path = dirs[0] / f"seed{seed}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            h = payload["arms"]["D"]["series"]["h"]
            onset = _onset_of(h)
            d_arm = payload["arms"]["D"]
            holds = {"W_ORACLE": 950, "W_R1": 650, "W_R2": 650, "W_HOLD": 650}
            for arm_name, hold in holds.items():
                arm = payload["arms"].get(arm_name)
                if arm is None:
                    continue
                entries.append({
                    "study": "stage1c", "cell": str(cell), "arm": arm_name,
                    "hold_s": hold, "t_d": 250,
                    "lead": None if onset is None else int(onset - 250),
                    "lead_bin": _lead_bin(None if onset is None else int(onset - 250)),
                    "delta_s_per_veh": _delta_s_per_veh(arm, d_arm),
                    "collapsed": bool(arm["health_collapse"]["incidence"]),
                    "d_collapsed": bool(d_arm["health_collapse"]["incidence"]),
                    "prevented": _prevented(arm, d_arm),
                })

    # --- Stage 1c-2 arms ---
    specs = [("stage1c2_arms", cell, range(lo, hi))
             for cell, lo, hi in ((2400, 7431, 7461), (2600, 7461, 7491),
                                  (2700, 7491, 7521))]
    specs.append(("stage1c2_won", 2200, range(7401, 7431)))
    for prefix, cell, seed_range in specs:
        dirs = sorted(PROBE.glob(f"{prefix}_m{cell}_p20_s*/seeds"))
        if not dirs:
            continue
        for seed in seed_range:
            path = dirs[0] / f"seed{seed}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            onset = payload.get("onset_step")
            ref_dirs = sorted(PROBE.glob(f"stage1c_release_m{cell}_p20_s*/seeds"))
            ref = json.loads((ref_dirs[0] / f"seed{seed}.json").read_text(encoding="utf-8"))
            d_arm = ref["arms"]["D"]
            for arm_name, arm in payload.get("arms", {}).items():
                hold = 300 if "H300" in arm_name else (450 if "H450" in arm_name else None)
                engage = int(arm.get("engage_step") or 251)
                effective_onset = onset if onset is not None else None
                lead = None if effective_onset is None else int(effective_onset - engage)
                entries.append({
                    "study": "stage1c2", "cell": str(cell), "arm": arm_name,
                    "hold_s": hold, "t_d": 250,
                    "lead": lead,
                    "lead_bin": _lead_bin(lead),
                    "delta_s_per_veh": _delta_s_per_veh(arm, d_arm),
                    "collapsed": bool(arm["health_collapse"]["incidence"]),
                    "d_collapsed": bool(d_arm["health_collapse"]["incidence"]),
                    "prevented": _prevented(arm, d_arm),
                })
    return entries


def summarize(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, List[Dict[str, Any]]] = {b: [] for b in LEAD_BINS}
    for e in entries:
        buckets[e["lead_bin"]].append(e)
    return {b: _bucket(v) for b, v in buckets.items() if v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    entries = collect_entries()
    summary: Dict[str, Any] = {
        "schema": "stage1c2_partD_lead_time/1",
        "n_entries": len(entries),
        "pooled": summarize(entries),
        "by_demand_bin": {},
        "by_hold": {},
        "by_study_cell": {},
    }
    demand_bins = sorted({e["cell"] for e in entries})
    for bin_name in demand_bins:
        subset = [e for e in entries if e["cell"] == bin_name]
        summary["by_demand_bin"][bin_name] = summarize(subset)
    holds = sorted({e["hold_s"] for e in entries if e["hold_s"] is not None})
    for hold in holds:
        subset = [e for e in entries if e["hold_s"] == hold]
        summary["by_hold"][str(hold)] = summarize(subset)
    study_cells = sorted({(e["study"], e["cell"]) for e in entries})
    for study, cell in study_cells:
        subset = [e for e in entries if e["study"] == study and e["cell"] == cell]
        summary["by_study_cell"][f"{study}:{cell}"] = summarize(subset)
    args.output.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output / "lead_time_summary.json", summary)

    print(f"entries: {len(entries)}")
    for bin_name, block in summary["pooled"].items():
        print(f"  {bin_name:15s} n={block['n']:4d} "
              f"dDelay s/veh mean={block['mean_delta_delay_s_per_veh']:+.3f} "
              f"med={block['median_delta_delay_s_per_veh']:+.3f} "
              f"prevention={block['prevention_rate']} "
              f"collapse_free={block['collapse_free_rate']}")
    print("by demand bin (mean dDelay s/veh per lead bin):")
    for bin_name, blocks in summary["by_demand_bin"].items():
        line = " ".join(
            f"{b}={blocks[b]['mean_delta_delay_s_per_veh']:+.2f}(n={blocks[b]['n']})"
            for b in LEAD_BINS if b in blocks)
        print(f"  {bin_name}: {line}")
    print("by hold (mean dDelay s/veh per lead bin):")
    for hold, blocks in summary["by_hold"].items():
        line = " ".join(
            f"{b}={blocks[b]['mean_delta_delay_s_per_veh']:+.2f}(n={blocks[b]['n']})"
            for b in LEAD_BINS if b in blocks)
        print(f"  hold {hold}: {line}")
    print(f"\nsummary written: {args.output / 'lead_time_summary.json'}")


if __name__ == "__main__":
    main()

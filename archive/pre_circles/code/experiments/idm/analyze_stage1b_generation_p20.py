"""Part B (p20) analysis: inventory, boundary curve, timing-matters, feature variance.

Restricted to seeds 7001-7240 (p = 0.20). Reports: branch inventory and
integrity pass counts, repeat identity, per-100 veh/h bin fractions of
seeds with Delta_delay(W_td - D) < 0 and medians for W150/W250/W350,
the timing-matters fraction per bin (best of three beneficial AND worst
not), and a feature-variance check (per-lane CAV position features and
cav_count_* must vary across seeds at every t_d).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from experiments import REPO_ROOT
from experiments.io import atomic_write_json as _atomic_json

PROBE = REPO_ROOT / "runs" / "probe"
OUTPUT = PROBE / "stage1b_generation"
SEEDS = tuple(range(7001, 7241))


def _load() -> list[Mapping[str, Any]]:
    dirs = sorted(PROBE.glob("stage1b_generation_p20_s*/seeds"))
    if not dirs:
        return []
    payloads = []
    for path in sorted(dirs[0].glob("seed*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def analyze() -> Dict[str, Any]:
    payloads = _load()
    result: Dict[str, Any] = {
        "schema": "stage1b_generation_p20_summary/1",
        "n_seeds": len(payloads),
        "missing_seeds": [s for s in SEEDS if s not in {int(p["seed"]) for p in payloads}],
    }
    integrity_fail = []
    for p in payloads:
        for arm in ("W150", "W250", "W350"):
            chk = p["integrity"][arm]
            if not chk["prefix_identical_through_t_d"] or chk["first_divergence_step"] != int(arm[1:]) + 1:
                integrity_fail.append((int(p["seed"]), arm))
            branch = p["arms"][arm]
            if branch["events"]["collision_unique_vehicle_count"] or branch["events"]["teleport_unique_vehicle_count"]:
                integrity_fail.append((int(p["seed"]), arm, "events"))
            if branch["termination"]["classification"] not in ("natural_clearance", "horizon"):
                integrity_fail.append((int(p["seed"]), arm, "term"))
        if p.get("repeat_check") is not None and not p["repeat_check"]["exact_identity"]:
            integrity_fail.append((int(p["seed"]), "repeat"))
    result["integrity_failures"] = integrity_fail
    result["repeat_identity_count"] = sum(
        1 for p in payloads if p.get("repeat_check") and p["repeat_check"]["exact_identity"]
    )
    result["total_branches"] = len(payloads) * 4

    # Boundary curve per 100 veh/h bin
    bin_delays: Dict[str, Dict[int, Dict[str, Any]]] = {}
    bins = {}
    for p in payloads:
        demand = int(p["hidden_demand_vph"])
        bin_key = demand // 100 * 100
        bins.setdefault(bin_key, []).append(p)
    curve = {}
    for bin_key in sorted(bins):
        rows = bins[bin_key]
        entry: Dict[str, Any] = {"n": len(rows)}
        for arm in ("W150", "W250", "W350"):
            deltas = [float(p["targets"][f"Delta_delay_{arm}_minus_D"]) for p in rows]
            entry[f"{arm}_frac_beneficial"] = float(np.mean(np.asarray(deltas) < 0.0))
            entry[f"{arm}_median_delta_delay"] = float(np.median(deltas))
        # timing-matters
        best = np.min(np.array([
            [float(p["targets"][f"Delta_delay_{arm}_minus_D"]) for arm in ("W150", "W250", "W350")]
            for p in rows
        ]), axis=1)
        worst = np.max(np.array([
            [float(p["targets"][f"Delta_delay_{arm}_minus_D"]) for arm in ("W150", "W250", "W350")]
            for p in rows
        ]), axis=1)
        entry["timing_matters_fraction"] = float(np.mean((best < 0.0) & (worst >= 0.0)))
        curve[str(bin_key)] = entry
    result["boundary_curve"] = curve
    result["timing_matters_overall"] = float(np.mean([
        (min(float(p["targets"][f"Delta_delay_{a}_minus_D"]) for a in ("W150", "W250", "W350")) < 0.0)
        and (max(float(p["targets"][f"Delta_delay_{a}_minus_D"]) for a in ("W150", "W250", "W350")) >= 0.0)
        for p in payloads
    ]))

    # Feature variance check
    variance = {}
    for t_d in ("150", "250", "350"):
        keys = set()
        for p in payloads[:1]:
            keys.update(p["features"].get(t_d, {}).keys())
        constants = {}
        for key in sorted(keys):
            values = [p["features"].get(t_d, {}).get(key) for p in payloads]
            clean = [v for v in values if isinstance(v, (int, float))]
            if not clean:
                continue
            if float(np.ptp(np.asarray(clean))) == 0.0:
                constants[key] = float(clean[0])
        variance[t_d] = {
            "n_features": len(keys),
            "constant_features": constants,
        }
    result["feature_variance"] = variance

    # p30 status
    result["p30"] = {"deferred": True, "registered": True, "seeds": "7241-7360"}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=str(OUTPUT))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = analyze()
    _atomic_json(output / "p20_summary.json", result)
    print(json.dumps(result, indent=1)[:2600])


if __name__ == "__main__":
    main()

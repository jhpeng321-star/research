"""Golden-trace tool for the TraCI snapshot performance refactor.

Runs a fixed, deterministic episode (dense, seed 42, all non_intervention
actions) and records every step's per-CAV observation vector and reward.

Usage:
  conda run -n rl_driving python -u tools/golden_trace.py dump --out PATH
  conda run -n rl_driving python -u tools/golden_trace.py verify PATH

``verify`` replays the identical configuration on the current code and
compares obs/reward sequences element-wise (abs tolerance 1e-5). It exits
non-zero and prints the first mismatches when the traces diverge.

The trace is deterministic: reset(seed=42) fixes CAV selection / HDV type
seeds, SUMO runs with its default seed, and the all-zero (non_intervention)
policy issues the same TraCI commands every run.
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, ".")

from env.core import TrafficShepherdMultiAgentEnv

ENV_CONFIG = {
    "sumo_config": "osm.dense.sumocfg",
    "cav_penetration": 0.1,
    "max_episode_steps": 300,
    # Keep golden obs width stable (no eligibility-mask tail).
    "action_masking": False,
}
SEED = 42
N_STEPS = 100
TOL = 1e-5


def _f(x) -> float:
    """Record float; normalize -0.0 to 0.0 for byte-stable golden files."""
    return float(x) + 0.0


def run_trace() -> dict:
    """Run the fixed episode and return the recorded trace as a dict."""
    env = TrafficShepherdMultiAgentEnv(ENV_CONFIG)
    try:
        obs, info = env.reset(seed=SEED)
        steps = [
            {
                "kind": "reset",
                "obs": {k: [_f(x) for x in v] for k, v in obs.items()},
                "episode_time": _f(info["episode_time"]),
                "warmup_steps": int(info["warmup_steps"]),
            }
        ]
        for _ in range(N_STEPS):
            actions = {cav_id: 0 for cav_id in obs.keys()}
            obs, rewards, terminated, truncated, info = env.step(actions)
            steps.append(
                {
                    "kind": "step",
                    "actions": sorted(actions.keys()),
                    "obs": {k: [_f(x) for x in v] for k, v in obs.items()},
                    "rewards": {k: _f(v) for k, v in rewards.items()},
                    "terminated": {k: bool(v) for k, v in terminated.items()},
                    "truncated": {k: bool(v) for k, v in truncated.items()},
                    "episode_time": _f(info["episode_time"]),
                    "total_vehicles": int(info["total_vehicles"]),
                    "active_cavs": sorted(info["active_cavs"]),
                }
            )
            if terminated.get("__all__") or truncated.get("__all__"):
                break
    finally:
        env.close()
    return {"config": dict(ENV_CONFIG), "seed": SEED, "steps": steps}


def _diff_value(path: str, a, b, mismatches: list) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                mismatches.append(f"{path}.{key}: missing in reference")
            elif key not in b:
                mismatches.append(f"{path}.{key}: missing in candidate")
            else:
                _diff_value(f"{path}.{key}", a[key], b[key], mismatches)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            mismatches.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _diff_value(f"{path}[{i}]", x, y, mismatches)
    elif isinstance(a, float) or isinstance(b, float):
        fa, fb = float(a), float(b)
        if math.isnan(fa) and math.isnan(fb):
            return
        if fa == fb:
            return
        if math.isinf(fa) or math.isinf(fb):
            mismatches.append(f"{path}: {fa!r} != {fb!r}")
            return
        if abs(fa - fb) > TOL:
            mismatches.append(f"{path}: {fa!r} != {fb!r} (|d|={abs(fa - fb):.3g})")
    else:
        if a != b:
            mismatches.append(f"{path}: {a!r} != {b!r}")


def compare_traces(reference: dict, candidate: dict, max_report: int = 20) -> list:
    """Return a list of human-readable mismatches (empty when equivalent)."""
    mismatches: list = []
    ref_steps = reference.get("steps", [])
    cand_steps = candidate.get("steps", [])
    if len(ref_steps) != len(cand_steps):
        mismatches.append(f"steps: length {len(ref_steps)} != {len(cand_steps)}")
    for i, (ra, ca) in enumerate(zip(ref_steps, cand_steps)):
        _diff_value(f"steps[{i}]", ra, ca, mismatches)
        if len(mismatches) >= max_report:
            break
    return mismatches[:max_report]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_dump = sub.add_parser("dump", help="run and write a golden trace")
    p_dump.add_argument("--out", required=True, help="output JSON path")
    p_verify = sub.add_parser("verify", help="replay and compare against a golden trace")
    p_verify.add_argument("golden", help="golden JSON path")
    args = parser.parse_args()

    if args.cmd == "dump":
        trace = run_trace()
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(trace, sort_keys=True))
        print(f"wrote {out} ({len(trace['steps'])} steps)")
        return 0

    reference = json.loads(Path(args.golden).read_text())
    candidate = run_trace()
    mismatches = compare_traces(reference, candidate)
    if mismatches:
        print(f"MISMATCH ({len(mismatches)} shown):")
        for m in mismatches:
            print(f"  {m}")
        return 1
    print(f"OK: candidate trace matches {args.golden} within {TOL:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

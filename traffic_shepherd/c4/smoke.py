"""C4 SUMO smoke: one direct-control episode with every parity check on.

    python -m traffic_shepherd.c4.smoke --out runs/active/C4-SUMO-SMOKE-x \\
        [--seed 4107 --demand 2700 --penetration 0.2 --warmup 300 --horizon 120]

Checks (written to ``<out>/SMOKE.json``): commands applied (issued vs realised CAV
acceleration over each 1 s interval), zero collisions/teleports, RSU 50 m cell counts of
the converted state == SUMO lane-length reference on every step, table sensor ==
``SumoLocalSensor.capture`` (sampled), live conversion == frozen WM-N1 featurizer on the
recorded physics log, the episode windows load in ``wms`` ``TrajectoryDataset``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def commands_applied(ep: dict) -> dict:
    """Issued direct acceleration vs realised (v[t+1 s] - v[t]) of the same CAV."""
    direct = 3
    errs, constrained = [], 0
    for c, t in enumerate(ep["control_time_s"].tolist()):
        i = int(round(t / 0.5))
        if i + 2 >= len(ep["time_s"]):
            break
        for slot in np.flatnonzero(ep["controller"][c] == direct).tolist():
            vid = ep["vid_index"][i, slot]
            if vid < 0 or ep["vid_index"][i + 2, slot] != vid:
                continue
            realised = float(ep["v_mps"][i + 2, slot] - ep["v_mps"][i, slot])
            issued = float(ep["issued_acceleration_mps2"][c, slot])
            err = realised - issued
            errs.append(err)
            if err < -0.05:
                constrained += 1          # SUMO safe-speed kept the CAV slower
    errs = np.asarray(errs)
    if not len(errs):
        return {"n": 0}
    return {"n": int(len(errs)), "median_abs_err_mps2": float(np.median(np.abs(errs))),
            "frac_within_0_05": float(np.mean(np.abs(errs) <= 0.05)),
            "frac_slower_than_issued": float(constrained / len(errs)),
            "frac_faster_than_issued_by_0_05": float(np.mean(errs > 0.05))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m traffic_shepherd.c4.smoke")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=4107)
    ap.add_argument("--demand", type=int, default=2700)
    ap.add_argument("--penetration", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=120)
    args = ap.parse_args(argv)
    from .baselines import RandomAccelPolicy, run_policy_episode
    from .env import CavSumoEnv
    from .parity import verify_episode
    from .recorder import episode_windows, load_episode, write_trajectory_npz
    from .wms_glue import trajectory_dataset

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = CavSumoEnv(check_parity=True, sensor_check_every_s=5)
    detections = []

    def on_step(state, info):
        detections.append(len(info["rsu_detections"]))

    summary = run_policy_episode(env, RandomAccelPolicy(args.seed), seed=args.seed,
                                 demand_vph=args.demand, penetration=args.penetration,
                                 warmup_s=args.warmup, horizon_s=args.horizon,
                                 episode_dir=out / "episode", on_step=on_step)
    ep = load_episode(out / "episode")
    parity = verify_episode(out / "episode")
    applied = commands_applied(ep)
    windows = episode_windows(ep, window_s=30, stride_s=30, history_s=60)
    path = write_trajectory_npz(out / "trajectory" / "trajectory.npz", windows)
    ds = trajectory_dataset(path, history_s=60)
    result = {
        "schema": "c4_sumo_smoke/1", "args": vars(args),
        "collisions": summary["collisions"], "teleports": summary["teleports"],
        "command_counts": summary["command_counts"],
        "fallback_counts": summary["fallback_counts"],
        "commands_applied": applied,
        "rsu_parity_failures": summary["rsu_parity_failures"],
        "sensor_parity_checks": summary["sensor_parity_checks"],
        "sensor_parity_mismatches": summary["sensor_parity_mismatches"],
        "frozen_featurizer_parity": parity,
        "rsu_detections_per_step": {"min": min(detections), "max": max(detections),
                                    "mean": float(np.mean(detections))},
        "max_present_vehicles": summary["max_present_vehicles"],
        "dropped_vehicle_steps": summary["dropped_vehicle_steps"],
        "trajectory_windows": int(len(ds)),
        "window_metrics": summary["window"],
        "sim_s_per_wall_s": summary["sim_s_per_wall_s"],
        "window_sim_s_per_wall_s_with_parity_checks": summary["window_sim_s_per_wall_s"],
    }
    result["pass"] = bool(result["collisions"] == 0 and result["teleports"] == 0
                          and result["rsu_parity_failures"] == 0
                          and result["sensor_parity_mismatches"] == 0 and parity["ok"]
                          and applied.get("n", 0) > 0 and len(ds) > 0)
    (out / "SMOKE.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

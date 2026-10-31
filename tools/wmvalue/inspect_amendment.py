"""Bounded, read-only amendment preflight. Never load hidden archive members."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--corpus", choices=("wmdata", "wmdata2"), required=True)
    parser.add_argument("--arm", default="native", choices=("native", "pi"))
    args = parser.parse_args()
    root = Path(args.root)
    seed = 969400 if args.corpus == "wmdata" else 969550
    job = root / "jobs" / f"s{seed}_{args.arm}"
    with np.load(job / "deploy/episode/episode.npz", allow_pickle=False) as z:
        keys = ("time_s", "s_m", "lane", "v_mps", "a_mps2", "present",
                "is_cav", "vehicle_ids", "vid_index", "control_time_s",
                "issued_acceleration_mps2", "issued_target_speed_mps")
        print(json.dumps({k: dict(shape=z[k].shape, dtype=str(z[k].dtype)) for k in keys}))
        for key in ("vehicle_ids", "control_time_s", "issued_acceleration_mps2",
                    "issued_target_speed_mps"):
            a = z[key]
            print(key, a.reshape(-1)[:12].tolist())
        ticks, slots = np.where(z["present"] & z["is_cav"] & (z["lane"] < 0))
        print("invalid own lane", [(float(z["time_s"][t]), int(j), float(z["s_m"][t, j]),
                                    float(z["v_mps"][t, j])) for t, j in zip(ticks[:8], slots[:8])])
    print("objective", (job / "objective_600.json").read_text())
    design = root / "design.json"
    if design.is_file():
        d = json.loads(design.read_text())
        print("design keys", list(d))
        print("first world", d["worlds"][0])


if __name__ == "__main__":
    main()

"""Read explicit FIT recording schemas, never launch a simulator or read truth.

Run on the data host with its existing numpy environment. Output is exclusive-
create and contains schema evidence only, not reserved-world outcomes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "hetdata_truth", "vehicle_params", "driver_params", "vtype")
ARMS = ("native", "ctrl0", "mappo", "pi", "circles")


def safe(path):
    p = Path(path)
    for value in (str(p).lower(), str(p.resolve()).lower()):
        if any(x in value for x in FORBIDDEN):
            raise ValueError("prohibited path")
    return p


def allowed_seed(seed, role):
    seed = int(seed)
    fit = 969400 <= seed <= 969431 or 969550 <= seed <= 969599
    test = 969432 <= seed <= 969439
    if role not in ("fit", "test") or not (fit if role == "fit" else test):
        raise ValueError("seed outside registered role")
    return seed


def digest(path):
    h = hashlib.sha256()
    with safe(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def audit(root, corpus):
    root = safe(root)
    if corpus not in ("wmdata", "wmdata2"):
        raise ValueError("unknown corpus")
    seeds = range(969400, 969432) if corpus == "wmdata" else range(969550, 969600)
    arms = ARMS if corpus == "wmdata" else (*ARMS, "generic_random")
    records = []
    for seed in seeds:
        allowed_seed(seed, "fit")
        for arm in arms:
            job = safe(root / "jobs" / f"s{seed}_{arm}")
            ep = safe(job / "deploy/episode/episode.npz")
            row = dict(seed=seed, arm=arm, episode=str(ep), exists=ep.is_file())
            if ep.is_file():
                # NPZ member names inspected; only clock arrays loaded here.
                with np.load(ep, allow_pickle=False) as z:
                    keys = set(z.files)
                    times = np.asarray(z["time_s"])
                    row.update(keys=sorted(keys), time_start=float(times[0]),
                               time_end=float(times[-1]), frames=len(times),
                               uniform_half_seconds=bool(np.allclose(np.diff(times), .5)))
                    row["recorded_queue"] = "entry_backlog" in keys
                    row["recorded_sensor_mask"] = "sensor_mask" in keys
                    row["recorded_request_counts"] = "request_counter" in keys
                row["episode_sha256"] = digest(ep)
            row["real_objective600_exists"] = safe(job / "objective_600.json").is_file()
            # Fixed filenames only; no traversal or opening of scenario/truth siblings.
            row["queue_sidecar_exists"] = safe(job / "observations.npz").is_file()
            row["observation_contract_exists"] = safe(job / "observation_contract.json").is_file()
            traj = safe(job / "traj_base.npz")
            if traj.is_file():
                with np.load(traj, allow_pickle=False) as z:
                    row["trajectory_keys"] = sorted(z.files)
            row["ready"] = bool(row.get("time_end", 0) >= 900
                                and row.get("recorded_queue")
                                and row.get("recorded_sensor_mask")
                                and row.get("recorded_request_counts")
                                and row["real_objective600_exists"])
            records.append(row)
    return dict(schema="wmvalue_audit/1", corpus=corpus, root=str(root), role="fit",
                code_sha256=digest(__file__), episodes=len(records),
                ready=sum(r["ready"] for r in records), records=records,
                verdict="READY" if all(r["ready"] for r in records) else "INCOMPLETE")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--corpus", choices=("wmdata", "wmdata2"), required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    if hasattr(os, "nice"):
        os.nice(19)
    result = audit(a.root, a.corpus)
    with safe(a.out).open("x") as f:
        json.dump(result, f, indent=1, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()

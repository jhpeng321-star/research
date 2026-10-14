"""Small SUMO smoke of the HDV driver population switch (owner 2026-10-14).

One process = one (population, policy); episodes run sequentially through
``traffic_shepherd.c4.collect.run_job`` (the collector's own worker path).  Per episode it reports
collisions/teleports, window mean speed and jam fraction (vehicle-steps with v < 8 m/s, from
``run/physics_steps.jsonl.gz``), mainline-exit crossings (throughput past the merge), window
source wait and CPU seconds (python + SUMO child) per simulated second.

    python tools/c4_population_smoke.py --population textbook --policy native --out DIR
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = ((4107, 2700, 0.2), (4108, 2700, 0.2), (4109, 2400, 0.4), (4110, 2400, 0.4))
JAM_MPS = 8.0


def physics_metrics(episode: Path, t0: float, t1: float) -> dict:
    n = jam = 0
    speed_sum = 0.0
    hdv_n = hdv_sum = 0
    with gzip.open(episode / "run" / "physics_steps.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if not t0 < row["time_s"] <= t1:
                continue
            cavs = set(row.get("cav_ids", ()))
            for vid, veh in row["vehicles"].items():
                v = float(veh["speed_mps"])
                n += 1
                speed_sum += v
                jam += v < JAM_MPS
                if vid not in cavs:
                    hdv_n += 1
                    hdv_sum += v
    return dict(vehicle_steps=n, mean_speed_mps=speed_sum / n if n else None,
                hdv_mean_speed_mps=hdv_sum / hdv_n if hdv_n else None,
                jam_fraction=jam / n if n else None)


class SumoCpu(threading.Thread):
    """Samples the CPU time of this process's SUMO children every 0.5 s (SUMO exits at episode
    end, so the last sample per pid is used; error <= 0.5 s of SUMO CPU per episode)."""

    def __init__(self):
        super().__init__(daemon=True)
        import psutil
        self.psutil, self.me, self.seen, self.lock = psutil, psutil.Process(), {}, threading.Lock()

    def run(self):
        while True:
            for child in self.me.children(recursive=True):
                try:
                    if "sumo" in child.name().lower():
                        t = child.cpu_times()
                        with self.lock:
                            self.seen[child.pid] = t.user + t.system
                except self.psutil.Error:
                    pass
            time.sleep(0.5)

    def take(self) -> float:
        with self.lock:
            total, self.seen = sum(self.seen.values()), {}
        return total


def main() -> int:

    from traffic_shepherd.c4 import collect

    ap = argparse.ArgumentParser()
    ap.add_argument("--population", choices=("legacy3", "textbook"), required=True)
    ap.add_argument("--policy", choices=("native", "generic_random"), required=True)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out) / f"{args.population}__{args.policy}"
    out.mkdir(parents=True, exist_ok=False)
    opts = dict(policy=args.policy, policy_loader="wms.control.policy:load_policy_fn", k=192,
                warmup=args.warmup, horizon=args.horizon, check_parity=False, no_ttc=False,
                no_physics_log=False, branch_cadence=0, branch_arms=4, branch_horizon=60,
                no_branch_native=False, branch_repeat_check=0, gr_mode="speed",
                cav_vtype="normal_car", population=args.population)
    sampler = SumoCpu()
    sampler.start()
    results = []
    for seed, demand, pen in CASES:
        jid = collect.job_id(seed, demand, pen)
        job = dict(job_id=jid, seed=seed, demand=demand, penetration=pen,
                   episode_dir=str(out / "episodes" / jid), opts=opts, attempt=1)
        py0 = time.process_time()
        wall0 = time.monotonic()
        record = collect.run_job(job)
        py_cpu = time.process_time() - py0
        time.sleep(0.6)
        sumo_cpu = sampler.take()
        rec = dict(population=args.population, policy=args.policy, seed=seed,
                   demand_vph=demand, penetration=pen, status=record["status"],
                   error=record.get("error"))
        if record["status"] == "complete":
            summary = json.loads((Path(job["episode_dir"]) / "summary.json")
                                 .read_text(encoding="utf-8"))
            window = summary["window"]
            sim = float(summary["sim_seconds"])
            rec.update(collisions=summary["collisions"], teleports=summary["teleports"],
                       mainline_exit_crossings=window.get("mainline_exit_crossings"),
                       ramp_entry_crossings=window.get("ramp_entry_crossings"),
                       mainline_entry_crossings=window.get("mainline_entry_crossings"),
                       source_wait_vehicle_s=window.get("source_wait_vehicle_seconds"),
                       system_vehicle_s=window.get("system_vehicle_seconds"),
                       arrivals=window.get("arrivals"),
                       min_closing_ttc_s=summary.get("minimum_closing_ttc_s"),
                       wall_s=time.monotonic() - wall0, python_cpu_s=py_cpu,
                       sumo_cpu_s=sumo_cpu, cpu_s_per_sim_s=(py_cpu + sumo_cpu) / sim,
                       sim_s=sim,
                       **physics_metrics(Path(job["episode_dir"]), float(args.warmup),
                                         float(args.warmup + args.horizon)))
        results.append(rec)
        print(json.dumps(rec), flush=True)
    if collect._ENV is not None:
        collect._ENV.close()
    (out / "SMOKE_RESULTS.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(main())

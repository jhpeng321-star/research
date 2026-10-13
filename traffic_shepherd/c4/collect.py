"""Parallel, resumable C4 SUMO episode collector.

    python -m traffic_shepherd.c4.collect --policy random \\
        --penetrations 0.05,0.1,0.2,0.4 --demands 2400,2600,2700 --seeds 4107-4110 \\
        --warmup 300 --horizon 600 --workers 6 --out runs/active/C4-COLLECT-x

``--policy``: ``native|damping|pi|wall`` (frozen baselines), ``random`` (seeded smooth
random direct accelerations), ``zero``, ``module:factory`` or a checkpoint path loaded with
``--policy-loader`` (default ``wms.control.policy:load_policy_fn``).  One SUMO per worker
process (spawn); every episode writes its own directory
``<out>/episodes/<policy_tag>/<job>/`` (see :mod:`traffic_shepherd.c4.recorder`).
Resumable: complete episodes are skipped; an incomplete directory is preserved as
``<job>.incomplete-<n>`` and the job is re-run (the attempt count is recorded).
``<out>/INDEX.json`` lists every job; ``<out>/COLLECT_SUMMARY.json`` reports the
measured SUMO speed (simulated seconds per wall second, per worker and aggregate).
"""
from __future__ import annotations

import os as _os

# One SUMO + one policy per worker process: keep BLAS/OpenMP single-threaded, otherwise every worker
# spawns one thread per visible core (127 on the AutoDL vGPU box with a 16-core quota) and the box
# thrashes (2026-10-13: >18 min/episode instead of ~25 s).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

import argparse
import json
import multiprocessing as mp
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

FORBIDDEN = ("holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "sealed")

_ENV = None
_POLICIES: dict = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_id(seed: int, demand: int, penetration: float) -> str:
    return f"s{seed}_m{demand}_p{int(round(penetration * 1000)):04d}"


def policy_tag(spec: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(spec).stem if os.sep in spec or "/" in spec
                 else spec)
    return tag[:60] or "policy"


def _episode_complete(path: Path) -> bool:
    manifest = path / "manifest.json"
    if not manifest.is_file() or not (path / "summary.json").is_file():
        return False
    try:
        return json.loads(manifest.read_text(encoding="utf-8")).get("status") == "complete"
    except (OSError, ValueError):
        return False


def _worker_env(opts: dict):
    global _ENV
    if _ENV is None:
        from traffic_shepherd.c4.env import CavSumoEnv
        _ENV = CavSumoEnv(k_slots=opts["k"], check_parity=opts["check_parity"],
                          ttc=not opts["no_ttc"], physics_log=not opts["no_physics_log"])
    return _ENV


def _policy_for(opts: dict, seed: int, demand: int, penetration: float):
    from traffic_shepherd.c4.baselines import BASELINES, RandomAccelPolicy, make_policy
    spec = opts["policy"]
    if spec in BASELINES:
        return spec
    if spec == "random":
        return RandomAccelPolicy(seed * 1_000_003 + demand * 101 + int(penetration * 1000))
    if spec not in _POLICIES:
        _POLICIES[spec] = make_policy(spec, seed=seed, loader=opts["policy_loader"])
    return _POLICIES[spec]


def run_job(job: dict) -> dict:
    """Worker entry: one episode (never raises; failures are returned as records)."""
    opts = job["opts"]
    out = Path(job["episode_dir"])
    started = time.monotonic()
    record = dict(job_id=job["job_id"], seed=job["seed"], demand_vph=job["demand"],
                  penetration=job["penetration"], episode_dir=str(out), pid=os.getpid(),
                  attempt=job["attempt"], started_utc=_now())
    try:
        from traffic_shepherd.c4.baselines import run_policy_episode
        env = _worker_env(opts)
        policy = _policy_for(opts, job["seed"], job["demand"], job["penetration"])
        summary = run_policy_episode(
            env, policy, seed=job["seed"], demand_vph=job["demand"],
            penetration=job["penetration"], warmup_s=opts["warmup"], horizon_s=opts["horizon"],
            episode_dir=out, meta=dict(policy=opts["policy"], collector_job=job["job_id"],
                                       attempt=job["attempt"]))
        record.update(status="complete", sim_seconds=summary["sim_seconds"],
                      wall_seconds=time.monotonic() - started,
                      episode_sim_s_per_wall_s=summary["sim_s_per_wall_s"],
                      window_sim_s_per_wall_s=summary["window_sim_s_per_wall_s"],
                      collisions=summary["collisions"], teleports=summary["teleports"],
                      dropped_cav_steps=summary["dropped_cav_steps"])
    except Exception as exc:  # noqa: BLE001 - recorded, never silently retried
        try:
            if _ENV is not None:
                _ENV.close()
        except Exception:  # noqa: BLE001
            pass
        record.update(status="failed", error=repr(exc), traceback=traceback.format_exc(),
                      wall_seconds=time.monotonic() - started)
    record["finished_utc"] = _now()
    return record


def plan_jobs(args) -> list[dict]:
    from traffic_shepherd.c4.seeds import check_case, parse_seeds
    out = Path(args.out).resolve()
    root = out / "episodes" / policy_tag(args.policy)
    seeds = parse_seeds(args.seeds)
    demands = [int(x) for x in args.demands.split(",") if x.strip()]
    pens = [float(x) for x in args.penetrations.split(",") if x.strip()]
    opts = dict(policy=args.policy, policy_loader=args.policy_loader, k=args.k,
                warmup=args.warmup, horizon=args.horizon, check_parity=args.check_parity,
                no_ttc=args.no_ttc, no_physics_log=args.no_physics_log)
    jobs = []
    for seed in seeds:
        for demand in demands:
            check_case(seed, demand)              # fail before any SUMO
            for pen in pens:
                jid = job_id(seed, demand, pen)
                jobs.append(dict(job_id=jid, seed=seed, demand=demand, penetration=pen,
                                 episode_dir=str(root / jid), opts=opts, attempt=1))
    return jobs


def prepare(jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (todo, already complete); preserve incomplete directories."""
    todo, done = [], []
    for job in jobs:
        path = Path(job["episode_dir"])
        if path.exists():
            if _episode_complete(path):
                done.append(job)
                continue
            n = 1
            while path.with_name(f"{path.name}.incomplete-{n}").exists():
                n += 1
            path.rename(path.with_name(f"{path.name}.incomplete-{n}"))
            job["attempt"] = n + 1
        path.parent.mkdir(parents=True, exist_ok=True)
        todo.append(job)
    return todo, done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m traffic_shepherd.c4.collect",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True)
    ap.add_argument("--policy-loader", default="wms.control.policy:load_policy_fn")
    ap.add_argument("--penetrations", default="0.05,0.1,0.2,0.4")
    ap.add_argument("--demands", default="2400,2600,2700")
    ap.add_argument("--seeds", required=True, help="e.g. 4107-4112,4120 (exposed 4107-4148)")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-parity", action="store_true",
                    help="per-step RSU-cell and sensor parity checks (slower)")
    ap.add_argument("--no-ttc", action="store_true")
    ap.add_argument("--no-physics-log", action="store_true",
                    help="skip run/physics_steps.jsonl.gz (episode.npz is always written)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if any(tok in str(args.out).lower() for tok in FORBIDDEN):
        raise SystemExit("refusing an output path with a restricted-split token")
    jobs = plan_jobs(args)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    todo, done = prepare(jobs) if not args.dry_run else (jobs, [])
    print(f"[collect] {len(jobs)} jobs: {len(done)} complete, {len(todo)} to run, "
          f"workers={args.workers}", flush=True)
    if args.dry_run:
        return 0
    started = time.monotonic()
    records = []
    log = (out / "collect_log.jsonl").open("a", encoding="utf-8")
    try:
        if args.workers <= 1:
            results = map(run_job, todo)
            for rec in results:
                records.append(rec)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(f"[collect] {rec['job_id']} {rec['status']} "
                      f"{rec.get('episode_sim_s_per_wall_s')}", flush=True)
        else:
            with mp.get_context("spawn").Pool(args.workers) as pool:
                for rec in pool.imap_unordered(run_job, todo):
                    records.append(rec)
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    print(f"[collect] {rec['job_id']} {rec['status']} "
                          f"{rec.get('episode_sim_s_per_wall_s')}", flush=True)
    finally:
        log.close()
    wall = time.monotonic() - started
    ok = [r for r in records if r["status"] == "complete"]
    sim = sum(r["sim_seconds"] for r in ok)
    per_worker = [r["episode_sim_s_per_wall_s"] for r in ok]
    index = {"schema": "c4_collect_index/1", "policy": args.policy, "updated_utc": _now(),
             "jobs": [{k: v for k, v in j.items() if k != "opts"} for j in jobs],
             "run_records": records,
             "already_complete": [j["job_id"] for j in done]}
    (out / "INDEX.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    summary = {"schema": "c4_collect_summary/1", "policy": args.policy,
               "workers": args.workers, "warmup_s": args.warmup, "horizon_s": args.horizon,
               "episodes_run": len(records), "episodes_complete": len(ok),
               "episodes_failed": len(records) - len(ok), "wall_seconds": wall,
               "simulated_seconds": sim,
               "aggregate_sim_s_per_wall_s": sim / wall if wall > 0 else None,
               "per_worker_episode_sim_s_per_wall_s_mean": (sum(per_worker) / len(per_worker)
                                                            if per_worker else None),
               "per_worker_window_sim_s_per_wall_s_mean": (
                   sum(r["window_sim_s_per_wall_s"] for r in ok) / len(ok) if ok else None),
               "collisions": sum(r["collisions"] for r in ok),
               "teleports": sum(r["teleports"] for r in ok),
               "host_cpus": os.cpu_count(), "finished_utc": _now()}
    (out / "COLLECT_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    return 0 if len(ok) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

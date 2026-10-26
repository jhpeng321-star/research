"""Parallel, resumable C4 SUMO episode collector.

    python -m traffic_shepherd.c4.collect --policy random \\
        --penetrations 0.05,0.1,0.2,0.4 --demands 2400,2600,2700 --seeds 4107-4110 \\
        --warmup 300 --horizon 600 --workers 6 --out runs/active/C4-COLLECT-x

``--policy``: ``native|damping|pi|wall`` (frozen baselines), ``random`` (seeded smooth
random direct accelerations), ``generic_random`` (undirected random per-CAV commands + lane
intents, :mod:`traffic_shepherd.c4.genrand`; optional paired branches with ``--branch-cadence``),
``zero``, ``module:factory`` or a checkpoint path loaded with
``--policy-loader`` (default ``wms.control.policy:load_policy_fn``).  ``--policy-pairs`` (LOOP14,
:mod:`traffic_shepherd.c4.policy_pairs`): a native base per world and, from every saved state,
the learned policy arm plus the all-native reference arm (registered seeds only).  One SUMO per worker
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
        extra = None
        if opts["policy"] == "generic_random" or opts.get("policy_pairs"):
            from traffic_shepherd.c4.branching import SAVE_STATE_CLI
            extra = SAVE_STATE_CLI          # same launch command with and without branches
        _ENV = CavSumoEnv(k_slots=opts["k"], check_parity=opts["check_parity"],
                          ttc=not opts["no_ttc"], physics_log=not opts["no_physics_log"],
                          extra_sumo_cli=extra,
                          cav_vtype=None if opts.get("cav_vtype", "normal_car") == "legacy"
                          else opts.get("cav_vtype", "normal_car"),
                          population=opts.get("population", "legacy3"))
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
        meta = dict(policy=opts["policy"], collector_job=job["job_id"], attempt=job["attempt"])
        if opts.get("policy_pairs"):
            from traffic_shepherd.c4.policy_pairs import (decision_times, load_pair_policy_factory,
                                                          run_policy_pair_job)
            extras = None
            if opts.get("sumocredit"):              # S-SUMOCREDIT extras (absent = unchanged)
                from traffic_shepherd.c4.sumocredit import PairExtras
                extras = PairExtras(**opts["sumocredit"])
            key = ("pairs", opts["policy"]) if extras is None else ("pairs", opts["policy"], "wm")
            if key not in _POLICIES:
                _POLICIES[key] = load_pair_policy_factory(
                    opts["policy"], opts.get("settings_wm_ckpt"), opts.get("settings_wm_config"),
                    **({} if extras is None else {"need_wm": extras.needs_wm}))
            times = decision_times(opts["warmup"], opts["branch_first"] or opts["warmup"] + 60,
                                   opts["branch_cadence"], opts["branch_points"])
            result = run_policy_pair_job(
                env, policy_factory=_POLICIES[key], seed=job["seed"], demand_vph=job["demand"],
                penetration=job["penetration"], warmup_s=opts["warmup"], times=times,
                branch_h=opts["branch_horizon"], episode_dir=out,
                branch_root=Path(job["branch_dir"]), meta=meta, policy_ckpt=opts["policy"],
                **({} if extras is None else {"extras": extras}))
            summary = result["summary"]
            b = result["branches"]
            record.update(n_branches=b["n_branches"], branch_collisions=b["collisions"],
                          branch_teleports=b["teleports"], branch_wall_s=b["branch_wall_s"],
                          base_wall_s=b["base_wall_s"], pair_dJ=b["pair_dJ"],
                          pair_sim_seconds=b["sim_s"])
            if "sumocredit" in b:
                record.update(sumocredit=b["sumocredit"])
        elif opts["policy"] == "generic_random":
            from traffic_shepherd.c4.genrand import run_genrand_job
            branch_root = (Path(job["branch_dir"]) if opts.get("branch_cadence", 0) > 0
                           else None)
            result = run_genrand_job(
                env, seed=job["seed"], demand_vph=job["demand"], penetration=job["penetration"],
                warmup_s=opts["warmup"], horizon_s=opts["horizon"], episode_dir=out,
                branch_root=branch_root, cadence_s=opts.get("branch_cadence", 0),
                n_arms=opts.get("branch_arms", 4), branch_h=opts.get("branch_horizon", 60),
                native_arm=not opts.get("no_branch_native", False),
                repeat_check=opts.get("branch_repeat_check", 0), meta=meta,
                mode=opts.get("gr_mode", "speed"))
            summary = result["summary"]
            b = result["branches"]
            record.update(genrand=b["policy"], policy_events=b["policy_events"],
                          n_branches=b["n_branches"], branch_collisions=b["collisions"],
                          branch_teleports=b["teleports"], branch_wall_s=b["branch_wall_s"],
                          base_wall_s=b["base_wall_s"], repeat_checks=b["repeat_checks"])
        else:
            policy = _policy_for(opts, job["seed"], job["demand"], job["penetration"])
            summary = run_policy_episode(
                env, policy, seed=job["seed"], demand_vph=job["demand"],
                penetration=job["penetration"], warmup_s=opts["warmup"],
                horizon_s=opts["horizon"], episode_dir=out, meta=meta)
        record.update(status="complete", lane_commands=summary.get("lane_commands"), sim_seconds=summary["sim_seconds"],
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
                no_ttc=args.no_ttc, no_physics_log=args.no_physics_log,
                branch_cadence=getattr(args, "branch_cadence", 0),
                branch_arms=getattr(args, "branch_arms", 4),
                branch_horizon=getattr(args, "branch_horizon", 60),
                no_branch_native=getattr(args, "no_branch_native", False),
                branch_repeat_check=getattr(args, "branch_repeat_check", 0),
                gr_mode=getattr(args, "gr_mode", "speed"),
                cav_vtype=getattr(args, "cav_vtype", "normal_car"),
                population=getattr(args, "population", "legacy3"),
                policy_pairs=getattr(args, "policy_pairs", False),
                branch_first=getattr(args, "branch_first", 0),
                branch_points=getattr(args, "branch_points", 1),
                settings_wm_ckpt=getattr(args, "settings_wm_ckpt", None),
                settings_wm_config=getattr(args, "settings_wm_config", None))
    extras = sumocredit_options(args)
    if extras is not None:
        if not opts["policy_pairs"]:
            raise SystemExit("the S-SUMOCREDIT extras (--credit-horizon/--gate-*) need --policy-pairs")
        opts["sumocredit"] = extras
    if opts["policy_pairs"]:
        from traffic_shepherd.c4.baselines import BASELINES
        from traffic_shepherd.c4.policy_pairs import check_registered_seeds
        if args.policy in BASELINES or args.policy in ("random", "generic_random", "zero"):
            raise SystemExit("--policy-pairs needs a learned policy checkpoint")
        if opts["branch_cadence"] < 1:
            raise SystemExit("--policy-pairs needs --branch-cadence >= 1")
        try:
            check_registered_seeds(seeds, getattr(args, "registration", None))
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SystemExit(f"refused: {error}")
    elif opts["branch_cadence"] and args.policy != "generic_random":
        raise SystemExit("--branch-cadence is only supported with --policy generic_random")
    jobs = []
    for seed, demand, pen in plan_cases(seeds, demands, pens,
                                        getattr(args, "case_plan", "product"),
                                        getattr(args, "pen_weights", None)):
        check_case(seed, demand)                  # fail before any SUMO
        jid = job_id(seed, demand, pen)
        job = dict(job_id=jid, seed=seed, demand=demand, penetration=pen,
                   episode_dir=str(root / jid), opts=opts, attempt=1)
        if opts["branch_cadence"]:
            job["branch_dir"] = str(out / "branches" / policy_tag(args.policy) / jid)
        jobs.append(job)
    return jobs


def sumocredit_options(args) -> dict | None:
    """``PairExtras`` fields from the CLI, or None when every S-SUMOCREDIT flag is off."""
    from dataclasses import asdict
    if not (getattr(args, "credit_horizon", 0) or getattr(args, "gate_horizon", 0)
            or getattr(args, "gate_table", None) or getattr(args, "long_group_horizon", 0)):
        return None
    from traffic_shepherd.c4.sumocredit import PairExtras
    extras = PairExtras(credit_horizon=args.credit_horizon, credit_group_max=args.credit_group_max,
                        credit_flips=args.credit_flips, credit_imagine=not args.no_credit_imagine,
                        gate_horizon=args.gate_horizon, gate_unit=args.gate_unit,
                        gate_reference=args.gate_reference, gate_state=args.gate_state,
                        gate_max_units=args.gate_max_units, gate_table=args.gate_table,
                        gate_candidates=args.gate_candidates, gate_mean_table=args.gate_mean_table,
                        gate_decision_period=args.gate_decision_period,
                        n_driver_samples=args.gate_driver_samples, seed=args.sumocredit_seed,
                        long_group_horizon=getattr(args, "long_group_horizon", 0),
                        long_group_time=getattr(args, "long_group_time", 0))
    return asdict(extras)


def plan_cases(seeds, demands, pens, plan: str = "product", pen_weights=None) -> list[tuple]:
    """``(seed, demand, penetration)`` cases.

    * ``product``: every seed x demand x penetration;
    * ``weighted-cycle``: ONE case per seed (one world per seed): the penetrations repeated by
      their integer weights (``--pen-weights 1,1,2,2``) are cycled seed by seed, the demand
      advances after every full penetration cycle (every combination within
      ``sum(weights) * len(demands)`` consecutive seeds)."""
    if plan == "product":
        return [(s, d, p) for s in seeds for d in demands for p in pens]
    if plan != "weighted-cycle":
        raise SystemExit(f"unknown --case-plan {plan!r}")
    weights = ([1] * len(pens) if not pen_weights
               else [int(w) for w in str(pen_weights).split(",")])
    if len(weights) != len(pens) or min(weights) < 1:
        raise SystemExit("--pen-weights needs one positive integer per penetration")
    cycle = [p for p, w in zip(pens, weights) for _ in range(w)]
    return [(s, demands[(i // len(cycle)) % len(demands)], cycle[i % len(cycle)])
            for i, s in enumerate(seeds)]


def _job_complete(job: dict) -> bool:
    if not _episode_complete(Path(job["episode_dir"])):
        return False
    if job.get("branch_dir"):
        return (Path(job["branch_dir"]) / "BRANCHES_COMPLETE.json").is_file()
    return True


def _preserve(path: Path) -> int:
    """Rename an incomplete directory to ``<name>.incomplete-<n>``; returns n (0 = absent)."""
    if not path.exists():
        return 0
    n = 1
    while path.with_name(f"{path.name}.incomplete-{n}").exists():
        n += 1
    path.rename(path.with_name(f"{path.name}.incomplete-{n}"))
    return n


def prepare(jobs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (todo, already complete); preserve incomplete directories."""
    todo, done = [], []
    for job in jobs:
        path = Path(job["episode_dir"])
        if _job_complete(job):
            done.append(job)
            continue
        n = _preserve(path)
        if job.get("branch_dir"):
            n = max(n, _preserve(Path(job["branch_dir"])))
        if n:
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
    ap.add_argument("--cav-vtype", default="normal_car",
                    help="vType of every CAV (owner 2026-10-14: normal_car); 'legacy' = drawn driver type")
    ap.add_argument("--population", choices=("legacy3", "textbook"), default="legacy3",
                    help="HDV driver population (owner 2026-10-14): legacy3 = 3 discrete vTypes; "
                         "textbook = continuous per-vehicle IDM/LC2013 params, CAVs on cav_typical "
                         "(--cav-vtype is then ignored); see traffic_shepherd/c4/population.py")
    ap.add_argument("--no-physics-log", action="store_true",
                    help="skip run/physics_steps.jsonl.gz (episode.npz is always written)")
    ap.add_argument("--case-plan", choices=("product", "weighted-cycle"), default="product",
                    help="product = seeds x demands x penetrations; weighted-cycle = one case "
                         "per seed (penetrations cycled by --pen-weights, then demands)")
    ap.add_argument("--pen-weights", default=None, help="e.g. 1,1,2,2 (weighted-cycle)")
    ap.add_argument("--gr-mode", choices=("speed", "accel"), default="speed",
                    help="generic_random process: speed (random target speeds, default) or "
                         "accel (random target accelerations)")
    ap.add_argument("--branch-cadence", type=int, default=0,
                    help="generic_random only: paired branches every N s after warmup (0 = off)")
    ap.add_argument("--branch-arms", type=int, default=4, help="re-seeded random arms SEED:j")
    ap.add_argument("--branch-horizon", type=int, default=60)
    ap.add_argument("--no-branch-native", action="store_true",
                    help="omit the all-native reference arm")
    ap.add_argument("--branch-repeat-check", type=int, default=0,
                    help="re-run SEED:0 at the first N points (determinism check)")
    ap.add_argument("--policy-pairs", action="store_true",
                    help="LOOP14: native base + learned-policy and native arms from each saved state")
    ap.add_argument("--branch-first", type=int, default=0,
                    help="policy pairs: first decision time (s; default warmup + 60)")
    ap.add_argument("--branch-points", type=int, default=1, help="policy pairs: decision points")
    ap.add_argument("--settings-wm-ckpt", default=None,
                    help="policy pairs: accepted WM whose driver encoder supplies learned settings")
    ap.add_argument("--settings-wm-config", default=None)
    ap.add_argument("--registration", default=None,
                    help="policy pairs: registered fresh TRAIN seed block (JSON), required")
    sc = ap.add_argument_group("S-SUMOCREDIT extras of --policy-pairs (all off by default)")
    sc.add_argument("--credit-horizon", type=int, default=0,
                    help="real-credit arms ONLY:i/GROUP/FLIP:i of this many seconds (0 = off)")
    sc.add_argument("--credit-group-max", type=int, default=3)
    sc.add_argument("--long-group-horizon", type=int, default=0)
    sc.add_argument("--long-group-time", type=int, default=0)
    sc.add_argument("--credit-flips", type=int, default=0)
    sc.add_argument("--no-credit-imagine", action="store_true",
                    help="skip the imagined counterpart of the credit arms")
    sc.add_argument("--gate-horizon", type=int, default=0,
                    help="gate-calibration arms of this many seconds (0 = off)")
    sc.add_argument("--gate-unit", choices=("fleet", "group", "cav"), default="fleet")
    sc.add_argument("--gate-reference", choices=("hold", "native"), default="native")
    sc.add_argument("--gate-state", choices=("belief", "oracle"), default="belief")
    sc.add_argument("--gate-max-units", type=int, default=4)
    sc.add_argument("--gate-table", default=None,
                    help="TRAIN reliability table: adds the gated policy arm (branch horizon)")
    sc.add_argument("--gate-candidates", type=int, default=1,
                    help="K learned-policy candidates (mean + K-1 samples) in calibration and the "
                         "GATED_K arm; 1 = gate-only")
    sc.add_argument("--gate-mean-table", default=None,
                    help="TRAIN sample-vs-mean reliability table (needed for a sample to replace the mean)")
    sc.add_argument("--gate-decision-period", type=float, default=5.0)
    sc.add_argument("--gate-driver-samples", type=int, default=1)
    sc.add_argument("--sumocredit-seed", type=int, default=0)
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
               "branches": sum(r.get("n_branches", 0) for r in ok),
               "branch_collisions": sum(r.get("branch_collisions", 0) for r in ok),
               "branch_teleports": sum(r.get("branch_teleports", 0) for r in ok),
               "case_plan": args.case_plan, "pen_weights": args.pen_weights,
               "population": getattr(args, "population", "legacy3"),
               "gr_mode": args.gr_mode if args.policy == "generic_random" else None,
               "branch_cadence_s": args.branch_cadence,
               "policy_pairs": bool(args.policy_pairs),
               "pair_sim_seconds": sum(r.get("pair_sim_seconds", 0) for r in ok),
               "pair_dJ": {k: v for r in ok for k, v in (r.get("pair_dJ") or {}).items()},
               "host_cpus": os.cpu_count(), "finished_utc": _now()}
    extra = [r["sumocredit"] for r in ok if r.get("sumocredit")]
    if extra:                                     # S-SUMOCREDIT (key absent when off)
        summary["sumocredit"] = {"jobs": len(extra), "sim_s": sum(e["sim_s"] for e in extra),
                                 "files": {k: sum(e["files"].get(k, 0) for e in extra)
                                           for k in ("gate", "gated", "credit")}}
    (out / "COLLECT_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1), flush=True)
    return 0 if len(ok) == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""GENRAND collection driver for a many-core Linux box (GENRAND_NOTES.md).

    python -m traffic_shepherd.c4.genrand_driver --out runs/active/C4-GENRAND-<date> \\
        --plain-seeds 961000-961479 --branch-seeds 961480-961515 --workers 46

Runs two :mod:`traffic_shepherd.c4.collect` processes concurrently (thread env = 1 per worker):

* ``<out>/plain``  - generic_random episodes (one world per seed, weighted-cycle cases);
* ``<out>/branch`` - generic_random episodes with paired branches every ``--branch-cadence`` s
  (``--branch-arms`` re-seeded random arms + the all-native reference, ``--branch-horizon`` s).

Workers are split by estimated work.  Both phases are resumable (finished episodes are
skipped; incomplete ones preserved as ``.incomplete-N`` and re-run).  When both phases
complete, the episodes are converted with ``python -m wms.data.c4_episodes`` into
``<out>/c4_episodes`` (write-once: an existing non-empty conversion is kept and a new
``c4_episodes.<n>`` is written).  ``<out>/GENRAND_MANIFEST.json`` records the arguments, git
commit, phase return codes, the wall-time estimate and the conversion; ``<out>/
COLLECT_SUMMARY.json`` aggregates the two phases' summaries.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

THREAD_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
# Measured on the laptop (2026-10-14 smoke, speed mode, m2700): warmup ~6 sim-s/wall-s,
# generic_random window ~4 sim-s/wall-s, one 60 s branch incl. fresh SUMO + loadState ~7.7 s.
WARMUP_SPEED, WINDOW_SPEED, BRANCH_WALL_S = 6.0, 4.0, 7.7
FORBIDDEN = ("holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "sealed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def n_seeds(text: str) -> int:
    from traffic_shepherd.c4.seeds import parse_seeds
    return len(parse_seeds(text)) if text else 0


def estimate(*, n_plain: int, n_branch: int, warmup: int, horizon: int, cadence: int,
             arms: int, branch_h: int, workers: int, speed_factor: float = 1.0) -> dict:
    """Wall-time estimate (s) from the laptop speeds; ``speed_factor`` > 1 = slower cores."""
    from traffic_shepherd.c4.genrand import branch_times
    base = warmup / WARMUP_SPEED + horizon / WINDOW_SPEED
    points = len(branch_times(warmup, horizon, cadence, branch_h)) if cadence > 0 else 0
    per_branch_ep = base + points * (arms + 1) * BRANCH_WALL_S * branch_h / 60.0
    work = (n_plain * base + n_branch * per_branch_ep) * speed_factor
    wall = work / max(1, workers)
    # a branched episode cannot be split across workers: the wall time is at least one of them
    wall = max(wall, per_branch_ep * speed_factor if n_branch else 0.0)
    win60 = max(0, (horizon - 60) // 10 + 1)
    return {"cpu_s": work, "wall_s": wall, "wall_h": wall / 3600.0,
            "base_episode_wall_s": base * speed_factor,
            "branched_episode_wall_s": per_branch_ep * speed_factor,
            "branch_points_per_episode": points,
            "windows_60s_nonoverlap": n_plain * (horizon // 60) + n_branch * (horizon // 60),
            "windows_60s_stride10": (n_plain + n_branch) * win60,
            "branch_windows_60s": n_branch * points * (arms + 1)}


def split_workers(workers: int, n_plain: int, n_branch: int, est_plain: float,
                  est_branch: float) -> tuple[int, int]:
    if n_branch == 0:
        return workers, 0
    if n_plain == 0:
        return 0, workers
    share = n_branch * est_branch / (n_branch * est_branch + n_plain * est_plain)
    wb = min(n_branch, max(1, round(workers * share)))
    return max(1, workers - wb), wb


def collect_cmd(args, seeds: str, out: Path, workers: int, branch: bool) -> list[str]:
    cmd = [sys.executable, "-m", "traffic_shepherd.c4.collect", "--policy", "generic_random",
           "--gr-mode", args.gr_mode, "--seeds", seeds, "--case-plan", "weighted-cycle",
           "--penetrations", args.penetrations, "--pen-weights", args.pen_weights,
           "--demands", args.demands, "--warmup", str(args.warmup),
           "--horizon", str(args.horizon), "--workers", str(workers), "--out", str(out)]
    if args.no_physics_log:
        cmd.append("--no-physics-log")
    if getattr(args, "population", None) is not None:
        cmd += ["--population", args.population]
    if branch:
        cmd += ["--branch-cadence", str(args.branch_cadence), "--branch-arms",
                str(args.branch_arms), "--branch-horizon", str(args.branch_horizon),
                "--branch-repeat-check", str(args.branch_repeat_check)]
    return cmd


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).resolve().parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m traffic_shepherd.c4.genrand_driver",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--plain-seeds", default="", help="e.g. 961000-961479 (one world per seed)")
    ap.add_argument("--branch-seeds", default="", help="e.g. 961480-961515")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--penetrations", default="0.05,0.1,0.2,0.4")
    ap.add_argument("--pen-weights", default="1,1,2,2")
    ap.add_argument("--demands", default="2400,2700")
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=900)
    ap.add_argument("--gr-mode", choices=("speed", "accel"), default="speed")
    ap.add_argument("--population", choices=("legacy3", "textbook"), default=None,
                    help="forward the HDV population to both collectors (omitted: collector default)")
    ap.add_argument("--branch-cadence", type=int, default=60)
    ap.add_argument("--branch-arms", type=int, default=4)
    ap.add_argument("--branch-horizon", type=int, default=60)
    ap.add_argument("--branch-repeat-check", type=int, default=1,
                    help="determinism re-run of SEED:0 at the first N points per episode")
    ap.add_argument("--no-physics-log", action="store_true",
                    help="skip run/physics_steps.jsonl.gz (saves disk; episode.npz is kept)")
    ap.add_argument("--speed-factor", type=float, default=1.0,
                    help="estimate only: core slowness relative to the laptop (1.5 = 50%% slower)")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--no-convert", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out).resolve()
    if any(tok in str(out).lower() for tok in FORBIDDEN):
        raise SystemExit("refusing an output path with a restricted-split token")
    n_plain, n_branch = n_seeds(args.plain_seeds), n_seeds(args.branch_seeds)
    if n_plain + n_branch == 0:
        raise SystemExit("no seeds")
    est = estimate(n_plain=n_plain, n_branch=n_branch, warmup=args.warmup, horizon=args.horizon,
                   cadence=args.branch_cadence, arms=args.branch_arms,
                   branch_h=args.branch_horizon, workers=args.workers,
                   speed_factor=args.speed_factor)
    wp, wb = split_workers(args.workers, n_plain, n_branch, est["base_episode_wall_s"],
                           est["branched_episode_wall_s"])
    est.update(workers_plain=wp, workers_branch=wb)
    print(f"[genrand] {n_plain} plain + {n_branch} branched episodes on {args.workers} workers "
          f"(plain {wp}, branch {wb}); estimated wall {est['wall_h']:.2f} h "
          f"(x{args.speed_factor} core speed), {est['windows_60s_nonoverlap']} non-overlapping "
          f"60 s windows ({est['windows_60s_stride10']} at 10 s stride), "
          f"{est['branch_windows_60s']} paired-branch 60 s windows", flush=True)
    if args.estimate_only:
        print(json.dumps(est, indent=1))
        return 0
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "GENRAND_MANIFEST.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file() else {"schema": "c4_genrand_run/1", "runs": []})
    run = {"started_utc": _now(), "args": vars(args), "git_commit": _git_commit(),
           "host_cpus": os.cpu_count(), "estimate": est}
    manifest["runs"].append(run)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    env = dict(os.environ, **THREAD_ENV)
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")]
                                        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    procs = {}
    t0 = time.monotonic()
    for name, seeds, workers, branch in (("plain", args.plain_seeds, wp, False),
                                         ("branch", args.branch_seeds, wb, True)):
        if not seeds:
            continue
        cmd = collect_cmd(args, seeds, out / name, workers, branch)
        log = open(out / f"{name}.log", "a", encoding="utf-8")
        log.write(f"\n# {_now()} {' '.join(cmd)}\n")
        log.flush()
        procs[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                        cwd=root), log)
        print(f"[genrand] started {name}: pid {procs[name][0].pid}", flush=True)
    rcs = {}
    for name, (proc, log) in procs.items():
        rcs[name] = proc.wait()
        log.close()
        print(f"[genrand] {name} finished rc={rcs[name]}", flush=True)
    run.update(finished_utc=_now(), wall_s=time.monotonic() - t0, return_codes=rcs)
    summaries = {}
    for name in procs:
        path = out / name / "COLLECT_SUMMARY.json"
        if path.is_file():
            summaries[name] = json.loads(path.read_text(encoding="utf-8"))
    agg = {"schema": "c4_genrand_collect_summary/1", "phases": summaries, "return_codes": rcs,
           "wall_s": run["wall_s"], "estimate": est, "finished_utc": run["finished_utc"]}
    for key in ("episodes_run", "episodes_complete", "episodes_failed", "simulated_seconds",
                "collisions", "teleports", "branches", "branch_collisions", "branch_teleports"):
        agg[key] = sum((s.get(key) or 0) for s in summaries.values())
    (out / "COLLECT_SUMMARY.json").write_text(json.dumps(agg, indent=1), encoding="utf-8")
    ok = all(rc == 0 for rc in rcs.values())
    if ok and not args.no_convert:
        dst = out / "c4_episodes"
        n = 1
        while dst.exists() and any(dst.iterdir()):
            dst = out / f"c4_episodes.{n}"
            n += 1
        roots = [str(out / name) for name in procs]
        cmd = [sys.executable, "-m", "wms.data.c4_episodes", "--episodes", *roots,
               "--out", str(dst)]
        with open(out / "convert.log", "a", encoding="utf-8") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env,
                                cwd=root).returncode
        run["conversion"] = {"out": str(dst), "return_code": rc, "command": cmd}
        print(f"[genrand] conversion -> {dst} rc={rc}", flush=True)
        ok = rc == 0
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: agg[k] for k in ("episodes_complete", "episodes_failed", "branches",
                                          "collisions", "branch_collisions", "wall_s")}),
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

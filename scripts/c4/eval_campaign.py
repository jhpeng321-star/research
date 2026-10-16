#!/usr/bin/env python
"""Final C4 evaluation campaign driver (CPU box; SUMO episodes, paired across controllers).

    python eval_campaign.py plan   --out DIR --seeds eval_seeds.json [...]
    python eval_campaign.py run    --out DIR --seeds eval_seeds.json --workers N [...]
    python eval_campaign.py status --out DIR

Grid: controllers x penetrations x demands x seeds-of-that-demand.  The seed set of a demand cell is
fixed by make_eval_seeds.py (development seeds only); every controller and every penetration is run on
the SAME (seed, demand), so the comparison is paired.  Jobs are ordered seed-major so a stopped campaign
leaves complete paired blocks.

Resumable, no retry loops: a finished job leaves ``<job>.DONE``, a failed one ``<job>.FAILED`` (log kept
as ``<job>.log``); neither is rerun unless ``--rerun-failed`` (the failed directory is renamed, never
deleted).  Children run in their own process group and are killed by group id on timeout / SIGTERM
(never by image name).

Controller commands:
  native, damping, pi : python -m traffic_shepherd evaluate ...   (existing CLI; role gate
                        traffic_shepherd.seeds.require_development_case), or with
                        --classic-harness deploy the C4 runner below (same SUMO harness/summary
                        as the C4 controllers: CavSumoEnv, warmup + window)
  wall, c4_policy, c4_policy_rsu_check_oracle, c4_policy_rsu_check_belief :
                        python -m $C4_EVAL_MODULE (default wms.control.deploy) --controller
                        {wall,c4_policy,c4_policy_rsu_check} --seed --loading --penetration --role
                        --output [--policy-ckpt --wm-ckpt --rsu-state oracle|belief] + --deploy-args
  The belief variant uses the RSU BeliefTracker (no separate checkpoint); --belief-ckpt is passed
  through if given and ignored by the runner.  The C4 runner's seed gate
  (traffic_shepherd.c4.seeds.check_case) only admits 4107-4148: see OPS_NOTES section 10.
  MAPPO is deferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CLASSIC = {"native": ["--controller", "native"],
           "damping": ["--controller", "damping"],
           "pi": ["--controller", "pi-external-u", "--desired-speed-mps", "31.29"]}
C4_CONTROLLERS = ("wall", "c4_policy", "c4_policy_rsu_check_oracle", "c4_policy_rsu_check_belief")
DEFAULT_CONTROLLERS = ("native", "damping", "pi", "c4_policy",
                       "c4_policy_rsu_check_oracle", "c4_policy_rsu_check_belief")
DEFAULT_PENS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60)
TRAINED_PENS = (0.05, 0.10, 0.20, 0.40)
DEMANDS = (2400, 2600, 2700)


def build_jobs(args) -> list[dict]:
    seeds_doc = json.loads(Path(args.seeds).read_text(encoding="utf-8"))
    cells = seeds_doc["cells"]
    jobs = []
    py = args.python
    for demand in args.demands:
        entries = cells[str(demand)][: args.per_demand]
        for entry in entries:                       # seed-major => paired blocks complete first
            for pen in args.penetrations:
                for ctrl in args.controllers:
                    jdir = Path(args.out) / ctrl / f"m{demand}" / f"p{pen:.2f}" / f"s{entry['seed']}"
                    common = ["--seed", str(entry["seed"]), "--loading", str(demand),
                              "--penetration", f"{pen:g}", "--role", entry["role"], "--output", str(jdir)]
                    status = "pending"
                    extra = shlex.split(args.deploy_args or "")
                    if ctrl in CLASSIC and args.classic_harness == "deploy":
                        cmd = [py, "-m", args.c4_module, "--controller", ctrl, *common, *extra]
                    elif ctrl in CLASSIC:
                        cmd = [py, "-m", "traffic_shepherd", "evaluate", *common, *CLASSIC[ctrl]]
                    else:
                        base = "c4_policy_rsu_check" if ctrl.startswith("c4_policy_rsu_check") else ctrl
                        cmd = [py, "-m", args.c4_module, "--controller", base, *common, *extra]
                        if base in ("c4_policy", "c4_policy_rsu_check"):
                            if not args.policy_ckpt:
                                raise SystemExit(f"--policy-ckpt is required for {ctrl}")
                            cmd += ["--policy-ckpt", args.policy_ckpt]
                        if base == "c4_policy_rsu_check":
                            if not args.wm_ckpt:
                                raise SystemExit(f"--wm-ckpt is required for {ctrl}")
                            cmd += ["--wm-ckpt", args.wm_ckpt,
                                    "--rsu-state", "oracle" if ctrl.endswith("oracle") else "belief"]
                            if ctrl.endswith("belief") and args.belief_ckpt:
                                cmd += ["--belief-ckpt", args.belief_ckpt]
                    jobs.append({"controller": ctrl, "demand": demand, "penetration": pen,
                                 "seed": entry["seed"], "role": entry["role"], "dir": str(jdir),
                                 "cmd": cmd, "status0": status,
                                 "trained_penetration": pen in TRAINED_PENS})
    return jobs


def marker(job: dict, kind: str) -> Path:
    return Path(job["dir"] + "." + kind)


def job_state(job: dict) -> str:
    if marker(job, "DONE").exists():
        return "done"
    if marker(job, "FAILED").exists():
        return "failed"
    return job["status0"]


class Runner:
    def __init__(self, args):
        self.args = args
        self.stop = threading.Event()
        self.procs: dict[int, subprocess.Popen] = {}
        self.lock = threading.Lock()

    def terminate_all(self, *_):
        self.stop.set()
        with self.lock:
            for p in list(self.procs.values()):
                try:
                    os.killpg(p.pid, signal.SIGTERM)       # own child groups only
                except (ProcessLookupError, PermissionError, AttributeError):
                    pass

    def run_job(self, job: dict) -> str:
        state = job_state(job)
        if self.stop.is_set():
            return "skipped_stop"
        if state == "done":
            return "done"
        if state == "pending_belief":
            return state
        if state == "failed":
            if not self.args.rerun_failed:
                return "failed"
            stamp = time.strftime("%Y%m%dT%H%M%S")
            d = Path(job["dir"])
            if d.exists():
                d.rename(d.with_name(d.name + f".failed_{stamp}"))
            marker(job, "FAILED").rename(Path(job["dir"] + f".FAILED_{stamp}"))
            Path(job["dir"] + ".log").rename(Path(job["dir"] + f".log.failed_{stamp}"))
        d = Path(job["dir"])
        if d.exists():
            # partial output of an interrupted job: preserve it, the evaluator needs a fresh directory
            d.rename(d.with_name(d.name + f".partial_{time.strftime('%Y%m%dT%H%M%S')}"))
        d.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        t0 = time.time()
        with open(job["dir"] + ".log", "ab") as log:
            log.write((" ".join(job["cmd"]) + "\n").encode())
            log.flush()
            popen_kw = {"start_new_session": True} if hasattr(os, "setsid") else {}
            p = subprocess.Popen(job["cmd"], stdout=log, stderr=subprocess.STDOUT, env=env, **popen_kw)
            with self.lock:
                self.procs[p.pid] = p
            try:
                rc = p.wait(timeout=self.args.timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (ProcessLookupError, AttributeError):
                    p.terminate()
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except (ProcessLookupError, AttributeError):
                        p.kill()
                rc = -999
            finally:
                with self.lock:
                    self.procs.pop(p.pid, None)
        rec = {"rc": rc, "seconds": round(time.time() - t0, 1), "host": platform.node(),
               "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cmd": job["cmd"]}
        kind = "DONE" if rc == 0 else "FAILED"
        marker(job, kind).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
        return "done" if rc == 0 else "failed"


def summarize(jobs: list[dict]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for j in jobs:
        out.setdefault(j["controller"], {})
        s = job_state(j)
        out[j["controller"]][s] = out[j["controller"]].get(s, 0) + 1
    return out


def sizing(jobs: list[dict], workers: int, minutes: float) -> str:
    runnable = [j for j in jobs if j["status0"] == "pending"]
    core_h = len(runnable) * minutes / 60.0
    return (f"{len(runnable)} runnable episodes x {minutes:g} min/episode = {core_h:.0f} core-hours "
            f"-> {core_h / max(workers, 1):.1f} h wall on {workers} worker(s)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("plan", "run", "status"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default=None, help="eval_seeds.json from make_eval_seeds.py")
    ap.add_argument("--controllers", default=",".join(DEFAULT_CONTROLLERS))
    ap.add_argument("--include-wall", action="store_true", help="add the optional wall law controller")
    ap.add_argument("--penetrations", default=",".join(f"{p:.2f}" for p in DEFAULT_PENS))
    ap.add_argument("--demands", default=",".join(map(str, DEMANDS)))
    ap.add_argument("--per-demand", type=int, default=8, help="seeds per demand cell (max in the seeds file)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--timeout", type=float, default=7200.0, help="seconds per episode")
    ap.add_argument("--minutes-per-episode", type=float, default=6.0, help="only used by the sizing line")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--c4-module", default=os.environ.get("C4_EVAL_MODULE", "wms.control.deploy"))
    ap.add_argument("--classic-harness", choices=("evaluate", "deploy"),
                    default=os.environ.get("C4_CLASSIC_HARNESS", "evaluate"),
                    help="native/damping/pi via 'traffic_shepherd evaluate' or the C4 runner")
    ap.add_argument("--deploy-args", default=os.environ.get("C4_DEPLOY_ARGS", ""),
                    help="extra arguments for the C4 runner, e.g. '--warmup 300 --horizon 600 "
                         "--decision-period 5 --imagine-horizon 20 --margin 2' (drivers: point estimate)")
    ap.add_argument("--policy-ckpt", default=os.environ.get("POLICY_CKPT"))
    ap.add_argument("--wm-ckpt", default=os.environ.get("WM_CKPT"))
    ap.add_argument("--belief-ckpt", default=os.environ.get("BELIEF_CKPT"))
    ap.add_argument("--rerun-failed", action="store_true")
    args = ap.parse_args(argv)
    args.controllers = [c for c in args.controllers.split(",") if c]
    if args.include_wall and "wall" not in args.controllers:
        args.controllers.append("wall")
    if "mappo" in args.controllers:
        raise SystemExit("evaluate MAPPO as controller 'policy' with --policy-ckpt <mappo run>/policy_latest.pt "
                         "(c4_dreamer/3 payload, trainer=mappo) in its own --out; see MAPPO_NOTES.md")
    unknown = [c for c in args.controllers if c not in CLASSIC and c not in C4_CONTROLLERS]
    if unknown:
        raise SystemExit(f"unknown controllers {unknown}")
    args.penetrations = [float(p) for p in args.penetrations.split(",") if p]
    args.demands = [int(d) for d in args.demands.split(",") if d]
    out = Path(args.out)
    if args.action == "status" and args.seeds is None:
        doc = json.loads((out / "CAMPAIGN.json").read_text(encoding="utf-8"))
        args.seeds = doc["seeds_file"]
    if not args.seeds:
        raise SystemExit("--seeds is required")

    jobs = build_jobs(args)
    seeds_sha = hashlib.sha256(Path(args.seeds).read_bytes()).hexdigest()
    ident = {"controllers": args.controllers, "penetrations": args.penetrations, "demands": args.demands,
             "per_demand": args.per_demand, "seeds_file": str(args.seeds), "seeds_sha256": seeds_sha,
             "policy_ckpt": args.policy_ckpt, "wm_ckpt": args.wm_ckpt, "belief_ckpt": args.belief_ckpt,
             "c4_module": args.c4_module, "classic_harness": args.classic_harness,
             "deploy_args": args.deploy_args}
    ident_sha = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()

    if args.action in ("plan", "status"):
        print(f"{len(jobs)} jobs; controllers {args.controllers}; penetrations {args.penetrations}; "
              f"demands {args.demands}; seeds/demand {args.per_demand}")
        for c, st in summarize(jobs).items():
            print(f"  {c:30s} {st}")
        print(sizing(jobs, args.workers, args.minutes_per_episode))
        pend = [j for j in jobs if j["status0"] == "pending_belief"]
        if pend:
            print(f"  {len(pend)} belief-variant jobs wait for --belief-ckpt (the reconstructor)")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    camp = out / "CAMPAIGN.json"
    if camp.exists():
        prev = json.loads(camp.read_text(encoding="utf-8"))
        if prev.get("identity_sha256") != ident_sha:
            raise SystemExit("CAMPAIGN.json exists with a different definition; use a new --out directory")
    else:
        camp.write_text(json.dumps({"identity": ident, "identity_sha256": ident_sha, "seeds_file": args.seeds,
                                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "n_jobs": len(jobs)}, indent=1) + "\n", encoding="utf-8")
    (out / "jobs.tsv").write_text(
        "controller\tdemand\tpenetration\tseed\trole\tdir\n" +
        "".join(f"{j['controller']}\t{j['demand']}\t{j['penetration']:g}\t{j['seed']}\t{j['role']}\t{j['dir']}\n"
                for j in jobs), encoding="utf-8")
    runner = Runner(args)
    signal.signal(signal.SIGTERM, runner.terminate_all)
    signal.signal(signal.SIGINT, runner.terminate_all)
    print(sizing(jobs, args.workers, args.minutes_per_episode), flush=True)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, res in enumerate(pool.map(runner.run_job, jobs), 1):
            done += res == "done"
            failed += res == "failed"
            if i % 20 == 0 or res == "failed":
                print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(jobs)} processed: done {done}, failed {failed}",
                      flush=True)
    status = summarize(jobs)
    (out / "STATUS.json").write_text(json.dumps(status, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print("final:", json.dumps(status, sort_keys=True))
    return 0 if not any(s.get("failed") for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

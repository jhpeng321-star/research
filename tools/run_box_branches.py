"""Rented-box BRANCH stage runner (bundle jobs J1 chunk 4, J3 truearms).

The campaign launcher caps one launch at 10 workers when no branch step-down
reference exists (chunk-4 gate), so the box runs N launcher INSTANCES in
parallel (default 4 x 9 workers).  They share the box's claim dir; each has
its own machine id (box1..boxN) and its own --out dir, so the per-run O_EXCL
claims keep them disjoint.  The laptop's runs are excluded by pre-seeded
'laptop' claim files shipped in the payload (static split, WM-58 ledger).

Own child PIDs only (recorded in <out-prefix>_pids.json); waits for all; exit
code 0 only if every instance exits 0.  Re-running is safe: an existing --out
dir gets --resume (only when NO sibling instance is still running - a resume
archive-renames incomplete run dirs).

DEFAULTS (no plan/registration/... flags) are the J1 chunk-4 branch launch,
identical to the original wrapper.  Every path is overridable, e.g. the
WM-73 truearms launch (fresh runs root + --bases-root; instances 2..N join
the runs root that instance 1 creates):

  python tools/run_box_branches.py --smoke-verdict <path> \
      --plan output/campaign_stage1_v3/results_plan_truearms_chunk2/WORLD_PLAN.json \
      --registration output/campaign_stage1_v3/results_plan_truearms/TRUEARMS_REGISTRATION.md \
      --authorization output/campaign_stage1_v3/results_plan_truearms/AUTHORIZATION.json \
      --runs-root runs/campaign_stage1_chunk2_truearms \
      --bases-root runs/campaign_stage1_chunk2 \
      --claim-dir runs/campaign_stage1_chunk2_truearms_CLAIMS \
      --out-prefix output/campaign_stage1_chunk2_truearms_launch_box --fresh-root

Usage (cwd E:/research-dense):
  python tools/run_box_branches.py --smoke-verdict <path> [--instances 4] [--workers 9]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("RESEARCH_DENSE_ROOT", "E:/research-dense"))
WT = Path(os.environ.get("RD_WT_ROOT", "E:/rd-wt"))

#: J1 (chunk-4 branches) defaults - relative to ROOT unless absolute
DEFAULT_PLAN = "output/campaign_stage1_v3/results_plan_chunk4/WORLD_PLAN.json"
DEFAULT_AUTHORIZATION = "output/campaign_stage1_chunk4/AUTHORIZATION.json"
DEFAULT_REGISTRATION = "output/campaign_stage1_v3/results_plan_chunk4/CHUNK4_REGISTRATION.md"
DEFAULT_RUNS_ROOT = "runs/campaign_stage1_chunk4"
DEFAULT_CLAIM_DIR = "runs/campaign_stage1_chunk4/CLAIMS"
DEFAULT_OUT_PREFIX = "output/campaign_stage1_chunk4_launch_box"
DEFAULT_WM41_VERDICT = "output/campaign_pilot_v1_launch_r4/EXACTNESS_VERDICT.json"
DEFAULT_WM43_SUMMARY = "runs/active/wm40_wall_mechanics_v1_r7/LAUNCH_SUMMARY.json"


def _resolve(root: Path, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke-verdict", required=True)
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--stagger-s", type=float, default=30.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--authorization", default=DEFAULT_AUTHORIZATION)
    ap.add_argument("--registration", default=DEFAULT_REGISTRATION)
    ap.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--bases-root", default=None,
                    help="restore from <bases-root>/<from_run> (fix27/truearms plans)")
    ap.add_argument("--claim-dir", default=DEFAULT_CLAIM_DIR)
    ap.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX,
                    help="instance i uses <out-prefix><i> as --out, "
                    "<out-prefix><i>.log as its log and <out-prefix>_pids.json")
    ap.add_argument("--machine-prefix", default="box")
    ap.add_argument("--wm41-verdict", default=DEFAULT_WM41_VERDICT)
    ap.add_argument("--wm41-registration", default=None,
                    help="default: <RD_WT_ROOT>/wm34/output/campaign_pilot_v1/REGISTRATION.md")
    ap.add_argument("--wm43-summary", default=DEFAULT_WM43_SUMMARY)
    ap.add_argument("--fresh-root", action="store_true",
                    help="WM-73: the runs root does not exist yet; instance 1 creates "
                    "it (no --out pre-creation), instances 2..N pass --join-runs-root "
                    "after the LAUNCH_ROOT.json marker of instance 1 appears")
    ap.add_argument("--marker-timeout-s", type=float, default=1800.0)
    return ap


def build_base_argv(a, *, root: Path = ROOT, wt: Path = WT) -> list:
    """The launcher argv shared by every instance (no --out/--machine-id)."""
    wm41_registration = (
        _resolve(root, a.wm41_registration) if a.wm41_registration
        else wt / "wm34/output/campaign_pilot_v1/REGISTRATION.md")
    argv = [
        sys.executable, str(root / "output/campaign_stage1_v3/run_chunk.py"), "launch",
        "--stage", "branches",
        "--plan", str(_resolve(root, a.plan)),
        "--authorization", str(_resolve(root, a.authorization)),
        "--registration", str(_resolve(root, a.registration)),
        "--runs-root", str(_resolve(root, a.runs_root)),
        "--wm41-verdict", str(_resolve(root, a.wm41_verdict)),
        "--wm41-registration", str(wm41_registration),
        "--wm43-summary", str(_resolve(root, a.wm43_summary)),
        "--smoke-verdict", str(_resolve(root, a.smoke_verdict)),
        "--workers", str(a.workers), "--start-interval-s", "4", "--start-jitter-s", "1",
        "--claim-dir", str(_resolve(root, a.claim_dir)),
    ]
    if a.bases_root:
        argv += ["--bases-root", str(_resolve(root, a.bases_root))]
    return argv


def instance_argv(a, i: int, base: list, *, root: Path = ROOT, resume: bool = False) -> tuple:
    """(argv, out_dir, log_path) for instance ``i`` (1-based)."""
    out = _resolve(root, f"{a.out_prefix}{i}")
    argv_i = base + ["--out", str(out), "--machine-id", f"{a.machine_prefix}{i}"]
    if resume:
        argv_i += ["--resume"]
    elif a.fresh_root and i > 1:
        argv_i += ["--join-runs-root"]
    log = out.parent / f"{Path(a.out_prefix).name}{i}.log"
    return argv_i, out, log


def wait_for_marker(marker: Path, timeout_s: float, *, sleep=time.sleep, clock=time.monotonic,
                    poll_s: float = 5.0, alive=None) -> bool:
    """Wait for instance 1's LAUNCH_ROOT.json; False on timeout or when
    ``alive()`` (instance 1's process still running) turns false."""
    deadline = clock() + timeout_s
    while not marker.is_file():
        if clock() >= deadline or (alive is not None and not alive()):
            return marker.is_file()
        sleep(poll_s)
    return True


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if not 1 <= a.workers <= 10:
        print("REFUSING: --workers must be 1..10 per instance (launcher gate)", file=sys.stderr)
        return 2
    base = build_base_argv(a, root=ROOT, wt=WT)
    procs = []
    pid_path = _resolve(ROOT, f"{a.out_prefix}_pids.json")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    runs_root = _resolve(ROOT, a.runs_root)
    marker = runs_root / "LAUNCH_ROOT.json"
    for i in range(1, a.instances + 1):
        out = _resolve(ROOT, f"{a.out_prefix}{i}")
        resume = out.is_dir() and any(out.iterdir())
        if a.fresh_root and i == 1 and runs_root.exists() and not resume:
            print("REFUSING: --fresh-root but the runs root already exists "
                  f"({runs_root}); re-run without --fresh-root for a resume",
                  file=sys.stderr)
            return 2
        fresh_instance = a.fresh_root and not resume
        if not a.dry_run and not fresh_instance:
            out.mkdir(parents=True, exist_ok=True)  # the branches stage requires an existing --out
        argv_i, out, log = instance_argv(a, i, base, root=ROOT, resume=resume)
        print(("DRY: " if a.dry_run else "") + " ".join(argv_i), flush=True)
        if a.dry_run:
            continue
        if fresh_instance and i > 1 and not wait_for_marker(
                marker, a.marker_timeout_s,
                alive=lambda: procs[0][1].poll() is None):
            print(f"REFUSING: instance 1 never wrote {marker}", file=sys.stderr)
            break
        out.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "a", encoding="utf-8")
        p = subprocess.Popen(argv_i, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
        procs.append((i, p, fh))
        pid_path.write_text(json.dumps({f"{a.machine_prefix}{j}": q.pid for j, q, _ in procs},
                                       indent=1), encoding="utf-8")
        time.sleep(a.stagger_s)
    rc = 0
    for i, p, fh in procs:
        code = p.wait()
        fh.close()
        print(f"{a.machine_prefix}{i} exit {code}", flush=True)
        rc = rc or code
    if a.instances and not procs and not a.dry_run:
        return 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

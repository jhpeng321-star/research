"""Rented-box J3 (WM-73): run the truearms branch stage for chunk 2, then
chunk 3, via tools/run_box_branches.py (N launcher instances x <=9 workers,
shared per-chunk claim dir, fresh runs root created by instance 1).

Own child processes only; exit 0 only if both chunks exit 0.  Re-running is
safe: tools/run_box_branches.py resumes an instance whose --out already exists
(resume only when every instance of that chunk has stopped).

Usage (cwd E:/research-dense):
  python tools/run_box_truearms.py [--instances 4] [--workers 9] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("RESEARCH_DENSE_ROOT", "E:/research-dense"))
PKG = "output/campaign_stage1_v3"


def chunk_argv(chunk: str, instances: int, workers: int, dry_run: bool) -> list[str]:
    runs_root = ROOT / f"runs/campaign_stage1_{chunk}_truearms"
    argv = [
        sys.executable, str(ROOT / "tools/run_box_branches.py"),
        "--smoke-verdict", "output/campaign_stage1_truearms_smoke/SMOKE_VERDICT.json",
        "--plan", f"{PKG}/results_plan_truearms_{chunk}/WORLD_PLAN.json",
        "--registration", f"{PKG}/results_plan_truearms/TRUEARMS_REGISTRATION.md",
        "--authorization", "output/campaign_stage1_truearms/AUTHORIZATION.json",
        "--runs-root", f"runs/campaign_stage1_{chunk}_truearms",
        "--bases-root", f"runs/campaign_stage1_{chunk}",
        "--claim-dir", f"runs/campaign_stage1_{chunk}_truearms_CLAIMS",
        "--out-prefix", f"output/campaign_stage1_{chunk}_truearms_launch_box",
        "--instances", str(instances), "--workers", str(workers),
    ]
    # instance 1 creates the fresh root; a re-run after it exists resumes
    if not runs_root.exists():
        argv.append("--fresh-root")
    if dry_run:
        argv.append("--dry-run")
    return argv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rc = 0
    for chunk in ("chunk2", "chunk3"):
        cmd = chunk_argv(chunk, a.instances, a.workers, a.dry_run)
        print(" ".join(cmd), flush=True)
        code = subprocess.call(cmd, cwd=str(ROOT))
        print(f"{chunk} exit {code}", flush=True)
        rc = rc or code
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU-only real CLI train-argv smoke on synthetic kinematics + copied TRAIN IDs.

Run from workspace root with the specified rl_driving Python. Destination new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "output/wd01/wm_n1"))
from wms.tests.mini_switch_pack import build  # noqa: E402
from wms.mixdata import HeldOutGuard, IdentityBook  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = (ROOT / args.out).resolve()
    out.relative_to(ROOT)
    if out.exists():
        raise ValueError("CPU output exists; choose a new path")
    out.mkdir(parents=True)
    meta = ROOT / "_in/swpack_meta"
    index = json.loads((meta / "ACTION_TIMELINE_INDEX_train.json").read_text())
    guard = HeldOutGuard.load(meta / "ZERO_SHOT_HELDOUT.json")
    guard.refuse_worlds([item["world"] for item in index["items"]], where="copied TRAIN index")
    book = IdentityBook(meta / "identities_switching_train")
    if index["n_items"] != book.ids.shape[0] or index["k_max"] != book.ids.shape[1]:
        raise ValueError("copied timeline/identity counts disagree")
    sw = build(out / "switch", copied_meta=meta, k=8)
    original = build(out / "original", original=True, copied_meta=meta, k=8)
    argv = [sys.executable, "-m", "wms.cli", "train", "--device", "cpu", "--threads", "1",
        "--num-workers", "0", "--batch-size", "2", "--pack", original["pack"],
        "--identities", original["identities"], "--window-list", original["windows"],
        "--switch-pack", sw["pack"], "--switch-identities", sw["identities"],
        "--switch-window-list", sw["windows"], "--switch-timeline-index", sw["index"],
        "--zero-shot-heldout", sw["heldout"], "--switch-val-frac", "0", "--switch-mix-ratio", ".5",
        "--event-oversample", "2", "--leave-pi-wall-out", "--cav-drive", "teacher",
        "--unroll-s", "10", "--unroll-s", "30", "--unroll-s", "60", "--unroll-s", "120",
        "--stage-steps", "1,1,1,1", "--sumo-fidelity", "--truncation-steps", "20",
        "--entries-v2", "--entry-pad", "8", "--merge-cal", "--merge-cal-params", str(ROOT / "_out/merge_params.json"),
        "--traffic-loss", "--traffic-ramp-start-s", "0", "--traffic-ramp-end-s", "1",
        "--projection", "--residual-cap-mps2", "1.5", "--lane-logit-cap", "3",
        "--stability-weight", ".01", "--conservation-metric", "--conservation-weight", ".01",
        "--loss-share-logging", "--out", str(out / "train")]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "output/wd01/wm_n1"),
               WM_N1_TRAIN_IDENTITIES="", WM_N1_TRAIN_OUTCOMES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
    import psutil

    started, peak = time.monotonic(), 0
    with (out / "cli.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                try:
                    peak = max(peak, psutil.Process(child.pid).memory_info().rss)
                except psutil.NoSuchProcess:
                    break
                if peak >= 6_000_000_000:
                    child.terminate()
                    raise RuntimeError("CPU CLI memory limit exceeded")
                time.sleep(.1)
            rc = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
    sources = [meta / "ACTION_TIMELINE_INDEX_train.json", meta / "ZERO_SHOT_HELDOUT.json"]
    sources += [meta / "identities_switching_train" / name for name in
                ("identities.npy", "global_positions.npy", "vehicle_map.json", "TRAIN_IDENTITY_RECEIPT.json")]
    receipt = {"task_id": "WM-S-TRAINSETUP-20261009", "returncode": rc,
        "command_argv": argv, "peak_sampled_rss_bytes": peak, "seconds": time.monotonic() - started,
        "physical_data": False, "kinematics_and_commands": "synthetic",
        "real_inputs": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (out / "CPU_RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

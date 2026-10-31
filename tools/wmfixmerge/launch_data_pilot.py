"""One-shot v2-DATA dispatch: memory wait -> recovered parity -> amended pilot.

Run on gpuA from the registered root. Exclusive dispatch receipt prevents a
second launcher, including after a crash. No process signals or SUMO calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

from tools.wmfixmerge.data import IMPLEMENTATION_FILES, digest, implementation_hashes, safe
from tools.wmfixmerge.data_pilot import check_binding

ROOT = Path("/root/autodl-tmp/c4/wmfixmerge20261031")
CONFIG = "/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml"
CHECKPOINT = "/root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt"
CONTROLLERS = "/root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json"
NORMALIZED_TREE_SHA = "597fcc80b61c992a2fe9f353604cfd572cbcf8741a9e90eea4a333ffb6fb26fa"
RAW_TREE_SHA = "303830de3670bdef8585863a8b7b8c32bdfff588f29a92dae4e9263231bfac7f"


def tree_binding(root):
    raw = implementation_hashes()
    normalized = {p: hashlib.sha256(safe(root / p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                  for p in IMPLEMENTATION_FILES}
    hash_map = lambda h: hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()
    if hash_map(raw) != RAW_TREE_SHA or hash_map(normalized) != NORMALIZED_TREE_SHA:
        raise ValueError("registered staged implementation changed")
    return dict(implementation_hashes=raw, normalized_tree_sha256=hash_map(normalized),
                raw_tree_sha256=hash_map(raw),
                driver_hashes={n: digest(root / "tools/wmfixmerge" / n) for n in
                               ("pilot_v2.py", "data_pilot.py", "pilot.py", "panel_eval_v2.py",
                                "launch_data_pilot.py")})


def parity(smoke, original):
    if {(r["group"], r["horizon"], r["arm"]) for r in smoke["rows"]} != {
            (40, h, a) for h in (120, 300) for a in (0, 1)} or len(smoke["rows"]) != 4:
        raise ValueError("parity has an incomplete/wrong group")
    actual = {h: statistics.fmean(abs(r["contrast"]) for r in smoke["rows"] if r["horizon"] == h)
              for h in (120, 300)}
    expected = {h: original["panel"][str(h)]["per_group"]["0"] for h in (120, 300)}
    if not all(round(expected[h], 3) == target for h, target in ((120, 449.627), (300, 302.891))):
        raise ValueError("original v1 parity target differs from registered values")
    if actual != expected or smoke["step"] != 0 or smoke["groups"] != [40]:
        raise ValueError(f"parity failed: actual={actual}, expected={expected}")
    return dict(passed=True, actual=actual, expected=expected, exact=True)


def free_memory():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                       "--format=csv,noheader,nounits"], text=True).strip())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise ValueError("run from the registered gpuA root")
    binding_args = argparse.Namespace(smooth_events=False, prepared=str(ROOT / "v2/TRAIN_PREPARED_V2.json"),
        harness_contract=str(ROOT / "v2/HARNESS_CONTRACT_V2.json"), checkpoint=CHECKPOINT)
    binding = dict(**check_binding(binding_args), **tree_binding(ROOT))
    if not args.execute:
        print(json.dumps(binding, indent=2))
        return 0
    out = ROOT / "v2data"
    if not out.is_dir():
        raise ValueError("stage the new v2data directory first")
    with (out / "DISPATCH.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started=time.time(), binding=binding,
                       proc_start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21]), stream, indent=2)

    def status(state, **details):
        doc = dict(state=state, time=time.time(), **details)
        (out / "STATUS.json").write_text(json.dumps(doc, indent=2, allow_nan=False))
        with (out / "EVENTS.jsonl").open("a") as stream:
            stream.write(json.dumps(doc, allow_nan=False) + "\n")
        print(json.dumps(doc, allow_nan=False), flush=True)

    def wait_memory():
        while True:
            free = free_memory()
            if free >= 14 * 1024:
                return free
            status("WAIT_GPU_MEMORY", free_MiB=free, required_MiB=14 * 1024)
            time.sleep(60)

    env = dict(os.environ, PYTHONPATH=f"src:.:{ROOT}/_in/ortdeps", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    common = ["--prepared", binding_args.prepared, "--config", CONFIG,
              "--checkpoint", CHECKPOINT, "--controllers", CONTROLLERS]
    try:
        status("VERIFIED_WAITING", binding=binding)
        wait_memory()
        # Old smoke PID is dead and its receipt absent: preserve it, retry once
        # into an exclusive new directory. No historical output is overwritten.
        smoke_dir = out / "smoke"
        smoke_dir.mkdir()
        old_candidate = ROOT / "train/step_000000.pt"
        cmd = [sys.executable, "-m", "tools.wmfixmerge.panel_eval_v2", *common,
               "--candidate", str(old_candidate), "--out", str(smoke_dir), "--name", "fit973002_v1step0",
               "--groups", "40", "--device", "cuda", "--memory-fraction", "0.08"]
        with (smoke_dir / "smoke.log").open("x") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            status("PARITY_RUNNING", pid=proc.pid, command=cmd)
            code = proc.wait()
        if code:
            raise RuntimeError(f"parity smoke exited {code}")
        smoke_path = smoke_dir / "SHARD_fit973002_v1step0.json"
        original_path = ROOT / "train/PANEL_step0_r0.json"
        result = parity(json.loads(smoke_path.read_text()), json.loads(original_path.read_text()))
        result.update(smoke_sha256=digest(smoke_path), original_sha256=digest(original_path),
                      candidate_sha256=digest(old_candidate), prepared_sha256=digest(binding_args.prepared))
        with (out / "PARITY.json").open("x") as stream:
            json.dump(result, stream, indent=2)
        status("PARITY_PASS", result=result)
        free = wait_memory()
        current = tree_binding(ROOT)
        if current != {k: binding[k] for k in current}:
            raise ValueError("staged source changed while waiting")
        pilot_dir = out / "pilot"
        pilot_dir.mkdir()
        cmd = ["timeout", "--signal=INT", "--kill-after=30s", "21570s", sys.executable,
               "-m", "tools.wmfixmerge.pilot_v2", *common, "--registered-data-pilot",
               "--harness-contract", binding_args.harness_contract, "--out", str(pilot_dir),
               "--device", "cuda", "--memory-fraction", "0.16"]
        with (out / "pilot.log").open("x") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            status("PILOT_RUNNING", pid=proc.pid, command=cmd, free_MiB=free,
                   proc_start_ticks=Path(f"/proc/{proc.pid}/stat").read_text().split()[21])
            code = proc.wait()
        decision_path = pilot_dir / "PILOT_V2.json"
        decision = json.loads(decision_path.read_text()) if decision_path.exists() else None
        final = pilot_dir / "FINAL.json"
        status("FINISHED", exit_code=code, pilot_decision=decision,
               final=json.loads(final.read_text()) if final.exists() else None)
        return code
    except (RuntimeError, ValueError, OSError) as exc:
        status("BLOCKED", blocker=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())

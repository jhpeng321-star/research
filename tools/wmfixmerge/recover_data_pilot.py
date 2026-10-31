"""Explicit v2-DATA infrastructure recovery; preserve updates, gate, and budget."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.wmfixmerge.data import digest, implementation_hashes, safe
from tools.wmfixmerge.data_pilot import CONTRACT_SHA, PREPARED_SHA, WM2_SHA
from tools.wmfixmerge.launch_data_pilot import ROOT, CONFIG, CHECKPOINT, CONTROLLERS

RECOVERY = ROOT / "v2data/recovery_1705"
DRIVERS = ("pilot_v2.py", "data_pilot.py", "pilot.py", "panel_eval_v2.py",
           "entrant_binding.py", "recover_data_pilot.py")


def read(path):
    return json.loads(safe(path).read_text())


def write(path, doc):
    with safe(path).open("x") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)


def validate_recovery(args):
    """Verify a separately registered receipt; never weaken the original route."""
    if not args.data_recovery_sha256 or digest(args.data_recovery) != args.data_recovery_sha256:
        raise ValueError("recovery receipt hash mismatch")
    doc = read(args.data_recovery)
    if doc["schema"] != "wmfix_data_infrastructure_recovery/1" or doc["completed_steps"] != 4:
        raise ValueError("recovery must resume the four completed v2-DATA updates")
    for key in ("prepared", "harness_contract", "checkpoint", "controllers"):
        if digest(getattr(args, key)) != doc["inputs"][key + "_sha256"]:
            raise ValueError("recovery input changed: " + key)
    if doc["inputs"]["prepared_sha256"] != PREPARED_SHA or doc["inputs"]["checkpoint_sha256"] != WM2_SHA:
        raise ValueError("recovery changed registered data/base")
    if doc["implementation_hashes"] != implementation_hashes():
        raise ValueError("recovery implementation changed")
    for name, expected in doc["driver_hashes"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError("recovery driver changed: " + name)
    for path, expected in doc["parents"].items():
        if digest(path) != expected:
            raise ValueError("recovery parent changed: " + path)
    for name, expected in doc["resume_files"].items():
        if digest(safe(doc["previous_out"]) / name) != expected:
            raise ValueError("recovery completed update changed: " + name)
    if not 0 < doc["spent_seconds"] < 21570:
        raise ValueError("invalid spent pilot budget")
    return doc


def prepare():
    """CPU metadata-only preparation; exclusive new outputs, no CUDA allocation."""
    from tools.wmfixmerge.entrant_binding import bind_entrant_specs
    previous = ROOT / "v2data/pilot"
    status = read(ROOT / "v2data/STATUS.json")
    if status["state"] != "FINISHED" or status["pilot_decision"].get("status") != "BLOCKED" \
            or status["pilot_decision"]["completed_steps"] != 4:
        raise ValueError("previous run is not the reconciled infrastructure block")
    parity = read(ROOT / "v2data/PARITY.json")
    if parity.get("passed") is not True or parity.get("exact") is not True:
        raise ValueError("original exact parity did not pass")
    old = read(ROOT / "v2data/DISPATCH.json")["binding"]
    current = implementation_hashes()
    changed = {p for p in current if current[p] != old["implementation_hashes"].get(p)}
    if changed != {"tools/wmfixmerge/runtime.py", "tools/wmfixmerge/data.py",
                   "tools/wmfixmerge/entrant_binding.py"}:
        raise ValueError("only recorded-policy binding and its implementation hash list may change")
    prepared = ROOT / "v2/TRAIN_PREPARED_V2.json"
    old_contract = ROOT / "v2/HARNESS_CONTRACT_V2.json"
    if digest(prepared) != PREPARED_SHA or digest(old_contract) != CONTRACT_SHA:
        raise ValueError("registered data/parent contract changed")
    specs = read(CONTROLLERS)
    specs = specs.get("controllers", specs)
    speed = read("/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json")
    if digest(speed["path"]) != speed["sha256"] or speed["sha256"] != \
            "d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca":
        raise ValueError("WMDATA speed-MAPPO checkpoint mismatch")
    specs["mappo"] = dict(checkpoint=speed["path"], sha256=speed["sha256"])
    bound = bind_entrant_specs(specs, read(prepared)["episodes"])
    for spec in bound.values():
        if digest(spec["checkpoint"]) != spec["sha256"]:
            raise ValueError("recorded entrant policy bytes mismatch")
    if not RECOVERY.parent.is_dir():
        raise ValueError("recovery parent missing")
    RECOVERY.mkdir()
    controllers = RECOVERY / "CONTROLLERS.json"
    write(controllers, dict(controllers=bound))
    contract = read(old_contract)
    contract["runtime_sha256"] = digest(ROOT / "tools/wmfixmerge/runtime.py")
    new_contract = RECOVERY / "HARNESS_CONTRACT.json"
    write(new_contract, contract)
    events = [json.loads(line) for line in (ROOT / "v2data/EVENTS.jsonl").read_text().splitlines()]
    launched = next(e["time"] for e in events if e["state"] == "PILOT_RUNNING")
    finished = next(e["time"] for e in events if e["state"] == "FINISHED")
    files = [f"{kind}_{s:06d}.pt" for s in range(5) for kind in ("step", "optimizer")]
    files += [f"update_{s:06d}.json" for s in range(1, 5)]
    for s in range(1, 5):
        row = read(previous / f"update_{s:06d}.json")
        if row["step"] != s or not row["finite_gradients"] or not row["frozen_bitwise"]:
            raise ValueError("prior completed update is invalid")
    parents = [ROOT / "v2data" / name for name in ("DISPATCH.json", "STATUS.json", "PARITY.json")]
    parents += [old_contract, previous / "PROTOCOL.json", previous / "PILOT_V2.json"]
    doc = dict(schema="wmfix_data_infrastructure_recovery/1", previous_out=str(previous),
               completed_steps=4, spent_seconds=finished - launched,
               inputs={key + "_sha256": digest(path) for key, path in
                       (("prepared", prepared), ("harness_contract", new_contract),
                        ("checkpoint", CHECKPOINT), ("controllers", controllers))},
               parents={str(p): digest(p) for p in parents},
               resume_files={name: digest(previous / name) for name in files},
               implementation_hashes=current,
               driver_hashes={name: digest(Path(__file__).parent / name) for name in DRIVERS},
               amendment="Infrastructure only: hash-matched CTRL0 alias + recorded speed MAPPO; "
                         "resume exact step4/Adam; unchanged gate/objective/CRN/fit exclusion; spent budget retained")
    target = RECOVERY / "RECOVERY.json"
    write(target, doc)
    print(json.dumps(dict(receipt=str(target), sha256=digest(target), **doc), indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt")
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    if Path.cwd().resolve() != ROOT:
        raise ValueError("run from the registered gpuA root")
    if args.prepare:
        prepare()
        return 0
    bound = argparse.Namespace(data_recovery=args.receipt, data_recovery_sha256=args.receipt_sha256,
        prepared=str(ROOT / "v2/TRAIN_PREPARED_V2.json"), checkpoint=CHECKPOINT,
        controllers=str(RECOVERY / "CONTROLLERS.json"),
        harness_contract=str(RECOVERY / "HARNESS_CONTRACT.json"))
    doc = validate_recovery(bound)
    if not args.execute:
        print(json.dumps(doc, indent=2))
        return 0
    import fcntl
    lock = (ROOT / "v2data/RECOVERY.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(RECOVERY / "DISPATCH.json", dict(pid=os.getpid(), started=time.time(),
          proc_start_ticks=Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
          recovery_sha256=args.receipt_sha256))

    def status(state, **details):
        row = dict(state=state, time=time.time(), **details)
        (RECOVERY / "STATUS.json").write_text(json.dumps(row, indent=2, allow_nan=False))
        with (RECOVERY / "EVENTS.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    while True:
        free = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                   "--format=csv,noheader,nounits"], text=True).strip())
        if free >= 14336:
            break
        status("WAIT_GPU_MEMORY", free_MiB=free, required_MiB=14336)
        time.sleep(60)
    validate_recovery(bound)
    out = RECOVERY / "pilot"
    out.mkdir()
    budget = int(21570 - doc["spent_seconds"])
    cmd = ["timeout", "--signal=INT", "--kill-after=30s", str(budget) + "s", sys.executable,
           "-m", "tools.wmfixmerge.pilot_v2", "--registered-data-pilot", "--device", "cuda",
           "--memory-fraction", "0.16", "--out", str(out), "--prepared", bound.prepared,
           "--checkpoint", bound.checkpoint, "--config", CONFIG, "--controllers", bound.controllers,
           "--harness-contract", bound.harness_contract, "--data-recovery", bound.data_recovery,
           "--data-recovery-sha256", bound.data_recovery_sha256]
    env = dict(os.environ, PYTHONPATH=f"src:.:{ROOT}/_in/ortdeps", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with (RECOVERY / "pilot.log").open("x") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        status("PILOT_RUNNING", pid=proc.pid, command=cmd, free_MiB=free, budget_seconds=budget)
        seen = set()
        while proc.poll() is None:
            for name in ("update_000010.json", "FULL_PANEL_STEP0.json", "FULL_PANEL_STEP10.json", "PILOT_V2.json", "FINAL.json"):
                path = out / name
                if path.exists() and name not in seen:
                    try:
                        row = read(path)
                    except json.JSONDecodeError:
                        continue
                    status("MILESTONE", name=name, sha256=digest(path), receipt=row)
                    seen.add(name)
            time.sleep(60)
    status("FINISHED", exit_code=proc.returncode,
           decision=read(out / "PILOT_V2.json") if (out / "PILOT_V2.json").exists() else None,
           final=read(out / "FINAL.json") if (out / "FINAL.json").exists() else None)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

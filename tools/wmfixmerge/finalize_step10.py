"""MAIN21:32 partial-result stop; verify full panels and signal owned jobs only."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import time

from tools.wmfixmerge.data import digest
from tools.wmfixmerge.data_pilot import PREPARED_SHA, WM2_SHA
from tools.wmfixmerge.gate_only_watch import (
    OUT, RECOVERY_SHA, interrupt_owned, open_pidfd, _syscall,
)
from tools.wmfixmerge.pilot_v2 import assemble

STEP10_SHA = "8bbd3a6d518e752d7d30d06352abbf413e91ac795a82f890b8ef241443b951e2"


def verify_panel(out, step):
    docs = []
    for i in range(4):
        doc = json.loads((out / f"SHARD_step{step}_full_s{i}.json").read_text())
        groups = list(range(i * 13, (i + 1) * 13))
        producer = doc["producer"]
        if doc["step"] != step or doc["groups"] != groups or \
                producer["prepared_sha256"] != PREPARED_SHA or \
                producer["base_sha256"] != WM2_SHA or \
                producer["candidate_sha256"] != digest(out / f"step_{step:06d}.pt"):
            raise ValueError("panel binding mismatch")
        cells = [(r["group"], r["horizon"], r["arm"]) for r in doc["rows"]]
        if len(cells) != 52 or set(cells) != {(g, h, a) for g in groups
                                                for h in (120, 300) for a in (0, 1)}:
            raise ValueError("missing/duplicate panel cell")
        if not all(math.isfinite(r["contrast"]) for r in doc["rows"]):
            raise ValueError("nonfinite panel")
        docs.append(doc)
    panel = assemble([doc["rows"] for doc in docs], range(52))
    if panel != json.loads((out / f"FULL_PANEL_STEP{step}.json").read_text()):
        raise ValueError("published panel differs from reassembly")
    return {h: panel[str(h)]["paired_J_MAE"] for h in (120, 300)}


def later_worker(argv):
    """Exact registered command identity; never match peer or other output jobs."""
    try:
        return "tools.wmfixmerge.panel_eval_v2" in argv and \
            argv[argv.index("--out") + 1] == str(OUT) and \
            argv[argv.index("--candidate") + 1] in {
                str(OUT / f"step_{s:06d}.pt") for s in (20, 30)} and \
            argv[argv.index("--name") + 1] in {
                f"step{s}_full_s{i}" for s in (20, 30) for i in range(4)}
    except (ValueError, IndexError):
        return False


def identity(pid):
    p = Path("/proc") / str(pid)
    return ((p / "stat").read_text().rsplit(")", 1)[1].split()[19],
            (p / "cmdline").read_bytes().decode().strip("\0").split("\0"))


def stop_worker(pid, ticks, argv):
    fd = open_pidfd(pid)
    try:
        if identity(pid) != (ticks, argv) or not later_worker(argv):
            raise ValueError("worker identity changed")
        fn = getattr(signal, "pidfd_send_signal", None)
        if fn:
            fn(fd, signal.SIGINT)
        else:
            _syscall(424, fd, signal.SIGINT, 0, 0)
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if digest(OUT.parent / "RECOVERY.json") != RECOVERY_SHA or \
            digest(OUT / "step_000010.pt") != STEP10_SHA:
        raise ValueError("registered recovery/checkpoint changed")
    baseline, step10 = verify_panel(OUT, 0), verify_panel(OUT, 10)
    both = all(step10[h] < baseline[h] for h in (120, 300))
    if both:
        print(json.dumps(dict(action="REQUEST_DEADLINE_EXTENSION_AND_FULL_GATE",
                              baseline=baseline, step10=step10)))
        return 0
    files = [OUT / f"FULL_PANEL_STEP{s}.json" for s in (0, 10)]
    files += [OUT / f"SHARD_step{s}_full_s{i}.json" for s in (0, 10) for i in range(4)]
    updates = [json.loads((OUT / f"update_{s:06d}.json").read_text()) for s in range(1, 31)]
    result = dict(authority="REGISTRATION.md MAIN21:32 2026-10-31",
                  status="STOP_STEP10_PARTIAL_RESULT", full_gate_evaluated=False,
                  note="step10 failed; steps 20/30 not scored", baseline=baseline, step10=step10,
                  delta={h: step10[h] - baseline[h] for h in (120, 300)},
                  finite_gradients=all(r["finite_gradients"] for r in updates),
                  frozen_bitwise=all(r["frozen_bitwise"] for r in updates),
                  sha256={str(p): digest(p) for p in files},
                  source_sha256=digest(__file__), time=time.time(), stopped_workers=[])
    if not args.execute:
        print(json.dumps(result, indent=2))
        return 0
    target = OUT.parent / "MAIN2132_STOP.json"
    # Exclusive intent before side effects; never overwrite an earlier stop receipt.
    with (OUT.parent / "MAIN2132_STOP_INTENT.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    result["trainer_sigint_sent"] = interrupt_owned(223222, 881862702)
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            ticks, cmd = identity(int(p.name))
            if later_worker(cmd):
                stop_worker(int(p.name), ticks, cmd)
                result["stopped_workers"].append(dict(pid=int(p.name), start_ticks=ticks, argv=cmd))
        except (FileNotFoundError, ProcessLookupError):
            continue
    result["completed_later_panels"] = [p.name for s in (20, 30)
        for p in OUT.glob(f"SHARD_step{s}_full_s*.json")]
    result["extra_updates"] = [p.name for p in OUT.glob("update_*.json")
                                if int(p.stem.split("_")[1]) > 30]
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

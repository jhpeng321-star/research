"""MAIN17:52: finish the registered gate, then stop owned pilot continuation.

External observer only: no changes to the running, hash-bound scoring/training
tree; no CUDA allocation. Identity-check the exact owned trainer before SIGINT.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
import time

from tools.wmfixmerge.data import digest, safe
from tools.wmfixmerge.launch_data_pilot import ROOT

OUT = ROOT / "v2data/recovery_1705/pilot"
STEP30_SHA = "aa8df951643bb6eec3d5a341687db407d1d28aa1cd28f8c91851c67c6d7f0120"
RECOVERY_SHA = "056c295c441bff291ec3548a47be2b1c0dffd00c085e53a51e0dd5f59bca0d10"


def owned(pid, ticks, *, proc=Path("/proc")):
    try:
        path = proc / str(pid)
        start = (path / "stat").read_text().rsplit(")", 1)[1].split()[19]
        argv = (path / "cmdline").read_bytes().decode().split("\0")
    except (OSError, IndexError, UnicodeDecodeError):
        return False
    try:
        return start == str(ticks) and "tools.wmfixmerge.pilot_v2" in argv \
            and "--registered-data-pilot" in argv and "--out" in argv \
            and argv[argv.index("--out") + 1] == str(OUT) \
            and "--data-recovery-sha256" in argv \
            and argv[argv.index("--data-recovery-sha256") + 1] == RECOVERY_SHA
    except IndexError:
        return False


def _syscall(number, *args):
    # Some existing conda Python builds lack the pidfd wrappers despite running
    # on a kernel that supports them. Linux's generic 64-bit syscall numbers
    # are identical on the two supported architectures; fail closed elsewhere.
    import ctypes
    import platform
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise OSError("pidfd syscall fallback unsupported on this platform")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *(ctypes.c_long(int(v)) for v in args))
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return int(result)


def open_pidfd(pid):
    fn = getattr(os, "pidfd_open", None)
    return fn(pid) if fn else _syscall(434, pid, 0)


def send_pidfd(fd):
    fn = getattr(signal, "pidfd_send_signal", None)
    return fn(fd, signal.SIGINT) if fn else _syscall(424, fd, signal.SIGINT, 0, 0)


def interrupt_owned(pid, ticks):
    """A pidfd pins identity across the final check/signal race (Linux only)."""
    try:
        fd = open_pidfd(pid)
    except ProcessLookupError:
        return False
    try:
        if not owned(pid, ticks):
            return False
        try:
            send_pidfd(fd)
        except ProcessLookupError:
            return False
        return True
    finally:
        os.close(fd)


def check_gate(doc):
    # The failure/block path never continues; preserve its ordinary exit.
    if doc.get("passed") is not True:
        return False
    if doc.get("completed_steps") != 30 or doc.get("protocol", {}).get("recovery_sha256") != RECOVERY_SHA:
        raise ValueError("PASS receipt is not the owned registered thirty-step gate")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", required=True)
    args = parser.parse_args(argv)
    if not owned(args.pid, args.start_ticks) or digest(OUT / "step_000030.pt") != STEP30_SHA:
        raise ValueError("owned trainer/checkpoint identity mismatch")
    fd = open_pidfd(args.pid)  # prove kernel support before arming; never signal in dry-run
    os.close(fd)
    if not args.execute:
        print("owned trainer and step30 verified; dry-run only")
        return 0
    receipt = dict(authority="REGISTRATION.md MAIN17:52: Track1 finishes registered30-step gate ONLY",
                   watcher_pid=os.getpid(), trainer_pid=args.pid, trainer_start_ticks=args.start_ticks,
                   started=time.time(), watcher_sha256=digest(__file__), step30_sha256=STEP30_SHA)
    parent = OUT.parent
    with safe(parent / "GATE_ONLY_WATCH.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    decision = OUT / "PILOT_V2.json"
    while True:
        if decision.exists():
            try:
                doc = json.loads(decision.read_text())
            except json.JSONDecodeError:
                time.sleep(.1)
                continue
            stop = check_gate(doc)
            sent = False
            if stop:
                sent = interrupt_owned(args.pid, args.start_ticks)  # panel drain is complete
            result = dict(time=time.time(), decision_sha256=digest(decision), passed=doc.get("passed"),
                          sigint_sent=sent, source="MAIN17:52 gate-only; no continuation authorised",
                          extra_updates=[p.name for p in OUT.glob("update_*.json")
                                         if int(p.stem.split("_")[1]) > 30])
            with (parent / "GATE_ONLY_RESULT.json").open("x") as stream:
                json.dump(result, stream, indent=2)
            print(json.dumps(result), flush=True)
            return 0
        if not owned(args.pid, args.start_ticks):
            with (parent / "GATE_ONLY_RESULT.json").open("x") as stream:
                json.dump(dict(time=time.time(), state="TRAINER_EXITED_BEFORE_GATE", sigint_sent=False), stream)
            return 2
        time.sleep(.2)


if __name__ == "__main__":
    raise SystemExit(main())

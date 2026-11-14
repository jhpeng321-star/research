#!/usr/bin/env python3
"""run_queue.py - run pack_wt.py over all remaining non-active worktrees, <=3 in parallel.

Order: worker 0 takes the LARGEST remaining worktree first (long, bandwidth-bound), workers 1..N-1 take the
smallest first (clears the count quickly).  Skips: active, receipt ok, locked by a live worker, or currently
handled by the older per-file uploader (process.ps1 / its rclone), which this script never touches.
Usage: python -I run_queue.py [--workers 3] [--until 09:20] [--only a,b]
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
BASE = r"E:\rd-wt\reorg3\wtclean"
PY = r"C:\Users\jpeng\miniconda3\envs\rl_driving\python.exe"
PACK = BASE + r"\tools\pack_wt.py"
INV = BASE + r"\inventory\inventory.json"
RECDIR = BASE + r"\receipts"
LOCKDIR = BASE + r"\locks"
PROGRESS = BASE + r"\wt_progress.log"


def now():
    return datetime.datetime.now()


def log(msg):
    line = "%s   [queue] %s" % (now().strftime("%Y-%m-%dT%H:%M:%S"), msg)
    try:
        with open(PROGRESS, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def receipt_ok(name):
    p = os.path.join(RECDIR, name + ".json")
    if not os.path.exists(p):
        return False
    try:
        r = json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        return False
    if r.get("status") == "ok":
        return True
    # legacy per-file receipt that the clean-up template also accepts (commit_failed but upload+check exit 0)
    return (r.get("status") == "commit_failed" and (r.get("upload") or {}).get("exit") == 0
            and (r.get("check") or {}).get("exit") == 0 and r.get("method") is None)


def procs():
    """[(pid, name, cmdline)] for rclone/powershell processes (read-only query)."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'rclone|powershell' } | "
          "ForEach-Object { '{0}|{1}|{2}' -f $_.ProcessId,$_.Name,$_.CommandLine }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60)
    except Exception:
        return []
    out = []
    for l in r.stdout.splitlines():
        p = l.split("|", 2)
        if len(p) == 3:
            out.append((p[0], p[1], p[2]))
    return out


def old_state(names):
    """(old_script_alive, set of names an OTHER process is working on right now)."""
    alive, busy = False, set()
    for pid, nm, cmd in procs():
        lc = cmd.lower()
        if nm.lower().startswith("powershell") and "tools\\process.ps1" in lc:
            alive = True
        if nm.lower().startswith("rclone"):
            for n in names:
                if ("worktrees/%s " % n).lower() in lc or ("worktrees/%s\"" % n).lower() in lc \
                        or lc.rstrip().endswith("worktrees/%s" % n.lower()) \
                        or ("e:\\rd-wt\\%s " % n).lower() in lc:
                    busy.add(n)
    return alive, busy


def locked(name):
    p = os.path.join(LOCKDIR, name + ".lock")
    return os.path.exists(p)


def live_locks():
    """names whose pack_wt worker (lock file with a live pid) is running, from any queue instance"""
    out = set()
    try:
        names = [f[:-5] for f in os.listdir(LOCKDIR) if f.endswith(".lock")]
    except OSError:
        return out
    for n in names:
        try:
            pid = int(json.load(open(os.path.join(LOCKDIR, n + ".lock"))).get("pid", 0))
            r = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"], capture_output=True, text=True, timeout=30)
            if str(pid) in r.stdout:
                out.add(n)
        except Exception:
            out.add(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--until", default="09:20")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    hh, mm = [int(x) for x in a.until.split(":")]
    inv = [t for t in json.load(open(INV, encoding="utf-8-sig")) if not t.get("active")]
    inv.sort(key=lambda t: (t["uploadBytes"], t["name"]))
    order = [t["name"] for t in inv]
    if a.only:
        keep = set(a.only.split(","))
        order = [n for n in order if n in keep]
    attempts = {}
    sizes = {t["name"]: t["uploadBytes"] for t in inv}
    inprog = set()
    lk = threading.Lock()
    log("queue start: %d candidates, workers=%d, until %s" % (len(order), a.workers, a.until))

    def pick(big):
        alive, busy = old_state(order)
        cand = [n for n in order if not receipt_ok(n) and n not in inprog and not locked(n) and n not in busy
                and attempts.get(n, 0) < 2]
        if not cand:
            return None
        protect = set()
        if alive:
            # the old per-file uploader walks the same ascending-bytes order and takes the first non-ok item:
            # stay K positions ahead of it (items I am processing or that it is uploading do not count).
            allo = [n for n in order if not receipt_ok(n) and n not in inprog and not locked(n) and n not in busy]
            protect = set(allo[:3])
        free = [n for n in cand if n not in protect] or cand
        return free[-1] if big else free[0]

    def worker(i):
        big = (i == 0)
        while True:
            t = now()
            if (t.hour, t.minute) >= (hh, mm) and t.hour < 20:
                log("worker %d: deadline reached, no new starts" % i)
                return
            n = None
            with lk:
                run_now = live_locks() | inprog
                if len(run_now) >= a.workers:
                    n = ""  # no slot
                elif big and any(sizes.get(x, 0) > 1e9 for x in run_now):
                    n = ""  # a big item is already running: do not start a second one
                else:
                    n = pick(big)
                    if n is None:
                        log("worker %d: nothing left" % i)
                        return
                    inprog.add(n)
                    attempts[n] = attempts.get(n, 0) + 1
            if n == "":
                time.sleep(15)
                continue
            log("worker %d -> %s (attempt %d)" % (i, n, attempts[n]))
            outf = os.path.join(BASE, "inventory", "logs", n + ".pack.out")
            with open(outf, "ab") as fo:
                r = subprocess.run([PY, "-I", PACK, n], stdout=fo, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            log("worker %d <- %s rc=%d" % (i, n, r.returncode))
            if r.returncode not in (0,):
                try:
                    tail = open(outf, "rb").read()[-600:].decode("utf-8", "replace").strip().splitlines()[-3:]
                except OSError:
                    tail = []
                log("   %s tail: %s" % (n, " | ".join(tail)[:300]))
            with lk:
                inprog.discard(n)
            time.sleep(1)

    ths = []
    for i in range(a.workers):
        th = threading.Thread(target=worker, args=(i,), daemon=False)
        th.start()
        ths.append(th)
        time.sleep(3)
    for th in ths:
        th.join()
    log("queue done")


if __name__ == "__main__":
    main()

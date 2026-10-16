#!/usr/bin/env python3
"""C4 Dreamer-loop orchestrator.

Iteration k (state persisted in ``<loop_root>/state.json`` after every transition, so a crash or a
restart resumes at the right step; every action is appended to ``<loop_root>/loop_log.jsonl``):

  snapshot        copy the policy's newest complete checkpoint into ``iter_<k>/policy_snapshot.pt``
  collect_launch  allocate the next seeds of the registered C4_V6_LONGPAIR training block
                  (965184-965999 only), push the snapshot to the CPU box, start the SUMO collection
                  detached (``traffic_shepherd.c4.collect``)
  collect_wait    poll it (resumable collect: a failed attempt is relaunched into the same directory)
  pull            rsync the raw episodes back to ``<loop_root>/raw/iter_<k>`` (kept for good)
  convert         convert the policy episodes so far into a fresh ``<data_root>/<prefix><k>`` dir;
                  superseded converted dirs created by this loop are deleted (raw stays)
  wm_launch       short WM refresh (``python -m wms train``) from the current WM checkpoint
  wm_wait         poll it
  wm_check        finite training losses; final validation not worse than the reference by > tol.
                  A rejected WM is logged and the previous WM is kept (the policy is not restarted)
  policy_stop     stop the policy cleanly: STOP file (trainers launched by this loop) or, for an
                  explicitly adopted trainer without that feature, wait for its next periodic
                  checkpoint, then SIGTERM exactly that PID
  policy_start    restart the policy with ``WMS_LOOP_REFRESH=1 --resume`` on the new WM in a new dir

The policy keeps training on the old WM during every step but policy_stop/policy_start.

Usage (on the GPU host, from a deployed code root; stdlib only):
  python scripts/c4/dreamer_loop.py run --config CFG [--init-wm CKPT --init-wm-val-log LOG
                                         --init-policy-run-dir DIR --adopt-policy-pid PID]
  python scripts/c4/dreamer_loop.py status  --config CFG [--live]
  python scripts/c4/dreamer_loop.py dry-run --config CFG [same init flags]
  python scripts/c4/dreamer_loop.py stop    --config CFG      (= touch <loop_root>/STOP)

Safety: only kills processes it started (PID + /proc start time + command-line marker are checked
before every signal; launched jobs get their own session, the process group is signalled) or the one
PID given explicitly with --adopt-policy-pid; never pattern-kills; never reuses an output dir; pauses
while free disk < ``min_free_gb``; refuses seeds outside 965184-965999 and any restricted path name.
See scripts/c4/LOOP_ORCHESTRATOR.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

#: Registered policy-rollout seeds (unused part of the C4_V6_LONGPAIR training block).
SEED_BLOCK = (965184, 969999)  # C4_V6_LONGPAIR rest (965184-965999) + C4_LOOP_TRAIN (966000-969999), contiguous; both registered
#: C4 v5 evaluation seeds: never used for training.
EVAL_SEEDS = (964000, 964199)
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
STEPS = ("snapshot", "collect_launch", "collect_wait", "pull", "convert", "wm_launch", "wm_wait",
         "wm_check", "policy_stop", "policy_start", "done")
#: Steps that start new work (the disk guard runs before them).
LAUNCH_STEPS = {"snapshot", "collect_launch", "pull", "convert", "wm_launch", "policy_start"}

DEFAULTS = {
    "mode": "pipelined",          # or "sequential" (the simple turn-based fallback)
    "max_iterations": 8,
    "min_free_gb": 5.0,
    "poll_s": 60,
    "ssh_retries": 4,
    "max_step_errors": 5,
    "seeds": {"per_iteration": 100, "exclude": []},
    "collect": {"workers": 46, "warmup": 300, "horizon": 900, "population": "textbook",
                "case_plan": "weighted-cycle", "penetrations": "0.05,0.1,0.2,0.4",
                "pen_weights": "1,1,2,2", "demands": "1900,2100,2400,2700", "timeout_s": 7200,
                "max_attempts": 3, "min_complete_fraction": 0.5, "wait_for_absent": [],
                "hosts": ["cpu"], "batch_size": 50, "require_new_policy_ckpt": True, "min_interval_s": 0,
                "nice": 0},
    "convert": {"prefix": "v6_policy_iter", "max_episodes": 0, "mb_per_episode": 21.0,
                "timeout_s": 3600, "nice": 10, "min_window": 50},
    "wm": {"config": "configs/c4_wm_v6_refresh.toml", "stage_steps": [60, 40], "extra_set": [],
           "min_new_episodes": 0, "max_refreshes": 1000, "convert_timeout_s": 3600,
           "env": {}, "timeout_s": 3 * 3600, "val_metric": "mean_position", "val_tolerance": 0.30,
           "lane_calib": False, "lane_calib_config": "configs/c4_wm_v6_full.toml",
           "lane_calib_timeout_s": 1500, "nice": 0},
    "policy": {"config": "configs/c4_policy_v6_loop.toml", "initial_data_dir": None,
               "config_without_policy_data": "configs/c4_policy_v6_full.toml", "device": "cuda", "max_updates": 100000,
               "extra_args": [], "env": {}, "allowed_roots": [], "stop_timeout_s": 1800,
               "adopted_wait_max_s": 3600, "start_check_s": 90, "max_restarts": 2},
}


# --------------------------------------------------------------------------- helpers
def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def q(text) -> str:
    return shlex.quote(str(text))


def refuse_restricted(path) -> str:
    text = str(path)
    if any(token in text.lower() for token in FORBIDDEN):
        raise ValueError(f"restricted path refused: {text}")
    return text


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path) -> dict:
    return load_config_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_config_dict(raw: dict) -> dict:
    cfg = merge(DEFAULTS, raw)
    for key in ("loop_root", "data_root", "wm_runs_root"):
        if not cfg.get(key):
            raise ValueError(f"config needs {key}")
    if cfg["mode"] not in ("pipelined", "sequential"):
        raise ValueError("mode must be 'pipelined' or 'sequential'")
    collectors = list(cfg["collect"]["hosts"]) if cfg["mode"] == "pipelined" else ["cpu"]
    if not collectors or len(set(collectors)) != len(collectors):
        raise ValueError("collect.hosts must list distinct CPU host names")
    for name in ("wm", "policy", *collectors):
        if name not in cfg.get("hosts", {}):
            raise ValueError(f"config needs hosts.{name}")
        host = cfg["hosts"][name]
        for key in ("code_root", "python"):
            if not host.get(key):
                raise ValueError(f"hosts.{name} needs {key}")
    if cfg["hosts"]["wm"].get("ssh"):
        raise ValueError("hosts.wm must be the local host (the orchestrator runs where the WM trains)")
    for name in ("policy", *collectors):
        if not cfg["hosts"][name].get("work_root"):
            raise ValueError(f"hosts.{name} needs work_root")
    if cfg["collect"]["batch_size"] < 1:
        raise ValueError("collect.batch_size must be positive")
    lo, hi = cfg["seeds"].get("block", list(SEED_BLOCK))
    if lo < SEED_BLOCK[0] or hi > SEED_BLOCK[1] or lo > hi:
        raise ValueError(f"seed block {lo}-{hi} is outside the approved {SEED_BLOCK[0]}-{SEED_BLOCK[1]}")
    cfg["seeds"]["block"] = [lo, hi]
    if cfg["seeds"]["per_iteration"] < 1:
        raise ValueError("seeds.per_iteration must be positive")
    for path in _all_strings(cfg):
        refuse_restricted(path)
    return cfg


def _all_strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _all_strings(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _all_strings(value)


def ranges(seeds: list[int]) -> list[list[int]]:
    out: list[list[int]] = []
    for seed in sorted(seeds):
        if out and seed == out[-1][1] + 1:
            out[-1][1] = seed
        else:
            out.append([seed, seed])
    return out


def seed_spec(seeds: list[int]) -> str:
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges(seeds))


def expand(rs) -> set[int]:
    return {s for a, b in rs for s in range(int(a), int(b) + 1)}


def check_seeds(seeds) -> None:
    for seed in seeds:
        if type(seed) is not int or not SEED_BLOCK[0] <= seed <= SEED_BLOCK[1]:
            raise ValueError(f"seed {seed!r} outside the approved block {SEED_BLOCK}")
        if EVAL_SEEDS[0] <= seed <= EVAL_SEEDS[1]:
            raise ValueError(f"seed {seed} is an evaluation seed")


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path) -> list[dict]:
    rows = []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass            # a torn last line of a live log
    return rows


def last_val_event(rows: list[dict]) -> dict[float, dict]:
    """``{horizon_s: row}`` of the last (max step) validation event, revalidations excluded."""
    rows = [r for r in rows if not r.get("revalidated") and "step" in r and "horizon_s" in r]
    if not rows:
        return {}
    last = max(r["step"] for r in rows)
    return {float(r["horizon_s"]): r for r in rows if r["step"] == last}


def wm_sanity(train_log, val_log, ref_val_log, metric: str, tolerance: float) -> dict:
    """Finite training losses + final validation ``metric`` within (1 + tolerance) x reference."""
    out = {"ok": True, "reasons": [], "compare": {}}
    train = read_jsonl(train_log)
    if not train:
        out["reasons"].append("empty train_log")
    for row in train:
        value = row.get("loss")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            out["reasons"].append(f"nonfinite loss at step {row.get('step')}")
            break
    new = last_val_event(read_jsonl(val_log))
    if not new:
        out["reasons"].append("no validation rows")
    for h, row in new.items():
        value = row.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            out["reasons"].append(f"nonfinite val {metric} at {h:g}s")
    ref = last_val_event(read_jsonl(ref_val_log)) if ref_val_log else {}
    if not ref:
        out["note"] = "no reference validation; finiteness checks only"
    for h in sorted(set(new) & set(ref)):
        a, b = new[h].get(metric), ref[h].get(metric)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (a, b)) or b <= 0:
            continue
        ratio = a / b
        out["compare"][f"{h:g}"] = {"new": a, "ref": b, "ratio": ratio}
        if ratio > 1.0 + tolerance:
            out["reasons"].append(f"val {metric} at {h:g}s worse by {ratio:.3f}x (> {1 + tolerance:.2f})")
    out["ok"] = not out["reasons"]
    return out


# --------------------------------------------------------------------------- real backend
class Backend:
    """Shell/ssh/rsync primitives. ``host`` is a hosts.<name> dict; no ``ssh`` key = local."""

    def __init__(self, retries: int = 4, log=None):
        self.retries = retries
        self.log = log or (lambda *a, **k: None)

    # -- basic
    def _argv(self, host: dict, cmd: str) -> list[str]:
        if host.get("ssh"):
            return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
                    "-o", "ServerAliveCountMax=4", *host["ssh"], cmd]
        return ["bash", "-c", cmd]

    def run(self, host: dict, cmd: str, timeout: float = 300, check: bool = False):
        """(rc, stdout, stderr); ssh transport failures (rc 255) are retried with backoff."""
        attempts = self.retries if host.get("ssh") else 1
        for attempt in range(1, attempts + 1):
            try:
                proc = subprocess.run(self._argv(host, cmd), capture_output=True, text=True,
                                      timeout=timeout, stdin=subprocess.DEVNULL)
                rc, out, err = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired:
                rc, out, err = 124, "", f"timeout after {timeout}s"
            if rc != 255 or attempt == attempts:
                break
            self.log("ssh_retry", host=host.get("name"), attempt=attempt, err=err.strip()[-300:])
            time.sleep(min(60, 5 * attempt * attempt))
        if check and rc != 0:
            raise RuntimeError(f"command failed rc={rc} on {host.get('name')}: {cmd[:300]} :: {err.strip()[-500:]}")
        return rc, out, err

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def now(self) -> float:
        return time.time()

    def free_gb(self, path) -> float:
        return shutil.disk_usage(path).free / 1e9

    # -- files on a host
    def exists(self, host, path) -> bool:
        return self.run(host, f"test -e {q(path)}")[0] == 0

    def read_text(self, host, path, tail_bytes: int = 0) -> str | None:
        cmd = f"tail -c {int(tail_bytes)} {q(path)}" if tail_bytes else f"cat {q(path)}"
        rc, out, _ = self.run(host, cmd + " 2>/dev/null")
        return out if rc == 0 else None

    def list_ckpts(self, host, run_dir) -> list[tuple[float, int, str]]:
        """``(mtime, size, path)`` of ``policy_*.pt`` in ``run_dir``, newest first."""
        rc, out, _ = self.run(host, f"cd {q(run_dir)} 2>/dev/null && stat -c '%Y %s %n' policy_*.pt 2>/dev/null")
        rows = []
        for line in out.splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3 and parts[0].isdigit():
                rows.append((float(parts[0]), int(parts[1]), f"{run_dir.rstrip('/')}/{parts[2]}"))
        return sorted(rows, reverse=True)

    def copy(self, src_host, src, dst_host, dst) -> None:
        """Copy a file or a directory (trailing ``/`` = contents) between hosts; one side is local."""
        if src_host.get("ssh") and dst_host.get("ssh"):
            raise ValueError("copy needs one local side (route through the orchestrator host)")
        if not src_host.get("ssh") and not dst_host.get("ssh"):
            rc, _, err = self.run(src_host, f"mkdir -p {q(os.path.dirname(dst.rstrip('/')))} && "
                                            f"cp -a {q(src)} {q(dst)}", timeout=3600)
            if rc:
                raise RuntimeError(f"local copy failed: {err[-300:]}")
            return
        remote = src_host if src_host.get("ssh") else dst_host
        ssh = " ".join(q(x) for x in ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                                       "-o", "ServerAliveInterval=30", *remote["ssh"][:-1]])
        target = remote["ssh"][-1]
        parent = os.path.dirname(dst.rstrip("/"))
        if dst_host.get("ssh"):
            self.run(dst_host, f"mkdir -p {q(parent)}", check=True)
            argv = ["rsync", "-az", "--partial", "--timeout=180", "--exclude", "*.incomplete-*", "-e", ssh,
                    src, f"{target}:{dst}"]
        else:
            os.makedirs(parent, exist_ok=True)
            argv = ["rsync", "-az", "--partial", "--timeout=180", "--exclude", "*.incomplete-*", "-e", ssh,
                    f"{target}:{src}", dst]
        for attempt in range(1, self.retries + 2):
            proc = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL)
            if proc.returncode == 0:
                return
            self.log("rsync_retry", attempt=attempt, rc=proc.returncode, err=proc.stderr.strip()[-300:])
            time.sleep(min(120, 10 * attempt * attempt))
        raise RuntimeError(f"rsync failed after retries: {' '.join(argv[-2:])}")

    # -- processes
    def launch(self, host, cwd, cmd: str, log_path, exit_path, env: dict) -> dict:
        """Start ``cmd`` detached in its own session; returns ``{"pid", "start"}``."""
        exports = " ".join(f"{k}={q(v)}" for k, v in sorted(env.items()))
        inner = f"{cmd}; echo $? > {q(exit_path)}"
        line = (f"cd {q(cwd)} && mkdir -p {q(os.path.dirname(log_path))} && "
                f"{('export ' + exports + ' && ') if exports else ''}"
                f"(ulimit -n 65536 2>/dev/null; nohup setsid bash -c {q(inner)} > {q(log_path)} 2>&1 < /dev/null & "
                f"echo $!)")
        rc, out, err = self.run(host, line, timeout=120)
        pid = out.strip().splitlines()[-1] if out.strip() else ""
        if rc or not pid.isdigit():
            raise RuntimeError(f"launch failed rc={rc}: {err.strip()[-500:]}")
        time.sleep(1.0)
        info = self.proc_info(host, int(pid))
        return {"pid": int(pid), "start": info["start"] if info else None}

    def proc_info(self, host, pid: int) -> dict | None:
        """``{"start", "state", "cmdline", "pgid"}`` or None when the PID is gone."""
        rc, out, _ = self.run(host, f"cat /proc/{int(pid)}/stat 2>/dev/null && echo '@@' && "
                                    f"tr '\\0' ' ' < /proc/{int(pid)}/cmdline 2>/dev/null")
        if rc or "@@" not in out:
            return None
        stat, cmdline = out.split("@@", 1)
        rest = stat[stat.rfind(")") + 2:].split()
        if len(rest) < 20:
            return None
        return {"state": rest[0], "pgid": int(rest[2]), "start": rest[19], "cmdline": cmdline.strip()}

    def alive(self, host, proc: dict) -> bool:
        info = self.proc_info(host, proc["pid"])
        return bool(info) and info["state"] != "Z" and (proc.get("start") in (None, info["start"]))

    def signal(self, host, proc: dict, sig: str, marker: str, group: bool) -> bool:
        """Signal a process we own (or the explicitly adopted one) after re-checking its identity."""
        info = self.proc_info(host, proc["pid"])
        if not info or info["state"] == "Z":
            return False
        if proc.get("start") and info["start"] != proc["start"]:
            raise RuntimeError(f"PID {proc['pid']} was reused (start time changed); refusing to signal")
        if marker and marker not in info["cmdline"]:
            raise RuntimeError(f"PID {proc['pid']} command line lacks {marker!r}; refusing to signal")
        if group and info["pgid"] != proc["pid"]:
            raise RuntimeError(f"PID {proc['pid']} is not its own process-group leader; refusing group signal")
        target = f"-- -{proc['pid']}" if group else str(proc["pid"])
        return self.run(host, f"kill -{sig} {target}")[0] == 0

    def pgrep(self, host, pattern: str) -> list[str]:
        rc, out, _ = self.run(host, f"pgrep -af -- {q(pattern)} | grep -v pgrep || true")
        return [line for line in out.splitlines() if line.strip() and "dreamer_loop" not in line]


# --------------------------------------------------------------------------- the loop
class StopRequested(Exception):
    pass


class Loop:
    def __init__(self, cfg: dict, backend: Backend | None = None, out=None):
        self.cfg = cfg
        self.root = Path(cfg["loop_root"])
        self.state_path = self.root / "state.json"
        self.log_path = self.root / "loop_log.jsonl"
        self.stop_path = self.root / "STOP"
        self.out = out or (lambda text: print(text, flush=True))
        self.b = backend or Backend(cfg["ssh_retries"], log=self.log)
        if backend is not None and getattr(backend, "log", None) is None:
            backend.log = self.log
        self.hosts = {name: dict(host, name=name) for name, host in cfg["hosts"].items()}
        self.state: dict = {}

    # -- records
    def log(self, event: str, **data) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        row = {"t": utcnow(), "iter": self.state.get("iteration"), "step": self.state.get("step"),
               "event": event, **data}
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, default=str) + "\n")
        self.out(json.dumps(row, default=str))

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def load(self) -> bool:
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            return True
        return False

    def it(self) -> dict:
        return self.state["iters"].setdefault(str(self.state["iteration"]), {"steps": {}})

    def idir(self, k: int | None = None) -> Path:
        return self.root / f"iter_{int(k or self.state['iteration']):03d}"

    def goto(self, step: str, **data) -> None:
        it = self.it()
        old = self.state["step"]
        it["steps"].setdefault(old, {}).setdefault("start", utcnow())
        it["steps"][old]["end"] = utcnow()
        self.state["step"] = step
        it["steps"].setdefault(step, {})["start"] = utcnow()
        self.save()
        self.log("goto", frm=old, to=step, **data)

    # -- init
    def init_state(self, init_wm, init_wm_val_log, policy_run_dir, adopt_pid, policy_wm=None) -> None:
        if not init_wm or not policy_run_dir:
            raise ValueError("first run needs --init-wm and --init-policy-run-dir (or bootstrap in the config)")
        for p in (init_wm, policy_run_dir, init_wm_val_log or ""):
            refuse_restricted(p)
        if not Path(init_wm).is_file():
            raise ValueError(f"init WM {init_wm} not found on the WM host")
        policy_wm = policy_wm or str(init_wm)       # the WM the running policy uses (e.g. a lane-calibrated copy)
        policy = {"run_dir": policy_run_dir, "proc": None, "owned": False, "stop_file": False,
                  "started": utcnow(), "wm_local": policy_wm, "data_dir": None}
        if adopt_pid:
            info = self.b.proc_info(self.hosts["policy"], int(adopt_pid))
            if not info:
                raise ValueError(f"--adopt-policy-pid {adopt_pid} is not running")
            if "wms.control.train_cli" not in info["cmdline"] or policy_run_dir.rstrip("/") not in info["cmdline"]:
                raise ValueError(f"PID {adopt_pid} is not a train_cli writing to {policy_run_dir}: {info['cmdline']}")
            policy["proc"] = {"pid": int(adopt_pid), "start": info["start"]}
            policy["adopted"] = True
        self.root.mkdir(parents=True, exist_ok=True)
        created = utcnow()
        self.state = {"version": 1, "created": created, "iteration": 1, "step": "snapshot",
                      "loop_id": "loop_" + created.replace("-", "").replace(":", ""),
                      "status": "running", "config_snapshot": self.cfg,
                      "wm": {"ckpt": str(init_wm), "policy_ckpt": policy_wm,
                             "val_log": init_wm_val_log or None, "history": []},
                      "policy": policy, "policy_history": [], "seeds_used": [], "converted_dirs": [],
                      "iters": {}}
        self.save()
        self.log("init", wm=init_wm, policy_run_dir=policy_run_dir, adopted_pid=adopt_pid)

    def remote_root(self, host: dict) -> str:
        """Per-loop directory on a collector host (a new loop never reuses an old loop's dirs)."""
        return f"{host['work_root'].rstrip('/')}/{self.state.get('loop_id', 'loop')}"

    # -- guards
    def check_stop(self) -> None:
        if self.stop_path.exists():
            raise StopRequested()

    def wait(self, seconds: float) -> None:
        """Sleep in small slices; STOP ends the wait (the detached job keeps running)."""
        end = self.b.now() + seconds
        while self.b.now() < end:
            self.check_stop()
            self.b.sleep(min(15.0, max(0.0, end - self.b.now())))

    def disk_guard(self) -> None:
        paths = self.cfg.get("disk_paths") or [str(self.root)]
        while True:
            free = min(self.b.free_gb(p) for p in paths)
            if free >= self.cfg["min_free_gb"]:
                if self.state.get("status") == "paused_disk":
                    self.state["status"] = "running"
                    self.save()
                    self.log("disk_ok", free_gb=round(free, 2))
                return
            if self.state.get("status") != "paused_disk":
                self.state["status"] = "paused_disk"
                self.save()
                self.log("disk_pause", free_gb=round(free, 2), min_gb=self.cfg["min_free_gb"])
            self.wait(self.cfg["poll_s"] * 5)

    # -- main
    def run(self, max_steps: int | None = None) -> int:
        self.state["status"] = "running"
        self.save()
        self.log("start", pid=os.getpid())
        steps = 0
        try:
            while True:
                self.check_stop()
                if self.state["iteration"] > self.cfg["max_iterations"]:
                    self.state["status"] = "finished"
                    self.save()
                    self.log("finished", reason="max_iterations")
                    return 0
                step = self.state["step"]
                if step in LAUNCH_STEPS:
                    self.disk_guard()
                try:
                    getattr(self, "step_" + step)()
                    self.state.get("step_errors", {}).pop(f"{self.state['iteration']}:{step}", None)
                except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                    # transient trouble (ssh drop, rsync, a busy box): retry the same step after a pause
                    key = f"{self.state['iteration']}:{step}"
                    errors = self.state.setdefault("step_errors", {})
                    errors[key] = errors.get(key, 0) + 1
                    self.save()
                    self.log("step_error", error=f"{type(exc).__name__}: {exc}"[:800], count=errors[key])
                    if errors[key] >= self.cfg["max_step_errors"]:
                        raise
                    self.wait(self.cfg["poll_s"] * 5)
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    return 0
                if self.state.get("status") in ("finished", "failed"):
                    return 0 if self.state["status"] == "finished" else 2
        except StopRequested:
            self.state["status"] = "stopped"
            self.save()
            self.log("stopped", reason="STOP file", resume_step=self.state["step"])
            return 0
        except Exception as exc:       # noqa: BLE001 - recorded, then re-raised for the exit code
            self.state["status"] = "error"
            self.state["last_error"] = f"{type(exc).__name__}: {exc}"
            self.save()
            self.log("error", error=self.state["last_error"])
            raise

    # -- policy helpers
    def policy_alive(self) -> bool:
        proc = self.state["policy"].get("proc")
        return bool(proc) and self.b.alive(self.hosts["policy"], proc)

    def newest_ckpt(self, settle_s: float = 30.0, newer_than: float = 0.0):
        """Newest ``policy_*.pt`` that is settled (mtime older than ``settle_s``)."""
        now = self.b.now()
        for mtime, size, path in self.b.list_ckpts(self.hosts["policy"], self.state["policy"]["run_dir"]):
            if size > 0 and now - mtime >= settle_s and mtime > newer_than:
                return mtime, path
        return None

    def ensure_policy(self, reason: str) -> None:
        """Restart a dead policy (crash, finished) from its newest checkpoint on the current WM."""
        if self.policy_alive():
            return
        pol = self.state["policy"]
        if pol.get("proc") is None and not pol.get("owned") and not pol.get("adopted"):
            # bootstrap without a running trainer: start one on the current WM
            pass
        it = self.it()
        restarts = it.get("policy_restarts", 0)
        if restarts >= self.cfg["policy"]["max_restarts"]:
            if not it.get("policy_restart_exhausted"):
                it["policy_restart_exhausted"] = True
                self.save()
                self.log("policy_dead_no_restart", reason=reason, restarts=restarts)
            return
        found = self.newest_ckpt(settle_s=5.0)
        if not found:
            self.log("policy_dead_no_ckpt", run_dir=pol["run_dir"])
            return
        it["policy_restarts"] = restarts + 1
        self.save()
        self.log("policy_restart", reason=reason, resume=found[1])
        self.start_policy(found[1], pol.get("wm_local", self.state["wm"]["policy_ckpt"]),
                          self.policy_data_dir(), tag=f"r{restarts + 1}")

    def policy_data_dir(self) -> str | None:
        """Converted policy episodes for a (re)started trainer: the running policy's, else the newest."""
        pol = self.state["policy"]
        if pol.get("data_dir") and Path(pol["data_dir"]).exists():
            return pol["data_dir"]
        for d in reversed(self.state["converted_dirs"]):
            if Path(d).exists():
                return d
        return self.cfg["policy"].get("initial_data_dir")

    def start_policy(self, resume_ckpt: str, wm_ckpt: str, data_dir: str | None, tag: str = "",
                     name: str | None = None) -> None:
        """Launch ``train_cli --resume`` (WMS_LOOP_REFRESH=1) on ``wm_ckpt``; ``data_dir`` = the converted
        policy episodes (WMS_DATA_V6_POLICY: start windows of configs/c4_policy_v6_loop.toml)."""
        cfg, host = self.cfg["policy"], self.hosts["policy"]
        k = self.state["iteration"]
        runs_root = host["work_root"].rstrip("/") + "/runs"
        name = name or f"c4_policy_loop_iter{k:03d}"
        base = f"{runs_root}/{name}{('_' + tag) if tag else ''}"
        out, n = base, 1
        while self.b.exists(host, out):
            n += 1
            out = f"{base}.a{n}"
        wm_local = wm_ckpt
        wm_dir = os.path.dirname(wm_ckpt)
        if host.get("ssh"):          # remote policy host: ship the WM (ckpt + physics + receipt) and the ckpt
            rwm = f"{host['work_root'].rstrip('/')}/wm/{name}{('_' + tag) if tag else ''}"
            for name in (os.path.basename(wm_ckpt), "physics.json", "TRAIN_RECEIPT.json"):
                if os.path.exists(os.path.join(wm_dir, name)):
                    self.b.copy(self.hosts["wm"], os.path.join(wm_dir, name), host, f"{rwm}/{name}")
            wm_ckpt, wm_dir = f"{rwm}/{os.path.basename(wm_ckpt)}", rwm
        local_data_dir = data_dir
        if data_dir and host.get("ssh"):   # remote policy host: ship the converted policy episodes too
            remote_data = f"{host['work_root'].rstrip('/')}/data/{os.path.basename(data_dir)}"
            if not self.b.exists(host, remote_data + "/C4_EPISODES_MANIFEST.json"):
                self.b.copy(self.hosts["wm"], data_dir.rstrip("/") + "/", host, remote_data + "/")
            data_dir = remote_data
        allowed = [*cfg["allowed_roots"], wm_dir, runs_root, os.path.dirname(resume_ckpt)]
        if data_dir:
            allowed.append(data_dir)
        env = {"PYTHONPATH": "src:.", "OMP_NUM_THREADS": "12", "MKL_NUM_THREADS": "12",
               **cfg["env"], **({"WMS_DATA_V6_POLICY": data_dir} if data_dir else {}),
               "WMS_LOOP_REFRESH": "1", "C4_ALLOWED_ROOTS": ":".join(allowed)}
        # without converted policy episodes (a restart before the first conversion) the plain config runs
        config = cfg["config"] if data_dir else cfg["config_without_policy_data"]
        argv = [host["python"], "-m", "wms.control.train_cli", "--config", config, "--wm-ckpt", wm_ckpt,
                "--resume", resume_ckpt, "--device", cfg["device"], "--max-updates", str(cfg["max_updates"]),
                "--output-dir", out, *cfg["extra_args"]]
        cmd = (f"nice -n {int(cfg['nice'])} " if cfg.get("nice") else "") + " ".join(q(a) for a in argv)
        self.b.run(host, f"mkdir -p {q(out)}", check=True)
        launch_txt = (f"started_utc={utcnow()}\nrepo={host['code_root']}\nresume={resume_ckpt}\nwm_ckpt={wm_ckpt}\n"
                      f"env={json.dumps(env, sort_keys=True)}\ncmd={cmd}\n")
        self.b.run(host, f"printf %s {q(launch_txt)} > {q(out + '/LAUNCH.txt')}", check=True)
        proc = self.b.launch(host, host["code_root"], cmd, f"{out}/train.log", f"{out}/EXIT", env)
        self.b.run(host, f"echo {proc['pid']} > {q(out + '/pid')}")
        old = self.state["policy"]
        self.state["policy_history"].append(old)
        self.state["policy"] = {"run_dir": out, "proc": proc, "owned": True, "stop_file": True,
                                "started": utcnow(), "resume": resume_ckpt, "wm_ckpt": wm_ckpt,
                                "iteration": k, "data_dir": local_data_dir, "remote_data_dir":
                                data_dir if host.get("ssh") else None}
        self.save()
        self.state["policy"]["wm_local"] = wm_local
        self.save()
        self.log("policy_launched", run_dir=out, pid=proc["pid"], resume=resume_ckpt, wm=wm_ckpt,
                 data_dir=data_dir)

    # -- steps
    def step_snapshot(self) -> None:
        self.ensure_policy("snapshot")
        found = self.newest_ckpt(settle_s=30.0)
        if not found:
            self.log("snapshot_wait", reason="no settled policy checkpoint yet")
            self.wait(self.cfg["poll_s"])
            return
        mtime, path = found
        dst = self.idir() / "policy_snapshot.pt"
        if dst.exists():                 # our own copy from an interrupted attempt of this step
            dst.unlink()
        self.idir().mkdir(parents=True, exist_ok=True)
        self.b.copy(self.hosts["policy"], path, self.hosts["wm"], str(dst))
        it = self.it()
        it.update(snapshot={"source": path, "mtime": mtime, "path": str(dst), "sha256": sha256(dst),
                            "policy_run_dir": self.state["policy"]["run_dir"]},
                  wm_at_start=self.state["wm"]["ckpt"], started=utcnow())
        self.goto("collect_launch", snapshot=path)

    def allocate_seeds(self) -> list[int]:
        it = self.it()
        if it.get("seeds"):
            return it["seeds"]
        lo, hi = self.cfg["seeds"]["block"]
        taken = expand(self.state["seeds_used"]) | expand(self.cfg["seeds"]["exclude"])
        seeds = [s for s in range(lo, hi + 1) if s not in taken][: self.cfg["seeds"]["per_iteration"]]
        if not seeds:
            return []
        check_seeds(seeds)
        it["seeds"] = seeds
        self.state["seeds_used"] = ranges(sorted(expand(self.state["seeds_used"]) | set(seeds)))
        self.save()
        with (self.root / "seed_ledger.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"t": utcnow(), "iteration": self.state["iteration"], "block": ("C4_V6_LONGPAIR" if max(seeds) <= 965999 else "C4_V6_LONGPAIR+C4_LOOP_TRAIN" if min(seeds) <= 965999 else "C4_LOOP_TRAIN"),
                                     "use": "policy-rollout WM training data", "seeds": seed_spec(seeds),
                                     "n": len(seeds)}) + "\n")
        self.log("seeds_allocated", seeds=seed_spec(seeds), n=len(seeds))
        return seeds

    def collect_cmd(self, policy_path: str, seeds: list[int], out: str, host_name: str = "cpu") -> str:
        c, host = self.cfg["collect"], self.hosts[host_name]
        argv = [host["python"], "-m", "traffic_shepherd.c4.collect", "--policy", policy_path,
                "--population", c["population"], "--seeds", seed_spec(seeds), "--case-plan", c["case_plan"],
                "--penetrations", c["penetrations"], "--pen-weights", c["pen_weights"],
                "--demands", c["demands"], "--warmup", str(c["warmup"]), "--horizon", str(c["horizon"]),
                "--workers", str(c["workers"]), "--no-physics-log", "--out", out]
        prefix = f"nice -n {int(c['nice'])} " if c.get("nice") else ""
        # SUMO from the venv's eclipse-sumo package (as scripts/boxops/common.sh activate_box): a
        # non-interactive ssh shell has neither the venv bin dir nor SUMO_HOME on PATH.
        py = host["python"]
        sumo = (f'SUMO_HOME="$({q(py)} -c {q("import sumo; print(sumo.SUMO_HOME)")} 2>/dev/null)"; '
                f'export SUMO_HOME; export PATH={q(os.path.dirname(py))}:"$SUMO_HOME/bin:$PATH"; ')
        return sumo + prefix + " ".join(q(a) for a in argv)

    def step_collect_launch(self) -> None:
        host, c = self.hosts["cpu"], self.cfg["collect"]
        for pattern in c["wait_for_absent"]:
            busy = self.b.pgrep(host, pattern)
            if busy:
                self.log("collect_wait_busy", pattern=pattern, n=len(busy))
                self.wait(self.cfg["poll_s"] * 2)
                return
        seeds = self.allocate_seeds()
        if not seeds:
            self.state["status"] = "finished"
            self.save()
            self.log("finished", reason="seed block exhausted")
            return
        it = self.it()
        k = self.state["iteration"]
        rdir = f"{self.remote_root(host)}/iter_{k:03d}"
        policy_path = f"{rdir}/policy_snapshot.pt"
        if not it.get("collect_pushed") and self.b.exists(host, rdir):
            raise RuntimeError(f"{host['name']}:{rdir} exists; never reuse a collection directory")
        if not it.get("collect_pushed"):
            self.b.copy(self.hosts["wm"], it["snapshot"]["path"], host, policy_path)
            it["collect_pushed"] = True
            self.save()
        attempt = it.get("collect_attempts", 0) + 1
        cmd = self.collect_cmd(policy_path, seeds, f"{rdir}/policy")
        env = {"PYTHONPATH": f"{host['code_root']}/src:{host['code_root']}", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        self.b.run(host, f"mkdir -p {q(rdir)} && rm -f {q(rdir + '/EXIT')} && printf %s {q(cmd)} > "
                         f"{q(rdir + f'/collect_cmd.a{attempt}.txt')}", check=True)
        proc = self.b.launch(host, host["code_root"], cmd, f"{rdir}/collect.a{attempt}.log", f"{rdir}/EXIT", env)
        it.update(collect_attempts=attempt, collect={"proc": proc, "remote_dir": rdir, "launched": utcnow(),
                                                     "launched_ts": self.b.now(), "cmd": cmd})
        self.goto("collect_wait", pid=proc["pid"], attempt=attempt, seeds=seed_spec(seeds))

    def step_collect_wait(self) -> None:
        host, c, it = self.hosts["cpu"], self.cfg["collect"], self.it()
        col = it["collect"]
        while True:
            self.ensure_policy("collect_wait")
            code = self.b.read_text(host, col["remote_dir"] + "/EXIT")
            if code is not None and code.strip():
                rc = int(code.strip()) if code.strip().lstrip("-").isdigit() else -1
                col["exit"] = rc
                self.save()
                if rc == 0:
                    self.goto("pull", collect_exit=0)
                    return
                return self._collect_failed(f"exit {rc}")
            if not self.b.alive(host, col["proc"]):
                # double-check the exit file once more (race between exit and the check)
                code = self.b.read_text(host, col["remote_dir"] + "/EXIT")
                if code is None or not code.strip():
                    return self._collect_failed("process gone without EXIT file")
                continue
            if self.b.now() - col["launched_ts"] > c["timeout_s"]:
                self.b.signal(host, col["proc"], "TERM", "", group=True)
                return self._collect_failed("timeout")
            self.wait(self.cfg["poll_s"])

    def _collect_failed(self, why: str) -> None:
        it = self.it()
        self.log("collect_failed", why=why, attempt=it.get("collect_attempts"))
        if it.get("collect_attempts", 0) >= self.cfg["collect"]["max_attempts"]:
            self.state["status"] = "failed"
            self.save()
            self.log("loop_failed", why=f"collection failed {it['collect_attempts']} times")
            return
        self.goto("collect_launch", retry=True)   # same seeds and directory: collect resumes

    def step_pull(self) -> None:
        it, host, k = self.it(), self.hosts["cpu"], self.state["iteration"]
        rdir = it["collect"]["remote_dir"]
        local = self.root / "raw" / f"iter_{k:03d}"
        self.b.copy(host, rdir + "/policy/", self.hosts["wm"], str(local) + "/")
        for name in (f"collect.a{it['collect_attempts']}.log",):
            try:
                self.b.copy(host, f"{rdir}/{name}", self.hosts["wm"], str(self.idir() / name))
            except (RuntimeError, OSError) as exc:
                self.log("pull_log_failed", file=name, error=str(exc))
        episodes = find_episodes(local)
        expected = len(it["seeds"])
        it["pulled"] = {"dir": str(local), "complete_episodes": len(episodes), "expected": expected}
        self.save()
        if len(episodes) < max(1, math.ceil(self.cfg["collect"]["min_complete_fraction"] * expected)):
            return self._collect_failed(f"only {len(episodes)}/{expected} complete episodes")
        self.goto("convert", episodes=len(episodes))

    def step_convert(self) -> None:
        k, cv = self.state["iteration"], self.cfg["convert"]
        episodes = find_episodes(self.root / "raw")
        if cv["max_episodes"] and len(episodes) > cv["max_episodes"]:
            episodes = episodes[-cv["max_episodes"]:]          # the most recent iterations
        dest = Path(self.cfg["data_root"]) / f"{cv['prefix']}{k:03d}"
        refuse_restricted(dest)
        it = self.it()
        # A partial dir from a crashed attempt of this loop is ours; anything else is refused.
        if dest.exists():
            if str(dest) in it.get("convert_attempt_dirs", []):
                shutil.rmtree(dest)
                self.log("convert_removed_partial", dir=str(dest))
            else:
                raise RuntimeError(f"{dest} exists and was not created by this loop; refusing")
        # Superseded converted dirs (their WM refresh has finished) go first; the running policy's stays.
        self.delete_converted(keep={self.state["policy"].get("data_dir")})
        free = self.b.free_gb(dest.parent)
        fit = self.window_fit(free)
        if fit < len(episodes):
            if fit < max(1, min(len(episodes), cv.get("min_window", 50))):
                if self.state.get("status") != "paused_disk":
                    self.state["status"] = "paused_disk"
                    self.save()
                    self.log("disk_pause", free_gb=round(free, 2), fit_episodes=fit, episodes=len(episodes))
                self.wait(self.cfg["poll_s"] * 5)
                return
            self.log("convert_window_reduced", episodes=len(episodes), kept_recent=fit, free_gb=round(free, 2))
            episodes = episodes[-fit:]
        if self.state.get("status") == "paused_disk":
            self.state["status"] = "running"
        it.setdefault("convert_attempt_dirs", []).append(str(dest))
        self.save()
        wm = self.hosts["wm"]
        (self.idir() / "convert_episodes.txt").write_text("\n".join(episodes) + "\n", encoding="utf-8")
        prefix = f"nice -n {int(cv['nice'])} " if cv.get("nice") else ""
        cmd = (f"cd {q(wm['code_root'])} && PYTHONPATH=src:. OMP_NUM_THREADS=4 {prefix}"
               f"{convert_cmd(wm['python'], self.idir() / 'convert_episodes.txt', dest)} "
               f"> {q(self.idir() / 'convert.log')} 2>&1")
        started = self.b.now()
        rc, _, err = self.b.run(wm, cmd, timeout=cv["timeout_s"])
        if rc != 0 or not (dest / "C4_EPISODES_MANIFEST.json").exists():
            self.log("convert_failed", rc=rc, err=err[-300:])
            raise RuntimeError(f"conversion failed (rc={rc}); see {self.idir() / 'convert.log'}")
        self.state["converted_dirs"].append(str(dest))
        it["converted"] = {"dir": str(dest), "episodes": len(episodes), "seconds": round(self.b.now() - started)}
        self.save()
        self.goto("wm_launch", converted=str(dest), episodes=len(episodes))

    def window_fit(self, free_gb: float) -> int:
        """Episodes the next converted pool may hold.  A pool the running policy (or a queued switch)
        still reads cannot be deleted before the next one exists, so in steady state two pools coexist:
        each gets at most half of the space above ``min_free_gb`` (else the WM lane would stall)."""
        cv = self.cfg["convert"]
        kept = sum(dir_size_gb(d) for d in self.state["converted_dirs"] if Path(d).exists())
        room = max(0.0, free_gb - self.cfg["min_free_gb"])
        half = max(0.0, (free_gb + kept - self.cfg["min_free_gb"]) / 2.0)
        return int(min(room, half) * 1e3 / cv["mb_per_episode"])

    def delete_converted(self, keep) -> None:
        """Delete converted dirs THIS loop created (recorded in the state) except ``keep``; raw stays.
        Never deletes a dir the running policy reads (its start windows are memory-mapped lazily)."""
        keep = {d for d in keep if d}
        remaining = []
        for d in self.state["converted_dirs"]:
            if d in keep or not Path(d).name.startswith(self.cfg["convert"]["prefix"]):
                remaining.append(d)
                continue
            if Path(d).exists():
                shutil.rmtree(d)
                self.log("converted_deleted", dir=d)
        self.state["converted_dirs"] = remaining
        self.save()

    def wm_cmd(self, out: str, init_from: str) -> str:
        w, host = self.cfg["wm"], self.hosts["wm"]
        stages = "[" + ",".join(str(int(s)) for s in w["stage_steps"]) + "]"
        argv = [host["python"], "-m", "wms", "train", "--config", w["config"], "--mode", "run", "--out", out,
                "--set", f"model.init_from={init_from}", "--set", f"curriculum.stage_steps={stages}"]
        for item in w["extra_set"]:
            argv += ["--set", item]
        prefix = f"nice -n {int(w['nice'])} " if w.get("nice") else ""
        return prefix + " ".join(q(a) for a in argv)

    def step_wm_launch(self) -> None:
        it, k, host = self.it(), self.state["iteration"], self.hosts["wm"]
        base = Path(self.cfg["wm_runs_root"]) / f"c4_wm_loop_iter{k:03d}"
        out, n = base, 1
        while out.exists() or Path(str(out) + ".stdout.log").exists():
            n += 1
            out = Path(f"{base}.a{n}")
        env = {"PYTHONPATH": "src:.", "OMP_NUM_THREADS": "12", "WMS_JAM_WORKERS": "14", "WMS_LATEST_EVERY": "100",
               **self.cfg["wm"]["env"], "WMS_DATA_V6_POLICY": it["converted"]["dir"]}
        cmd = self.wm_cmd(str(out), self.state["wm"]["ckpt"])
        proc = self.b.launch(host, host["code_root"], cmd, f"{out}.stdout.log", f"{out}.EXIT", env)
        it["wm"] = {"out": str(out), "proc": proc, "launched": utcnow(), "launched_ts": self.b.now(), "cmd": cmd,
                    "env": env, "init_from": self.state["wm"]["ckpt"]}
        self.save()
        Path(f"{out}.pid").write_text(f"{proc['pid']}\n", encoding="utf-8")
        self.goto("wm_wait", out=str(out), pid=proc["pid"])

    def step_wm_wait(self) -> None:
        it, host = self.it(), self.hosts["wm"]
        wm = it["wm"]
        out = Path(wm["out"])
        while True:
            self.ensure_policy("wm_wait")
            if (out / "RUN_SUMMARY.json").exists():
                wm["seconds"] = round(self.b.now() - wm["launched_ts"])
                self.save()
                self.goto("wm_check", seconds=wm["seconds"])
                return
            if not self.b.alive(host, wm["proc"]):
                if (out / "RUN_SUMMARY.json").exists():
                    continue
                return self._wm_rejected("process ended without RUN_SUMMARY.json")
            if self.b.now() - wm["launched_ts"] > self.cfg["wm"]["timeout_s"]:
                self.b.signal(host, wm["proc"], "TERM", "", group=True)
                return self._wm_rejected("timeout")
            self.wait(self.cfg["poll_s"])

    def _wm_rejected(self, why: str, **data) -> None:
        it = self.it()
        it["wm_verdict"] = {"accepted": False, "why": why, **data}
        self.state["wm"]["history"].append({"iteration": self.state["iteration"], "accepted": False,
                                            "out": it.get("wm", {}).get("out"), "why": why})
        self.save()
        self.log("wm_rejected", why=why, kept=self.state["wm"]["ckpt"], **data)
        self.goto("done", wm_accepted=False)

    def step_wm_check(self) -> None:
        it, w = self.it(), self.cfg["wm"]
        out = Path(it["wm"]["out"])
        best = out / "wm_s_best.pt"
        if not best.exists() or not (out / "physics.json").exists():
            return self._wm_rejected("wm_s_best.pt or physics.json missing")
        verdict = wm_sanity(out / "train_log.jsonl", out / "val_log.jsonl", self.state["wm"].get("val_log"),
                            w["val_metric"], w["val_tolerance"])
        if not verdict["ok"]:
            return self._wm_rejected("; ".join(verdict["reasons"]), compare=verdict["compare"])
        policy_ckpt = str(best)
        if w["lane_calib"]:
            policy_ckpt = self.lane_calib(out, best) or policy_ckpt
        self.state["wm"].update(ckpt=str(best), policy_ckpt=policy_ckpt, val_log=str(out / "val_log.jsonl"))
        self.state["wm"]["history"].append({"iteration": self.state["iteration"], "accepted": True, "out": str(out),
                                            "compare": verdict["compare"]})
        it["wm_verdict"] = {"accepted": True, **verdict}
        self.save()
        self.log("wm_accepted", ckpt=str(best), policy_ckpt=policy_ckpt, compare=verdict["compare"],
                 note=verdict.get("note"))
        self.goto("policy_stop")

    def lane_calib(self, out: Path, best: Path) -> str | None:
        w, host = self.cfg["wm"], self.hosts["wm"]
        cal = out / "lanecal"
        cal.mkdir(exist_ok=True)
        for name in ("physics.json", "TRAIN_RECEIPT.json"):
            if (out / name).exists():
                shutil.copy2(out / name, cal / name)
        env = " ".join(f"{k}={q(v)}" for k, v in {"PYTHONPATH": "src:.", **w["env"]}.items())
        cmd = (f"cd {q(host['code_root'])} && export {env} && timeout {int(w['lane_calib_timeout_s'])} "
               f"{q(host['python'])} -B tools/lanehazard_calib.py {q(best)} --config {q(w['lane_calib_config'])} "
               f"--batches 40 --batch-size 8 --split val --out {q(cal / 'calib.json')} "
               f"--write-ap {q(cal / 'wm_s_best_ap.pt')} > {q(cal / 'calib.log')} 2>&1")
        rc, _, _ = self.b.run(host, cmd, timeout=w["lane_calib_timeout_s"] + 120)
        if rc == 0 and (cal / "wm_s_best_ap.pt").exists():
            self.log("lane_calib_ok", ckpt=str(cal / "wm_s_best_ap.pt"))
            return str(cal / "wm_s_best_ap.pt")
        self.log("lane_calib_failed", rc=rc, fallback=str(best))
        return None

    def step_policy_stop(self) -> None:
        pol, host, cfg, it = self.state["policy"], self.hosts["policy"], self.cfg["policy"], self.it()
        stop = it.setdefault("policy_stop", {"requested": utcnow(), "requested_ts": self.b.now()})
        if not self.policy_alive():
            found = self.newest_ckpt(settle_s=5.0)
            if not found:
                raise RuntimeError("policy is not running and has no checkpoint to resume from")
            stop.update(resume=found[1], how="already stopped")
            self.save()
            self.goto("policy_start", resume=found[1])
            return
        if pol.get("stop_file"):
            # Trainer launched by this loop: STOP file -> clean exit after the current update + policy_latest.pt
            stop_file = pol["run_dir"].rstrip("/") + "/STOP"
            self.b.run(host, f"touch {q(stop_file)}", check=True)
            self.log("policy_stop_file", file=stop_file)
            while self.policy_alive():
                if self.b.now() - stop["requested_ts"] > cfg["stop_timeout_s"]:
                    self.log("policy_stop_timeout", action="SIGTERM")
                    self.terminate_policy()
                    break
                self.wait(min(30, self.cfg["poll_s"]))
            latest = pol["run_dir"].rstrip("/") + "/policy_latest.pt"
            found = self.newest_ckpt(settle_s=0.0)
            resume = found[1] if found else None
            if found and self.b.exists(host, latest) and found[1].endswith("policy_latest.pt"):
                resume = latest
            stop.update(resume=resume, how="stop_file")
        else:
            # Adopted trainer (no STOP support): wait for its next periodic checkpoint policy_<n>.pt, then
            # for the update after it (validation and best-checkpoint saving of update n are done),
            # then SIGTERM exactly that PID.  At most one update of progress is lost.
            stop.setdefault("baseline_mtime", (self.newest_ckpt(settle_s=0.0) or (0.0, ""))[0])
            self.save()
            resume = None
            while True:
                periodic = [(m, p) for m, s, p in self.b.list_ckpts(host, pol["run_dir"])
                            if m > stop["baseline_mtime"] and _update_of(p) is not None and s > 0]
                if periodic:
                    mtime, path = max(periodic)
                    n = _update_of(path)
                    if self._updates_reached(n + 1):
                        resume = path
                        break
                if self.b.now() - stop["requested_ts"] > cfg["adopted_wait_max_s"]:
                    self.log("policy_adopted_wait_timeout")
                    break
                self.wait(5 if periodic else min(30, self.cfg["poll_s"]))
            self.terminate_policy()
            if resume is None:
                found = self.newest_ckpt(settle_s=10.0)
                resume = found[1] if found else None
            stop.update(resume=resume, how="adopted_checkpoint_then_sigterm")
        if not resume:
            raise RuntimeError("no checkpoint to resume the policy from")
        stop["stopped"] = utcnow()
        self.save()
        self.goto("policy_start", resume=resume)

    def _updates_reached(self, n: int) -> bool:
        text = self.b.read_text(self.hosts["policy"], self.state["policy"]["run_dir"].rstrip("/") + "/updates.jsonl",
                                tail_bytes=20000) or ""
        for line in reversed(text.splitlines()):
            try:
                return int(json.loads(line).get("update", -1)) >= n
            except ValueError:
                continue
        return False

    def terminate_policy(self) -> None:
        pol, host = self.state["policy"], self.hosts["policy"]
        proc = pol.get("proc")
        if not proc:
            return
        group = bool(pol.get("owned"))
        marker = "" if group else "wms.control.train_cli"   # owned: group leader is our bash wrapper
        self.b.signal(host, proc, "TERM", marker, group=group)
        self.log("policy_sigterm", pid=proc["pid"], group=group)
        for _ in range(12):
            if not self.b.alive(host, proc):
                return
            self.b.sleep(5)
        self.b.signal(host, proc, "KILL", marker, group=group)
        self.log("policy_sigkill", pid=proc["pid"])

    def step_policy_start(self) -> None:
        it = self.it()
        resume = it["policy_stop"]["resume"]
        prev = dict(self.state["policy"])
        data_dir = it.get("converted", {}).get("dir") or self.policy_data_dir()
        self.start_policy(resume, self.state["wm"]["policy_ckpt"], data_dir)
        self.wait(self.cfg["policy"]["start_check_s"])
        if not self.policy_alive():
            tail = self.b.read_text(self.hosts["policy"], self.state["policy"]["run_dir"] + "/train.log",
                                    tail_bytes=2000)
            self.log("policy_start_failed", tail=(tail or "")[-1500:])
            it["policy_start_failed"] = True
            # fall back to the previous WM so the policy keeps training
            prev_wm = it.get("wm", {}).get("init_from")
            if prev_wm:
                self.state["wm"]["history"].append({"iteration": self.state["iteration"], "policy_start_failed": True})
                self.start_policy(resume, prev_wm, data_dir, tag="fallback")
        it["policy_after"] = {"run_dir": self.state["policy"]["run_dir"], "prev_run_dir": prev["run_dir"]}
        self.save()
        self.goto("done")

    def step_done(self) -> None:
        it = self.it()
        it["finished"] = utcnow()
        self.save()
        # the running policy's converted dir stays; everything older (or a rejected WM's) goes
        self.delete_converted(keep={self.state["policy"].get("data_dir")})
        self.log("iteration_done", iteration=self.state["iteration"], wm=self.state["wm"]["ckpt"],
                 policy_run_dir=self.state["policy"]["run_dir"])
        self.state["iteration"] += 1
        self.state["step"] = "snapshot"
        self.state["iters"].setdefault(str(self.state["iteration"]), {"steps": {"snapshot": {"start": utcnow()}}})
        self.save()

    # -- dry run
    def plan(self) -> list[str]:
        k = self.state["iteration"]
        lo, hi = self.cfg["seeds"]["block"]
        taken = expand(self.state["seeds_used"]) | expand(self.cfg["seeds"]["exclude"])
        seeds = [s for s in range(lo, hi + 1) if s not in taken][: self.cfg["seeds"]["per_iteration"]]
        cpu = self.hosts["cpu"]
        rdir = f"{cpu['work_root'].rstrip('/')}/iter_{k:03d}"
        lines = [f"iteration {k} (step {self.state['step']}, max {self.cfg['max_iterations']})",
                 f"1 snapshot newest settled policy_*.pt of {self.state['policy']['run_dir']} -> {self.idir()}/policy_snapshot.pt",
                 f"2 seeds {seed_spec(seeds)} ({len(seeds)}); push -> {cpu.get('name')}:{rdir}/policy_snapshot.pt; run:",
                 "    " + self.collect_cmd(f"{rdir}/policy_snapshot.pt", seeds, f"{rdir}/policy"),
                 f"3 rsync {rdir}/policy/ -> {self.root}/raw/iter_{k:03d}/",
                 f"4 convert all complete episodes under {self.root}/raw -> "
                 f"{self.cfg['data_root']}/{self.cfg['convert']['prefix']}{k:03d}",
                 "5 " + self.wm_cmd(f"{self.cfg['wm_runs_root']}/c4_wm_loop_iter{k:03d}", self.state["wm"]["ckpt"]),
                 f"6 sanity: finite losses, val {self.cfg['wm']['val_metric']} <= (1+{self.cfg['wm']['val_tolerance']}) x "
                 f"{self.state['wm'].get('val_log')}",
                 "7 stop policy " + ("via STOP file" if self.state["policy"].get("stop_file") else
                                     f"(adopted pid {(self.state['policy'].get('proc') or {}).get('pid')}): "
                                     "next policy_<n>.pt + 1 update, then SIGTERM"),
                 f"8 restart: WMS_LOOP_REFRESH=1 train_cli --resume <ckpt> --wm-ckpt <new WM> --output-dir "
                 f"{self.hosts['policy']['work_root'].rstrip('/')}/runs/c4_policy_loop_iter{k:03d}"]
        return lines


# --------------------------------------------------------------------------- pipelined mode
class PipeLoop(Loop):
    """Continuous Dreamer loop: three lanes advanced by one polling tick.

    * collectors - every host in ``collect.hosts`` runs SUMO batches of ``collect.batch_size`` episodes
      back to back, each with the newest settled policy checkpoint and its own disjoint seed range;
      a finished batch is pulled into the pool ``<loop_root>/raw/b<NNNN>``;
    * WM lane    - refreshes run back to back: convert the pool (all batches so far, newest first when
      the disk forces a window) -> ``wms train`` from the newest accepted WM -> sanity check;
    * policy lane - every accepted WM is handed to the policy (STOP / adopted stop, then
      ``WMS_LOOP_REFRESH=1 --resume`` with the pool as ~50 % of its start windows).
    Every action is atomic per tick; a STOP file ends the loop between ticks (detached jobs go on and
    are picked up again on restart)."""

    # -- state
    def init_state(self, *args) -> None:
        super().init_state(*args)
        self.state.update(mode="pipelined", iteration=0, step="pipelined", batches={}, next_batch=1,
                          collectors={h: {"batch": None, "done": False} for h in self.cfg["collect"]["hosts"]},
                          refreshes={}, refresh_n=0, wm_lane={"refresh": None},
                          policy_lane={"pending": None, "stop": None}, policy_gen=0, policy_restarts=0)
        self.save()

    def it(self) -> dict:              # used by inherited helpers: one record per policy generation
        return self.state["iters"].setdefault(f"gen{self.state.get('policy_gen', 0)}", {"steps": {}})

    def run(self, max_steps: int | None = None) -> int:
        self.state["status"] = "running"
        for host in self.cfg["collect"]["hosts"]:          # a collector host added to the config later
            self.state["collectors"].setdefault(host, {"batch": None, "done": False})
        self.save()
        self.log("start", pid=os.getpid(), mode="pipelined")
        ticks = 0
        try:
            while True:
                self.check_stop()
                if self.finished():
                    self.state["status"] = "finished"
                    self.save()
                    self.log("finished", reason="seed block exhausted, pool refreshed, nothing running")
                    return 0
                self.lane("policy_ensure", self.ensure_policy, "tick")
                for host in self.cfg["collect"]["hosts"]:
                    self.lane(f"collect:{host}", self.tick_collector, host)
                self.lane("wm", self.tick_wm)
                self.lane("policy", self.tick_policy)
                ticks += 1
                if max_steps is not None and ticks >= max_steps:
                    return 0
                self.wait(self.cfg["poll_s"])
        except StopRequested:
            self.state["status"] = "stopped"
            self.save()
            self.log("stopped", reason="STOP file")
            return 0
        except Exception as exc:        # noqa: BLE001
            self.state["status"] = "error"
            self.state["last_error"] = f"{type(exc).__name__}: {exc}"
            self.save()
            self.log("error", error=self.state["last_error"])
            raise

    def lane(self, name, fn, *args) -> None:
        """One lane action; transient errors are logged and retried next tick (max_step_errors in a row)."""
        errors = self.state.setdefault("lane_errors", {})
        try:
            fn(*args)
            if errors.pop(name, None) is not None:
                self.save()
        except StopRequested:
            raise
        except (RuntimeError, OSError, subprocess.SubprocessError, ValueError) as exc:
            errors[name] = errors.get(name, 0) + 1
            self.save()
            self.log("lane_error", lane=name, error=f"{type(exc).__name__}: {exc}"[:800], count=errors[name])
            if errors[name] >= self.cfg["max_step_errors"] * 3:
                raise

    def finished(self) -> bool:
        cols = self.state["collectors"]
        idle = all(c["batch"] is None for c in cols.values())
        exhausted = all(c.get("done") for c in cols.values())
        wm_idle = self.state["wm_lane"]["refresh"] is None
        pooled = sum(b.get("episodes", 0) for b in self.state["batches"].values() if b["status"] == "pooled")
        last = self.state["refreshes"].get(str(self.state["refresh_n"]), {})
        refreshed_all = last.get("pool_episodes", -1) >= pooled
        all_idle = (idle and wm_idle and self.state["policy_lane"]["pending"] is None
                    and self.state["policy_lane"]["stop"] is None)
        return all_idle and ((exhausted and refreshed_all) or self.refreshes_done())

    def refreshes_done(self) -> bool:
        """``wm.max_refreshes`` refreshes started AND the last one finished (collection continues meanwhile)."""
        return (self.state["refresh_n"] >= self.cfg["wm"]["max_refreshes"]
                and self.state["wm_lane"]["refresh"] is None)

    # -- disk
    def disk_ok(self, what: str) -> bool:
        paths = self.cfg.get("disk_paths") or [str(self.root)]
        free = min(self.b.free_gb(p) for p in paths)
        if free >= self.cfg["min_free_gb"]:
            if self.state.get("status") == "paused_disk":
                self.state["status"] = "running"
                self.save()
                self.log("disk_ok", free_gb=round(free, 2))
            return True
        if self.state.get("status") != "paused_disk":
            self.state["status"] = "paused_disk"
            self.save()
            self.log("disk_pause", free_gb=round(free, 2), blocked=what)
        return False

    # -- collectors
    def next_seeds(self, n: int) -> list[int]:
        lo, hi = self.cfg["seeds"]["block"]
        taken = expand(self.state["seeds_used"]) | expand(self.cfg["seeds"]["exclude"])
        return [s for s in range(lo, hi + 1) if s not in taken][:n]

    def tick_collector(self, host_name: str) -> None:
        col = self.state["collectors"][host_name]
        if col["batch"] is not None:
            return self.poll_batch(host_name, self.state["batches"][col["batch"]])
        if col.get("done") or self.refreshes_done():
            return
        host, c = self.hosts[host_name], self.cfg["collect"]
        for pattern in c["wait_for_absent"]:
            if self.b.pgrep(host, pattern):
                if not col.get("busy_logged"):
                    col["busy_logged"] = True
                    self.save()
                    self.log("collect_wait_busy", host=host_name, pattern=pattern)
                return
        col.pop("busy_logged", None)
        if not self.disk_ok(f"collect:{host_name}"):
            return
        found = self.newest_ckpt(settle_s=30.0)
        if not found:
            return
        # Seed budget: a SUMO batch takes only minutes, so by default each batch needs a policy checkpoint
        # no earlier batch used (periodic policy_<n>.pt or a new generation) and min_interval_s since the last.
        last = self.state.get("last_batch") or {}
        if c["require_new_policy_ckpt"] and [found[1], found[0]] == last.get("ckpt"):
            return
        if self.b.now() - last.get("ts", -1e18) < c["min_interval_s"]:
            return
        seeds = self.next_seeds(c["batch_size"])
        if not seeds:
            col["done"] = True
            self.save()
            self.log("collector_done", host=host_name, reason="seed block exhausted")
            return
        check_seeds(seeds)
        bid = f"b{self.state['next_batch']:04d}"
        bdir = self.root / "batches" / bid
        bdir.mkdir(parents=True, exist_ok=True)
        snap = bdir / "policy_snapshot.pt"
        self.b.copy(self.hosts["policy"], found[1], self.hosts["wm"], str(snap))
        rdir = f"{self.remote_root(host)}/{bid}"
        if self.b.exists(host, rdir):
            raise RuntimeError(f"{host['name']}:{rdir} exists; never reuse a collection directory")
        # allocation is recorded BEFORE anything runs
        self.state["next_batch"] += 1
        self.state["seeds_used"] = ranges(sorted(expand(self.state["seeds_used"]) | set(seeds)))
        batch = {"id": bid, "host": host_name, "seeds": seeds, "status": "launching", "attempts": 0,
                 "remote_dir": rdir, "snapshot": {"source": found[1], "path": str(snap), "sha256": sha256(snap),
                                                  "policy_run_dir": self.state["policy"]["run_dir"]},
                 "wm_at_start": self.state["wm"]["ckpt"], "created": utcnow()}
        self.state["batches"][bid] = batch
        self.state["last_batch"] = {"ckpt": [found[1], found[0]], "ts": self.b.now(), "batch": bid}
        col["batch"] = bid
        self.save()
        with (self.root / "seed_ledger.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"t": utcnow(), "batch": bid, "host": host_name, "block": ("C4_V6_LONGPAIR" if max(seeds) <= 965999 else "C4_V6_LONGPAIR+C4_LOOP_TRAIN" if min(seeds) <= 965999 else "C4_LOOP_TRAIN"),
                                     "use": "policy-rollout WM training data", "seeds": seed_spec(seeds),
                                     "n": len(seeds)}) + "\n")
        self.log("seeds_allocated", batch=bid, host=host_name, seeds=seed_spec(seeds), n=len(seeds))
        self.b.copy(self.hosts["wm"], str(snap), host, f"{rdir}/policy_snapshot.pt")
        self.launch_batch(batch)

    def launch_batch(self, batch: dict) -> None:
        host = self.hosts[batch["host"]]
        rdir = batch["remote_dir"]
        attempt = batch["attempts"] + 1
        cmd = self.collect_cmd(f"{rdir}/policy_snapshot.pt", batch["seeds"], f"{rdir}/policy", batch["host"])
        env = {"PYTHONPATH": f"{host['code_root']}/src:{host['code_root']}", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        self.b.run(host, f"mkdir -p {q(rdir)} && rm -f {q(rdir + '/EXIT')} && printf %s {q(cmd)} > "
                         f"{q(rdir + f'/collect_cmd.a{attempt}.txt')}", check=True)
        proc = self.b.launch(host, host["code_root"], cmd, f"{rdir}/collect.a{attempt}.log", f"{rdir}/EXIT", env)
        batch.update(attempts=attempt, status="running", proc=proc, launched=utcnow(), launched_ts=self.b.now())
        self.save()
        self.log("batch_launched", batch=batch["id"], host=batch["host"], pid=proc["pid"], attempt=attempt,
                 seeds=seed_spec(batch["seeds"]))

    def poll_batch(self, host_name: str, batch: dict) -> None:
        host, c = self.hosts[host_name], self.cfg["collect"]
        if batch["status"] == "launching":          # crashed between allocation and launch
            return self.launch_batch(batch)
        if batch["status"] == "pulling":
            return self.pull_batch(batch)
        code = self.b.read_text(host, batch["remote_dir"] + "/EXIT")
        if code is not None and code.strip():
            rc = int(code.strip()) if code.strip().lstrip("-").isdigit() else -1
            if rc == 0:
                batch["status"] = "pulling"
                batch["collect_s"] = round(self.b.now() - batch["launched_ts"])
                self.save()
                return self.pull_batch(batch)
            return self.batch_failed(batch, f"exit {rc}")
        if not self.b.alive(host, batch["proc"]):
            code = self.b.read_text(host, batch["remote_dir"] + "/EXIT")
            if code is None or not code.strip():
                return self.batch_failed(batch, "process gone without EXIT file")
            return
        if self.b.now() - batch["launched_ts"] > c["timeout_s"]:
            self.b.signal(host, batch["proc"], "TERM", "", group=True)
            return self.batch_failed(batch, "timeout")

    def pull_batch(self, batch: dict) -> None:
        host = self.hosts[batch["host"]]
        local = self.root / "raw" / batch["id"]
        self.b.copy(host, batch["remote_dir"] + "/policy/", self.hosts["wm"], str(local) + "/")
        try:
            self.b.copy(host, f"{batch['remote_dir']}/collect.a{batch['attempts']}.log", self.hosts["wm"],
                        str(self.root / "batches" / batch["id"] / f"collect.a{batch['attempts']}.log"))
        except (RuntimeError, OSError) as exc:
            self.log("pull_log_failed", batch=batch["id"], error=str(exc)[:300])
        episodes = find_episodes(local)
        need = max(1, math.ceil(self.cfg["collect"]["min_complete_fraction"] * len(batch["seeds"])))
        if len(episodes) < need and batch["attempts"] < self.cfg["collect"]["max_attempts"]:
            return self.batch_failed(batch, f"only {len(episodes)}/{len(batch['seeds'])} complete episodes")
        batch.update(status="pooled" if episodes else "failed", episodes=len(episodes), raw_dir=str(local),
                     pooled=utcnow())
        self.state["collectors"][batch["host"]]["batch"] = None
        self.save()
        self.log("batch_pooled", batch=batch["id"], host=batch["host"], episodes=len(episodes),
                 collect_s=batch.get("collect_s"))

    def batch_failed(self, batch: dict, why: str) -> None:
        self.log("batch_failed", batch=batch["id"], host=batch["host"], why=why, attempt=batch["attempts"])
        if batch["attempts"] < self.cfg["collect"]["max_attempts"]:
            return self.launch_batch(batch)          # same seeds and directory: collect resumes
        batch["status"] = "failed"
        self.state["collectors"][batch["host"]]["batch"] = None
        self.save()
        self.log("batch_given_up", batch=batch["id"], seeds=seed_spec(batch["seeds"]))

    # -- WM lane
    def tick_wm(self) -> None:
        lane = self.state["wm_lane"]
        if lane["refresh"] is None:
            return self.start_refresh()
        r = self.state["refreshes"][str(lane["refresh"])]
        if r["status"] == "converting":
            return self.poll_convert(r)
        if r["status"] == "training":
            return self.poll_refresh(r)

    def start_refresh(self) -> None:
        if self.state["refresh_n"] >= self.cfg["wm"]["max_refreshes"]:
            return
        episodes = find_episodes(self.root / "raw")
        pool = len(episodes)
        last = self.state["refreshes"].get(str(self.state["refresh_n"]), {})
        new = pool - last.get("pool_episodes", 0)
        if pool == 0:
            return
        if self.state["refresh_n"] > 0 and new < self.cfg["wm"]["min_new_episodes"]:
            return
        collecting = any(not c.get("done") or c["batch"] is not None for c in self.state["collectors"].values())
        if new <= 0 and not collecting:
            return                      # seed block used up and the whole pool already refreshed: idle
        if not self.disk_ok("wm refresh"):
            return
        cv = self.cfg["convert"]
        n = self.state["refresh_n"] + 1
        dest = Path(self.cfg["data_root"]) / f"{cv['prefix']}{n:03d}"
        refuse_restricted(dest)
        if dest.exists():
            if str(dest) in self.state.get("convert_attempt_dirs", []):
                shutil.rmtree(dest)
            else:
                raise RuntimeError(f"{dest} exists and was not created by this loop; refusing")
        self.cleanup_converted()
        if cv["max_episodes"] and len(episodes) > cv["max_episodes"]:
            episodes = episodes[-cv["max_episodes"]:]
        free = self.b.free_gb(dest.parent)
        fit = self.window_fit(free)
        if fit < len(episodes):
            if fit < max(1, min(len(episodes), cv.get("min_window", 50))):
                if self.state.get("status") != "paused_disk":
                    self.state["status"] = "paused_disk"
                    self.save()
                    self.log("disk_pause", free_gb=round(free, 2), fit_episodes=fit)
                return
            self.log("convert_window_reduced", episodes=len(episodes), kept_recent=fit, free_gb=round(free, 2))
            episodes = episodes[-fit:]
        rdir = self.root / "refreshes" / f"r{n:03d}"
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "convert_episodes.txt").write_text("\n".join(episodes) + "\n", encoding="utf-8")
        wm = self.hosts["wm"]
        prefix = f"nice -n {int(cv['nice'])} " if cv.get("nice") else ""
        cmd = prefix + convert_cmd(wm["python"], rdir / "convert_episodes.txt", dest)
        self.state.setdefault("convert_attempt_dirs", []).append(str(dest))
        self.state["refresh_n"] = n
        r = {"n": n, "status": "converting", "converted": str(dest), "pool_episodes": pool,
             "episodes": len(episodes), "batches": sorted(b for b, v in self.state["batches"].items()
                                                          if v["status"] == "pooled"),
             "started": utcnow(), "started_ts": self.b.now(), "dir": str(rdir)}
        self.state["refreshes"][str(n)] = r
        self.state["wm_lane"]["refresh"] = n
        self.save()
        proc = self.b.launch(wm, wm["code_root"], cmd, str(rdir / "convert.log"), str(rdir / "CONVERT_EXIT"),
                             {"PYTHONPATH": "src:.", "OMP_NUM_THREADS": "4"})
        r["convert_proc"] = proc
        self.save()
        self.log("refresh_convert", refresh=n, episodes=len(episodes), dest=str(dest), pid=proc["pid"])

    def poll_convert(self, r: dict) -> None:
        wm = self.hosts["wm"]
        exit_file = Path(r["dir"]) / "CONVERT_EXIT"
        if exit_file.exists() and exit_file.read_text().strip():
            rc = exit_file.read_text().strip()
            dest = Path(r["converted"])
            if rc != "0" or not (dest / "C4_EPISODES_MANIFEST.json").exists():
                return self.refresh_failed(r, f"conversion failed (exit {rc}); see {r['dir']}/convert.log")
            self.state["converted_dirs"].append(str(dest))
            r["convert_s"] = round(self.b.now() - r["started_ts"])
            self.save()
            return self.launch_refresh(r)
        if not self.b.alive(wm, r["convert_proc"]) and not (exit_file.exists() and exit_file.read_text().strip()):
            return self.refresh_failed(r, "conversion process gone without exit code")
        if self.b.now() - r["started_ts"] > self.cfg["wm"]["convert_timeout_s"]:
            self.b.signal(wm, r["convert_proc"], "TERM", "", group=True)
            return self.refresh_failed(r, "conversion timeout")

    def launch_refresh(self, r: dict) -> None:
        host = self.hosts["wm"]
        base = Path(self.cfg["wm_runs_root"]) / f"c4_wm_loop_r{r['n']:03d}"
        out, k = base, 1
        while out.exists() or Path(str(out) + ".stdout.log").exists():
            k += 1
            out = Path(f"{base}.a{k}")
        env = {"PYTHONPATH": "src:.", "OMP_NUM_THREADS": "12", "WMS_JAM_WORKERS": "14", "WMS_LATEST_EVERY": "100",
               **self.cfg["wm"]["env"], "WMS_DATA_V6_POLICY": r["converted"]}
        cmd = self.wm_cmd(str(out), self.state["wm"]["ckpt"])
        proc = self.b.launch(host, host["code_root"], cmd, f"{out}.stdout.log", f"{out}.EXIT", env)
        Path(f"{out}.pid").write_text(f"{proc['pid']}\n", encoding="utf-8")
        r.update(status="training", out=str(out), proc=proc, cmd=cmd, env=env, init_from=self.state["wm"]["ckpt"],
                 ref_val_log=self.state["wm"].get("val_log"), train_started_ts=self.b.now(), train_started=utcnow())
        self.save()
        self.log("refresh_launched", refresh=r["n"], out=str(out), pid=proc["pid"], init_from=r["init_from"],
                 episodes=r["episodes"])

    def poll_refresh(self, r: dict) -> None:
        host, w = self.hosts["wm"], self.cfg["wm"]
        out = Path(r["out"])
        if not (out / "RUN_SUMMARY.json").exists():
            if not self.b.alive(host, r["proc"]):
                if not (out / "RUN_SUMMARY.json").exists():
                    return self.refresh_failed(r, "WM process ended without RUN_SUMMARY.json")
            elif self.b.now() - r["train_started_ts"] > w["timeout_s"]:
                self.b.signal(host, r["proc"], "TERM", "", group=True)
                return self.refresh_failed(r, "WM refresh timeout")
            else:
                return
        r["train_s"] = round(self.b.now() - r["train_started_ts"])
        best = out / "wm_s_best.pt"
        if not best.exists() or not (out / "physics.json").exists():
            return self.refresh_failed(r, "wm_s_best.pt or physics.json missing")
        verdict = wm_sanity(out / "train_log.jsonl", out / "val_log.jsonl", r.get("ref_val_log"),
                            w["val_metric"], w["val_tolerance"])
        if not verdict["ok"]:
            return self.refresh_failed(r, "; ".join(verdict["reasons"]), compare=verdict["compare"])
        policy_ckpt = str(best)
        if w["lane_calib"]:
            policy_ckpt = self.lane_calib(out, best) or policy_ckpt
        self.state["wm"].update(ckpt=str(best), policy_ckpt=policy_ckpt, val_log=str(out / "val_log.jsonl"))
        self.state["wm"]["history"].append({"refresh": r["n"], "accepted": True, "out": str(out),
                                            "compare": verdict["compare"]})
        r.update(status="accepted", verdict=verdict, finished=utcnow())
        self.state["policy_lane"]["pending"] = {"refresh": r["n"], "wm": policy_ckpt, "data_dir": r["converted"],
                                                "queued": utcnow()}
        self.state["wm_lane"]["refresh"] = None
        self.save()
        self.log("wm_accepted", refresh=r["n"], ckpt=str(best), compare=verdict["compare"], train_s=r["train_s"],
                 note=verdict.get("note"))

    def refresh_failed(self, r: dict, why: str, **data) -> None:
        r.update(status="rejected", why=why, finished=utcnow(), **data)
        self.state["wm"]["history"].append({"refresh": r["n"], "accepted": False, "out": r.get("out"), "why": why})
        self.state["wm_lane"]["refresh"] = None
        self.save()
        self.log("wm_rejected", refresh=r["n"], why=why, kept=self.state["wm"]["ckpt"], **data)

    # -- policy lane
    def ensure_policy(self, reason: str) -> None:
        if self.state["policy_lane"]["stop"] is not None or self.policy_alive():
            return
        pol = self.state["policy"]
        if self.state["policy_restarts"] >= self.cfg["policy"]["max_restarts"]:
            if not pol.get("restart_exhausted"):
                pol["restart_exhausted"] = True
                self.save()
                self.log("policy_dead_no_restart", run_dir=pol["run_dir"], restarts=self.state["policy_restarts"])
            return
        found = self.newest_ckpt(settle_s=5.0)
        if not found:
            return
        wm, data = pol.get("wm_local", self.state["wm"]["policy_ckpt"]), self.policy_data_dir()
        quick = pol.get("owned") and pol.get("started_ts") and self.b.now() - pol["started_ts"] < \
            3 * self.cfg["policy"]["start_check_s"]
        if quick and pol.get("prev_wm_local"):
            # died right after a WM switch: fall back to the previous WM (and its data) and keep training
            wm, data = pol["prev_wm_local"], pol.get("prev_data_dir")
            self.log("policy_switch_fallback", run_dir=pol["run_dir"], wm=wm)
        self.state["policy_restarts"] += 1
        self.save()
        self.log("policy_restart", reason=reason, resume=found[1], wm=wm)
        self.launch_policy(found[1], wm, data, tag=f"r{self.state['policy_restarts']}")

    def launch_policy(self, resume: str, wm: str, data_dir: str | None, tag: str = "") -> None:
        prev = self.state["policy"]
        self.state["policy_gen"] += 1
        self.start_policy(resume, wm, data_dir, tag=tag, name=f"c4_policy_loop_g{self.state['policy_gen']:03d}")
        self.state["policy"].update(started_ts=self.b.now(), prev_wm_local=prev.get("wm_local"),
                                    prev_data_dir=prev.get("data_dir"))
        self.save()

    def tick_policy(self) -> None:
        lane, cfg, host = self.state["policy_lane"], self.cfg["policy"], self.hosts["policy"]
        pending = lane["pending"]
        if pending is None and lane["stop"] is None:
            return
        pol = self.state["policy"]
        stop = lane["stop"]
        if stop is None:
            stop = lane["stop"] = {"requested": utcnow(), "requested_ts": self.b.now(), "run_dir": pol["run_dir"],
                                   "how": "stop_file" if pol.get("stop_file") else "adopted"}
            if self.policy_alive():
                if pol.get("stop_file"):
                    self.b.run(host, f"touch {q(pol['run_dir'].rstrip('/') + '/STOP')}", check=True)
                else:
                    stop["baseline_mtime"] = (self.newest_ckpt(settle_s=0.0) or (0.0, ""))[0]
            self.save()
            self.log("policy_stop_requested", how=stop["how"], run_dir=pol["run_dir"], for_refresh=pending["refresh"])
        resume = None
        if self.policy_alive():
            waited = self.b.now() - stop["requested_ts"]
            if stop["how"] == "stop_file":
                if waited <= cfg["stop_timeout_s"]:
                    return
                self.log("policy_stop_timeout", action="SIGTERM")
                self.terminate_policy()
            else:
                periodic = [(m, p) for m, s, p in self.b.list_ckpts(host, pol["run_dir"])
                            if m > stop["baseline_mtime"] and _update_of(p) is not None and s > 0]
                if periodic and self._updates_reached(_update_of(max(periodic)[1]) + 1):
                    resume = max(periodic)[1]
                elif waited <= cfg["adopted_wait_max_s"]:
                    return
                else:
                    self.log("policy_adopted_wait_timeout")
                self.terminate_policy()
        if resume is None:
            latest = pol["run_dir"].rstrip("/") + "/policy_latest.pt"
            found = self.newest_ckpt(settle_s=0.0)
            if stop["how"] == "stop_file" and self.b.exists(host, latest):
                resume = latest
            elif found:
                resume = found[1]
        if not resume:
            raise RuntimeError("no checkpoint to resume the policy from")
        pending = lane["pending"] or {"wm": pol.get("wm_local", self.state["wm"]["policy_ckpt"]),
                                      "data_dir": pol.get("data_dir"), "refresh": None}
        stop.update(resume=resume, stopped=utcnow())
        self.state.setdefault("policy_switches", []).append({**stop, "refresh": pending["refresh"]})
        lane["stop"], lane["pending"] = None, None
        self.state["policy_restarts"] = 0
        self.save()
        self.launch_policy(resume, pending["wm"], pending["data_dir"])
        self.log("policy_switched", refresh=pending["refresh"], resume=resume, run_dir=self.state["policy"]["run_dir"])
        self.cleanup_converted()

    def cleanup_converted(self) -> None:
        """Keep the converted pools still read by the running policy, a queued switch or the active refresh."""
        pending = self.state["policy_lane"]["pending"] or {}
        active = self.state["refreshes"].get(str(self.state["wm_lane"]["refresh"] or ""), {})
        self.delete_converted(keep={self.state["policy"].get("data_dir"), pending.get("data_dir"),
                                    active.get("converted")})

    # -- dry run
    def plan(self) -> list[str]:
        c = self.cfg["collect"]
        seeds = self.next_seeds(c["batch_size"] * len(c["hosts"]))
        lines = [f"PIPELINED: collectors {c['hosts']} x batches of {c['batch_size']} episodes; "
                 f"WM refresh stage_steps {self.cfg['wm']['stage_steps']} back to back; policy switches per accepted WM"]
        for i, host in enumerate(c["hosts"]):
            part = seeds[i * c["batch_size"]:(i + 1) * c["batch_size"]]
            rdir = f"{self.hosts[host]['work_root'].rstrip('/')}/b{i + 1:04d}"
            lines.append(f"  {host}: seeds {seed_spec(part)} -> " +
                         self.collect_cmd(f"{rdir}/policy_snapshot.pt", part, f"{rdir}/policy", host))
        lines.append("  WM: convert pool <loop_root>/raw/b* -> " + f"{self.cfg['data_root']}/{self.cfg['convert']['prefix']}NNN; "
                     + self.wm_cmd(f"{self.cfg['wm_runs_root']}/c4_wm_loop_r001", self.state["wm"]["ckpt"]))
        lines.append("  policy: " + ("STOP file" if self.state["policy"].get("stop_file") else
                                     f"adopted pid {(self.state['policy'].get('proc') or {}).get('pid')}: next "
                                     "policy_<n>.pt + 1 update, then SIGTERM")
                     + f" -> WMS_LOOP_REFRESH=1 train_cli --config {self.cfg['policy']['config']} --resume ... "
                     f"--wm-ckpt <accepted WM> (WMS_DATA_V6_POLICY=<that refresh's converted pool>)")
        return lines


def _update_of(path: str) -> int | None:
    name = os.path.basename(path)
    if name.startswith("policy_") and name.endswith(".pt") and name[7:-3].isdigit():
        return int(name[7:-3])
    return None


def dir_size_gb(path) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / 1e9


def convert_cmd(python: str, list_file, dest) -> str:
    """``wms.data.c4_episodes`` over the episode dirs listed in ``list_file`` (no huge argv)."""
    code = ("import sys; from wms.data.c4_episodes import main; "
            "sys.exit(main(['--episodes', *open(sys.argv[1]).read().split(), '--out', sys.argv[2]]))")
    return f"{q(python)} -B -c {q(code)} {q(list_file)} {q(dest)}"


def find_episodes(root) -> list[str]:
    """Complete raw episode dirs (episode.npz + manifest status complete + summary.json), sorted."""
    found = []
    for path, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if ".incomplete-" not in d and not d.startswith("c4_episodes")
                         and not any(t in d.lower() for t in FORBIDDEN))
        if all(n in files for n in ("episode.npz", "manifest.json", "summary.json")):
            try:
                if json.loads((Path(path) / "manifest.json").read_text()).get("status") == "complete":
                    found.append(str(Path(path)))
            except (OSError, ValueError):
                pass
    return sorted(found)


# --------------------------------------------------------------------------- CLI
def status(loop: Loop, live: bool) -> int:
    if not loop.load():
        print(f"no state at {loop.state_path}")
        return 1
    s = loop.state
    pol = s["policy"]
    mode = s.get("mode", "sequential")
    print(f"mode={mode} status={s['status']} last_error={s.get('last_error')} STOP={loop.stop_path.exists()}")
    print(f"wm={s['wm']['ckpt']}\npolicy_wm={s['wm']['policy_ckpt']}")
    print(f"policy run_dir={pol['run_dir']} pid={(pol.get('proc') or {}).get('pid')} owned={pol.get('owned')} "
          f"data_dir={pol.get('data_dir')}")
    print(f"seeds_used={seed_spec(sorted(expand(s['seeds_used'])))} converted={s['converted_dirs']}")
    if mode == "pipelined":
        for host, c in s["collectors"].items():
            b = s["batches"].get(c["batch"] or "", {})
            print(f"  collector {host}: batch={c['batch']} {b.get('status', '')} seeds={seed_spec(b.get('seeds', []))} "
                  f"done={c.get('done')}")
        pooled = [b for b in s["batches"].values() if b["status"] == "pooled"]
        print(f"  pool: {len(pooled)} batches, {sum(b.get('episodes', 0) for b in pooled)} episodes; "
              f"failed batches {sum(b['status'] == 'failed' for b in s['batches'].values())}")
        r = s["refreshes"].get(str(s["wm_lane"]["refresh"] or ""), {})
        print(f"  WM lane: refresh={s['wm_lane']['refresh']} {r.get('status', 'idle')} out={r.get('out')}; "
              f"refreshes so far {s['refresh_n']}, accepted "
              f"{sum(v['status'] == 'accepted' for v in s['refreshes'].values())}")
        print(f"  policy lane: pending={s['policy_lane']['pending']} stop={s['policy_lane']['stop']} "
              f"switches={len(s.get('policy_switches', []))}")
    else:
        print(f"iteration={s['iteration']}/{loop.cfg['max_iterations']} step={s['step']}")
        for name, t in s["iters"].get(str(s["iteration"]), {}).get("steps", {}).items():
            print(f"  {name:15s} {t.get('start', '')} -> {t.get('end', '')}")
    if live:
        print(f"policy alive={loop.policy_alive()}")
        print(f"free_gb={min(loop.b.free_gb(p) for p in (loop.cfg.get('disk_paths') or [str(loop.root)])):.1f}")
    for row in read_jsonl(loop.log_path)[-8:]:
        print("  log", json.dumps(row)[:300])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("run", "status", "dry-run", "stop"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--sequential", action="store_true", help="the simple turn-based loop (fallback)")
    ap.add_argument("--init-wm", help="WM checkpoint the first refresh starts from")
    ap.add_argument("--init-wm-val-log", help="its val_log.jsonl (reference for the sanity check)")
    ap.add_argument("--init-policy-wm", help="WM the running policy uses (default --init-wm)")
    ap.add_argument("--init-policy-run-dir")
    ap.add_argument("--adopt-policy-pid", type=int)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--max-steps", type=int, help="testing: return after N steps/ticks")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.sequential:
        cfg["mode"] = "sequential"
        cfg = load_config_dict(cfg)
    cls = PipeLoop if cfg["mode"] == "pipelined" else Loop
    loop = cls(cfg)
    if loop.state_path.exists():
        saved = json.loads(loop.state_path.read_text(encoding="utf-8")).get("mode", "sequential")
        if saved != cfg["mode"] and args.action in ("run", "dry-run"):
            print(f"state at {loop.state_path} is {saved}; config/flags say {cfg['mode']} (use another loop_root)")
            return 1
    if args.action == "status":
        return status(loop, args.live)
    if args.action == "stop":
        loop.root.mkdir(parents=True, exist_ok=True)
        loop.stop_path.write_text(utcnow() + "\n", encoding="utf-8")
        print(f"STOP written: {loop.stop_path} (the loop finishes its current step/tick and exits)")
        return 0
    boot = cfg.get("bootstrap", {})
    init = (args.init_wm or boot.get("wm_ckpt"), args.init_wm_val_log or boot.get("wm_val_log"),
            args.init_policy_run_dir or boot.get("policy_run_dir"), args.adopt_policy_pid or boot.get("adopt_policy_pid"),
            args.init_policy_wm or boot.get("policy_wm"))
    if args.action == "dry-run":
        if not loop.load():
            loop.state = {"iteration": 1, "step": "snapshot", "seeds_used": [], "converted_dirs": [],
                          "wm": {"ckpt": init[0], "val_log": init[1]},
                          "policy": {"run_dir": init[2], "proc": {"pid": init[3]} if init[3] else None,
                                     "stop_file": False}}
        print("\n".join(loop.plan()))
        return 0
    if loop.stop_path.exists():
        print(f"STOP file present ({loop.stop_path}); remove it to run")
        return 1
    if not loop.load():
        loop.init_state(*init)
    elif args.init_wm:
        print("state exists; --init-* flags ignored (use a new loop_root to start over)")
    return loop.run(max_steps=args.max_steps)


if __name__ == "__main__":
    sys.exit(main())

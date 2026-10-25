#!/usr/bin/env python3
"""LOOP14 Dreamer-style cycle orchestrator (stdlib only; LOOP14_DESIGN.md).

One cycle k = five phases, each a SEPARATE command with fail-closed gates:

  policy   gpuR  policy14 trainer, N updates, WM FROZEN (``policy14_cli``; cycle 0 trains from
                 scratch on the initial WM unless an existing policy is adopted at init; later
                 cycles ``--resume`` on the currently ACCEPTED WM with the previous cycle's
                 current-policy SUMO returns)
  collect  cpuN  SUMO: native base per fresh registered TRAIN seed + learned-policy and native
                 arms from the same saved states (``collect --policy-pairs``)
  convert  cpuN  ``wms.data.c4_episodes`` + ``wms.control.loop14_returns`` (real critic
                 returns), then pull the converted pairs + returns to the WM host
  wm_ft    gpuR  WM fine-tune from the ACCEPTED WM (``configs/c4_wm_loop14_ft.toml``: core
                 frozen; residual/courtesy/yield trained; D1 on; R14 D3 mix + collected pairs)
  check    gpuR  CPU acceptance vs the accepted WM (``tools/loop14_checks/accept.py``); ACCEPT
                 makes the candidate the accepted WM, REJECT keeps the previous one

    python scripts/c4/loop14.py init   --config CFG [--adopt-policy CKPT]
    python scripts/c4/loop14.py status --config CFG
    python scripts/c4/loop14.py plan   --config CFG             # next phase command, no execution
    python scripts/c4/loop14.py run    --config CFG PHASE       # one phase of the current cycle
    python scripts/c4/loop14.py next   --config CFG             # the next pending phase
    python scripts/c4/loop14.py retry  --config CFG PHASE       # failed phase -> new attempt dir

State: ``<loop_root>/LOOP14_STATE.json`` (atomic rewrite after every transition); ledger:
``<loop_root>/cycle_ledger.jsonl`` (one line per phase outcome + one per cycle decision) and
``<loop_root>/seed_ledger.jsonl`` (seed allocations, written BEFORE collection starts).
Guards: registration (status registered + acknowledged, sha pinned at init), budgets, GPU-busy
(any foreign compute process refuses; nothing is ever signalled), disk (free space minus the
phase estimate must stay above ``min_free_gb``), restricted path tokens, evaluation seeds
964000-964199.  Output directories are never reused (``.aN`` attempts).  Exit codes: 0 ok,
2 refused by a gate, 3 phase failed, 4 WM update rejected (cycle still completes), 5 GPU busy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASES = ("policy", "collect", "convert", "wm_ft", "check")
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth")
EVAL_SEEDS = (964000, 964199)
STATE = "LOOP14_STATE.json"


class Refused(Exception):
    """A fail-closed gate refused the phase (exit 2)."""


class GpuBusy(Refused):
    pass


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def guard(value) -> str:
    text = str(value)
    if any(t in text.lower() for t in FORBIDDEN):
        raise Refused(f"restricted path refused: {text}")
    return text


def sha256(path) -> str:
    with open(guard(path), "rb") as stream:
        h = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
        return h.hexdigest()


def read_json(path):
    return json.loads(Path(guard(path)).read_text(encoding="utf-8"))


def write_atomic(path: Path, doc) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def append(path: Path, doc) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"t": now(), **doc}) + "\n")


# ------------------------------------------------------------------ config / registration
def load_config(path) -> dict:
    cfg = read_json(path)
    for key in ("loop_id", "loop_root", "registration", "hosts", "policy", "collect", "convert",
                "wm", "check", "guards"):
        if key not in cfg:
            raise Refused(f"config needs {key}")
    for value in _strings(cfg):
        guard(value)
    for host in ("gpu", "cpu"):
        if host not in cfg["hosts"]:
            raise Refused(f"config needs hosts.{host}")
    return cfg


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def check_registration(cfg) -> dict:
    reg = read_json(cfg["registration"])
    if reg.get("status") != "registered" or reg.get("split") != "TRAIN":
        raise Refused("registration must be status 'registered', split 'TRAIN'")
    if not str(reg.get("acknowledged_by", "")).strip():
        raise Refused("registration must be acknowledged (acknowledged_by) before any run")
    if reg.get("id") != cfg["loop_id"]:
        raise Refused("registration id != loop_id")
    lo, hi = (int(x) for x in reg["seed_block"])
    if lo > hi or not (hi < EVAL_SEEDS[0] or lo > EVAL_SEEDS[1]):
        raise Refused("seed block overlaps the evaluation seeds or is empty")
    b = reg["budget"]
    for key in ("max_cycles", "policy_updates_per_cycle", "sim_s_per_cycle", "sim_s_total",
                "gpu_h_policy_per_cycle", "gpu_h_wm_per_cycle", "wm_steps_per_cycle"):
        if not isinstance(b.get(key), (int, float)) or b[key] <= 0:
            raise Refused(f"registration budget.{key} must be positive")
    return reg


def collect_sim_s(cfg) -> int:
    c = cfg["collect"]
    base = int(c["branch_first"]) + (int(c["branch_points"]) - 1) * int(c["branch_cadence"]) + 1
    extra = int(c["branch_points"]) * sumocredit_sim_s_per_point(c) if c.get("sumocredit") else 0
    return int(c["worlds_per_cycle"]) * (base + int(c["branch_points"]) * 2 * int(c["branch_horizon"])
                                         + extra)


# ------------------------------------------------------------------ S-SUMOCREDIT (flag-gated)
SUMOCREDIT_FLAGS = {"credit_horizon": "--credit-horizon", "credit_group_max": "--credit-group-max",
                    "credit_flips": "--credit-flips", "gate_horizon": "--gate-horizon",
                    "gate_unit": "--gate-unit", "gate_reference": "--gate-reference",
                    "gate_state": "--gate-state", "gate_max_units": "--gate-max-units",
                    "gate_candidates": "--gate-candidates",
                    "gate_decision_period": "--gate-decision-period",
                    "gate_driver_samples": "--gate-driver-samples", "seed": "--sumocredit-seed"}


def sumocredit_sim_s_per_point(c) -> int:
    """Upper bound of the extra SUMO seconds per decision point (= traffic_shepherd.c4.sumocredit
    .sim_s_per_point; the gated arms are charged whenever gated_arm is on, also before a table
    exists: GATED, plus GATED_K when gate_candidates > 1)."""
    x, h = c["sumocredit"], int(c["branch_horizon"])
    total = 0
    g, cr = int(x.get("gate_horizon", 0)), int(x.get("credit_horizon", 0))
    k = int(x.get("gate_candidates", 1))
    if g:
        unit = x.get("gate_unit", "fleet")
        total += 0 if (x.get("gate_reference", "native") == "native" and g <= h) else g
        n_units = 1 if unit == "fleet" else int(x.get("gate_max_units", 4))
        total += n_units * k * g - (g if (unit == "fleet" and g <= h) else 0)
    if x.get("gated_arm"):
        total += h * (2 if k > 1 else 1)
    if cr:
        total += (int(x.get("credit_group_max", 3)) + 1 + int(x.get("credit_flips", 0))) * cr
        total += 0 if cr <= h else 2 * cr
    return total


def sumocredit_args(cfg, state) -> list:
    x = cfg["collect"].get("sumocredit")
    if not x:
        return []
    out = []
    for key, flag in SUMOCREDIT_FLAGS.items():
        if key in x:
            out += [flag, str(x[key])]
    if x.get("credit_imagine") is False:
        out.append("--no-credit-imagine")
    return out


def check_budget(cfg, reg, state) -> None:
    b = reg["budget"]
    if int(cfg["policy"]["updates_per_cycle"]) > b["policy_updates_per_cycle"]:
        raise Refused("policy.updates_per_cycle exceeds the registered budget")
    if collect_sim_s(cfg) > b["sim_s_per_cycle"]:
        raise Refused(f"collection {collect_sim_s(cfg)} sim-s exceeds the registered per-cycle budget")
    if state["sim_s_charged"] + collect_sim_s(cfg) > b["sim_s_total"] and _phase(state)["name"] == "collect":
        raise Refused("registered total SUMO budget exhausted")
    if float(cfg["policy"]["timeout_h"]) > b["gpu_h_policy_per_cycle"] or float(cfg["wm"]["timeout_h"]) > b["gpu_h_wm_per_cycle"]:
        raise Refused("phase timeouts exceed the registered GPU-hour budget")


# ------------------------------------------------------------------ hosts
def host(cfg, name) -> dict:
    h = dict(cfg["hosts"][name])
    h.setdefault("python", sys.executable)
    h.setdefault("code_root", str(Path(__file__).resolve().parents[2]))
    return h


def run_cmd(h: dict, argv, *, env: dict, log: Path, timeout_s: float, cwd=None) -> int:
    """Foreground command (own child only; on timeout only that child is stopped)."""
    cwd = cwd or h["code_root"]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"# {now()} {' '.join(map(shlex.quote, map(str, argv)))}\n")
        stream.flush()
        if h.get("ssh"):
            exports = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items())
            remote = f"cd {shlex.quote(str(cwd))} && env {exports} {' '.join(map(shlex.quote, map(str, argv)))}"
            full = [*h["ssh"], remote]
            proc_env = None
        else:
            full, proc_env = [str(a) for a in argv], {**os.environ, **{k: str(v) for k, v in env.items()}}
        try:
            return subprocess.run(full, cwd=None if h.get("ssh") else cwd, env=proc_env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout_s, check=False).returncode
        except subprocess.TimeoutExpired:
            stream.write(f"# {now()} TIMEOUT after {timeout_s:.0f} s (own child stopped)\n")
            return 124


def sync(cfg, src_host: str, src: str, dst_host: str, dst: str, log: Path) -> None:
    """Copy a directory/file between hosts (rsync over the configured ssh; local = copytree)."""
    s, d = host(cfg, src_host), host(cfg, dst_host)
    if not s.get("ssh") and not d.get("ssh"):
        if Path(src).resolve() == Path(dst).resolve():
            return
        if Path(src).is_dir():
            shutil.copytree(src, dst)
        else:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return
    remote = s if s.get("ssh") else d
    target = remote["rsync_target"]          # e.g. "ubuntu@cpuN" (ssh alias resolved by ssh)
    rsh = " ".join(map(shlex.quote, remote.get("rsync_ssh", ["ssh"])))
    a = f"{target}:{src}" if s.get("ssh") else src
    b = f"{target}:{dst}" if d.get("ssh") else dst
    with log.open("a", encoding="utf-8") as stream:
        code = subprocess.run(["rsync", "-a", "--partial", "-e", rsh, a, b], stdout=stream,
                              stderr=subprocess.STDOUT, check=False, timeout=3 * 3600).returncode
    if code != 0:
        raise RuntimeError(f"rsync {a} -> {b} exit {code}")


# ------------------------------------------------------------------ guards
def gpu_guard(cfg, h: dict) -> dict:
    """Refuse when ANY compute process is on the GPU (13E training, scoring, ...)."""
    query = cfg["guards"].get("gpu_query", ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                            "--format=csv,noheader"])
    full = [*h["ssh"], " ".join(map(shlex.quote, query))] if h.get("ssh") else query
    try:
        proc = subprocess.run(full, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Refused(f"GPU query failed (fail closed): {error}")
    if proc.returncode != 0:
        raise Refused(f"GPU query exit {proc.returncode} (fail closed)")
    busy = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if busy:
        raise GpuBusy(f"GPU busy: {busy}")
    return {"gpu_query": query, "compute_apps": 0}


def disk_guard(cfg, h: dict, path: str, need_gb: float) -> dict:
    floor = float(cfg["guards"]["min_free_gb"])
    if h.get("ssh"):
        proc = subprocess.run([*h["ssh"], f"df -Pk {shlex.quote(path)} | tail -1"], capture_output=True,
                              text=True, timeout=60, check=False)
        if proc.returncode != 0:
            raise Refused("remote df failed (fail closed)")
        free_gb = int(proc.stdout.split()[3]) / 1024 ** 2
    else:
        probe = Path(path)
        while not probe.exists():
            probe = probe.parent
        free_gb = shutil.disk_usage(probe).free / 1024 ** 3
    if free_gb - need_gb < floor:
        raise Refused(f"disk guard: {free_gb:.2f} GB free - {need_gb:.2f} GB needed < {floor} GB floor at {path}")
    return {"path": path, "free_gb": round(free_gb, 3), "need_gb": round(need_gb, 3), "floor_gb": floor}


# ------------------------------------------------------------------ state
def state_path(cfg) -> Path:
    return Path(guard(cfg["loop_root"])) / STATE


def load_state(cfg) -> dict:
    path = state_path(cfg)
    if not path.is_file():
        raise Refused("loop not initialised (run init)")
    state = json.loads(path.read_text(encoding="utf-8"))
    reg = check_registration(cfg)
    if sha256(cfg["registration"]) != state["registration_sha256"]:
        raise Refused("registration changed after init (sha mismatch)")
    check_budget(cfg, reg, state)
    return state


def save_state(cfg, state) -> None:
    write_atomic(state_path(cfg), state)


def new_cycle(k: int) -> dict:
    return {"cycle": k, "phases": {p: {"status": "pending", "attempt": 0} for p in PHASES}}


def _phase(state) -> dict:
    cyc = state["cycles"][-1]
    for name in PHASES:
        ph = cyc["phases"][name]
        if ph["status"] != "done":
            return {"name": name, **ph}
    return {"name": None}


def out_dir(root: Path, name: str, attempt: int) -> Path:
    return root / (name if attempt <= 1 else f"{name}.a{attempt}")


def init(cfg, args) -> int:
    reg = check_registration(cfg)
    root = Path(guard(cfg["loop_root"]))
    if state_path(cfg).exists():
        raise Refused("loop already initialised (state exists); resume with next/run")
    root.mkdir(parents=True, exist_ok=False)
    wm = cfg["wm"]
    for key in ("initial_wm", "initial_wm_config"):
        if not Path(guard(wm[key])).is_file():
            raise Refused(f"wm.{key} missing: {wm[key]}")
    state = {"schema": "loop14_state/1", "loop_id": cfg["loop_id"], "created": now(),
             "registration": str(cfg["registration"]), "registration_sha256": sha256(cfg["registration"]),
             "accepted_wm": {"checkpoint": wm["initial_wm"], "config": wm["initial_wm_config"],
                             "sha256": sha256(wm["initial_wm"]), "source": "initial"},
             "policy": None, "real_returns": cfg["policy"].get("initial_real_returns"),
             "sim_s_charged": 0, "seeds_used": [], "consecutive_rejects": 0, "stopped": None,
             "cycles": [new_cycle(0)]}
    if args.adopt_policy:
        state["policy"] = {"checkpoint": str(guard(args.adopt_policy)), "sha256": sha256(args.adopt_policy),
                           "source": "adopted", "trained_on_wm_sha256": None}
        state["cycles"][0]["phases"]["policy"].update(status="done", adopted=True, finished=now())
    check_budget(cfg, reg, state)
    save_state(cfg, state)
    append(root / "cycle_ledger.jsonl", {"event": "init", "loop_id": cfg["loop_id"],
                                         "registration_sha256": state["registration_sha256"],
                                         "accepted_wm": state["accepted_wm"], "policy": state["policy"]})
    print(json.dumps({"initialised": str(root), "next": _phase(state)["name"]}))
    return 0


# ------------------------------------------------------------------ phase commands
def policy_cmd(cfg, state, out: Path) -> tuple[list, dict]:
    p, h = cfg["policy"], host(cfg, "gpu")
    k = state["cycles"][-1]["cycle"]
    acc = state["accepted_wm"]
    if not state.get("real_returns"):
        raise Refused("no real-return file for the critic (policy.initial_real_returns)")
    argv = [h["python"], "-m", "wms.control.policy14_cli", "--config", p["config"],
            "--wm-ckpt", acc["checkpoint"], "--wm-config", acc["config"],
            "--real-returns", state["real_returns"], "--device", p["device"],
            "--seed", str(int(p["seed"]) + k), "--max-updates", str(int(p["updates_per_cycle"])),
            "--output-dir", str(out), "--compact"]
    if state.get("policy"):
        argv += ["--resume", state["policy"]["checkpoint"]]
        argv += ["--critic-warmup-updates", str(int(p.get("critic_warmup_updates_resume", 0)))]
    if p.get("smoke"):
        argv.append("--smoke")
    if p.get("real_credit") and state.get("real_credit"):        # S-SUMOCREDIT (flag-gated)
        argv += ["--real-credit", state["real_credit"]]
    env = {"PYTHONPATH": os.pathsep.join([str(Path(h["code_root"]) / "src"), h["code_root"]]),
           "OMP_NUM_THREADS": p.get("threads", 2), "MKL_NUM_THREADS": p.get("threads", 2),
           "C4_ALLOWED_ROOTS": os.pathsep.join(p.get("allowed_roots", []) + [cfg["loop_root"]]),
           **p.get("env", {})}
    return argv, env


def run_policy(cfg, state, ph, root) -> dict:
    h = host(cfg, "gpu")
    info = {"disk": disk_guard(cfg, h, cfg["loop_root"], 0.2)}
    if cfg["policy"]["device"] == "cuda":
        info["gpu"] = gpu_guard(cfg, h)
    out = out_dir(root / "policy", f"c{state['cycles'][-1]['cycle']:02d}", ph["attempt"])
    argv, env = policy_cmd(cfg, state, out)
    code = run_cmd(h, argv, env=env, log=out.with_name(out.name + ".log"),
                   timeout_s=3600 * float(cfg["policy"]["timeout_h"]))
    summary = read_json(out / "RUN_SUMMARY.json") if (out / "RUN_SUMMARY.json").is_file() else {}
    ckpt = out / "policy_latest.pt"
    if code != 0 or summary.get("status") != "completed" or not ckpt.is_file():
        raise RuntimeError(f"policy phase failed (exit {code}): {summary.get('error')}")
    last = summary["updates"][-1] if summary.get("updates") else {}
    state["policy"] = {"checkpoint": str(ckpt), "sha256": sha256(ckpt), "source": f"cycle {state['cycles'][-1]['cycle']}",
                       "trained_on_wm_sha256": state["accepted_wm"]["sha256"]}
    return {**info, "out": str(out), "exit": code, "policy_sha256": state["policy"]["sha256"],
            "updates": len(summary.get("updates", [])), "resumed_from": summary.get("resumed_from"),
            "independent_J_before": (summary.get("independent_score_before") or {}).get("J"),
            "independent_J_after": (summary.get("independent_score") or {}).get("J"),
            "last_actor_loss": last.get("actor_loss"), "last_critic_loss": last.get("critic_loss")}


def allocate_seeds(cfg, state, reg, root) -> list[int]:
    lo, hi = (int(x) for x in reg["seed_block"])
    excluded = [(int(a), int(b)) for a, b in reg.get("exclude", [])]
    used = set(state["seeds_used"])
    ledger = root / "seed_ledger.jsonl"
    if ledger.is_file():
        for line in ledger.read_text(encoding="utf-8").splitlines():
            used.update(json.loads(line).get("seeds", []))
    n = int(cfg["collect"]["worlds_per_cycle"])
    seeds = [s for s in range(lo, hi + 1) if s not in used
             and not any(a <= s <= b for a, b in excluded)
             and not EVAL_SEEDS[0] <= s <= EVAL_SEEDS[1]][:n]
    if len(seeds) < n:
        raise Refused("registered seed block exhausted")
    return seeds


def run_collect(cfg, state, ph, root) -> dict:
    reg = check_registration(cfg)
    c, h = cfg["collect"], host(cfg, "cpu")
    k = state["cycles"][-1]["cycle"]
    if not state.get("policy"):
        raise Refused("no policy checkpoint to collect with")
    work = f"{c['work_root']}/{cfg['loop_id']}/c{k:02d}" + (f".a{ph['attempt']}" if ph["attempt"] > 1 else "")
    need_gb = float(c.get("raw_mb_per_world", 60)) * int(c["worlds_per_cycle"]) / 1024
    info = {"disk": disk_guard(cfg, h, c["work_root"], need_gb)}
    seeds = allocate_seeds(cfg, state, reg, root)
    sim_s = collect_sim_s(cfg)
    # Record the allocation and the charge BEFORE any SUMO starts (never re-allocated).
    append(root / "seed_ledger.jsonl", {"cycle": k, "attempt": ph["attempt"], "seeds": seeds,
                                        "block": reg["seed_block"], "host": "cpu", "sim_s_charged": sim_s})
    state["seeds_used"] += seeds
    state["sim_s_charged"] += sim_s
    save_state(cfg, state)
    inputs = f"{work}/inputs"
    log = root / "logs" / f"c{k:02d}_collect.log"
    mk = [h["python"], "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).mkdir(parents=True, exist_ok=False)", inputs]
    if run_cmd(h, mk, env={}, log=log, timeout_s=120) != 0:
        raise RuntimeError("could not create the (new) collection directory")
    acc, pol = state["accepted_wm"], state["policy"]
    staged = {}
    for name, src in (("policy.pt", pol["checkpoint"]), ("wm/wm_s.pt", acc["checkpoint"])):
        sync(cfg, "gpu", src, "cpu", f"{inputs}/{name}", log)
        staged[name] = f"{inputs}/{name}"
    for sidecar in ("physics.json", "TRAIN_RECEIPT.json", "lane_calib.json"):
        src = Path(acc["checkpoint"]).with_name(sidecar)
        if src.is_file():
            sync(cfg, "gpu", str(src), "cpu", f"{inputs}/wm/{sidecar}", log)
    sc_args = sumocredit_args(cfg, state)
    if sc_args and cfg["collect"]["sumocredit"].get("gated_arm") and state.get("gate_table"):
        sync(cfg, "gpu", state["gate_table"], "cpu", f"{inputs}/GATE_TABLE.json", log)
        sc_args += ["--gate-table", f"{inputs}/GATE_TABLE.json"]
        if int(cfg["collect"]["sumocredit"].get("gate_candidates", 1)) > 1 and state.get("gate_mean_table"):
            sync(cfg, "gpu", state["gate_mean_table"], "cpu", f"{inputs}/GATE_MEAN_TABLE.json", log)
            sc_args += ["--gate-mean-table", f"{inputs}/GATE_MEAN_TABLE.json"]
    reg_copy = f"{inputs}/REGISTRATION.json"
    sync(cfg, "gpu", cfg["registration"], "cpu", reg_copy, log)
    if Path(acc["config"]).suffix == ".json":
        wm_cfg = f"{inputs}/wm/config.json"            # flat/resolved JSON: self-contained, staged
        sync(cfg, "gpu", acc["config"], "cpu", wm_cfg, log)
    else:
        # TOML configs 'extends' siblings: use the collector code root's copy of the same file
        # (the code roots are deployed from the same commit; LOOP14_DESIGN.md).
        wm_cfg = str(Path(h["code_root"]) / "configs" / Path(acc["config"]).name)
    argv = [h["python"], "-m", "traffic_shepherd.c4.collect", "--policy-pairs", "--policy", staged["policy.pt"],
            "--settings-wm-ckpt", staged["wm/wm_s.pt"], "--settings-wm-config", wm_cfg,
            "--registration", reg_copy, "--seeds", ",".join(map(str, seeds)),
            "--case-plan", "weighted-cycle", "--penetrations", c["penetrations"], "--pen-weights", c["pen_weights"],
            "--demands", c["demands"], "--population", c.get("population", "textbook"),
            "--warmup", str(c["warmup"]), "--horizon", str(c["branch_first"]),
            "--branch-first", str(c["branch_first"]), "--branch-cadence", str(c["branch_cadence"]),
            "--branch-points", str(c["branch_points"]), "--branch-horizon", str(c["branch_horizon"]),
            "--workers", str(c["workers"]), "--out", f"{work}/collect", *sc_args]
    if c.get("no_physics_log", True):
        argv.append("--no-physics-log")
    env = {"PYTHONPATH": os.pathsep.join([str(Path(h["code_root"]) / "src"), h["code_root"]]),
           "C4_ALLOWED_ROOTS": os.pathsep.join([work, *c.get("allowed_roots", [])]), **c.get("env", {})}
    code = run_cmd(h, argv, env=env, log=log, timeout_s=3600 * float(c["timeout_h"]))
    summary_path = f"{work}/collect/COLLECT_SUMMARY.json"
    local_summary = root / "collect" / f"c{k:02d}_a{max(ph['attempt'], 1)}_COLLECT_SUMMARY.json"
    try:
        sync(cfg, "cpu", summary_path, "gpu", str(local_summary), log)
        summary = read_json(local_summary)
    except (OSError, RuntimeError, ValueError):
        summary = {}
    if code != 0 or summary.get("episodes_failed", 1) != 0 or not summary.get("episodes_complete"):
        raise RuntimeError(f"collection failed (exit {code}; summary {bool(summary)})")
    deltas = list((summary.get("pair_dJ") or {}).values())
    return {**info, "work": work, "seeds": seeds, "sim_s_charged": sim_s, "exit": code,
            "episodes": summary["episodes_complete"], "branches": summary.get("branches"),
            "pair_sim_seconds": summary.get("pair_sim_seconds"),
            "branch_collisions": summary.get("branch_collisions"), "branch_teleports": summary.get("branch_teleports"),
            "sumo_pairs": len(deltas), "sumo_mean_policy_minus_native_J": (sum(deltas) / len(deltas)) if deltas else None,
            **({"sumocredit": summary.get("sumocredit"), "gate_table_used": "--gate-table" in sc_args}
               if sc_args else {})}


def run_convert(cfg, state, ph, root) -> dict:
    h, k = host(cfg, "cpu"), state["cycles"][-1]["cycle"]
    col = state["cycles"][-1]["phases"]["collect"]["receipt"]
    work = col["work"]
    tag = f"c{k:02d}" + (f".a{ph['attempt']}" if ph["attempt"] > 1 else "")
    remote_conv, remote_ret = f"{work}/pairs_{tag}", f"{work}/returns_{tag}.npz"
    log = root / "logs" / f"c{k:02d}_convert.log"
    env = {"PYTHONPATH": os.pathsep.join([str(Path(h["code_root"]) / "src"), h["code_root"]]),
           "C4_ALLOWED_ROOTS": os.pathsep.join([work, *cfg["collect"].get("allowed_roots", [])])}
    n_branch = int(col.get("branches") or 0)
    gpu_need = n_branch * float(cfg["convert"]["mb_per_episode"]) / 1024
    info = {"disk_cpu": disk_guard(cfg, h, work, gpu_need), "disk_gpu": disk_guard(cfg, host(cfg, "gpu"), cfg["loop_root"], gpu_need)}
    conv = [h["python"], "-m", "wms.data.c4_episodes", "--episodes", f"{work}/collect/branches",
            "--out", remote_conv, "--val-frac", str(cfg["convert"].get("val_frac", 0.0)),
            "--split-seed", str(cfg["convert"].get("split_seed", 0))]
    if run_cmd(h, conv, env=env, log=log, timeout_s=3 * 3600) != 0:
        raise RuntimeError("conversion failed")
    ret = [h["python"], "-m", "wms.control.loop14_returns", "--source", remote_conv, "--output", remote_ret,
           "--policy-checkpoint", state["policy"]["sha256"], "--kind", "POLICY"]
    if run_cmd(h, ret, env=env, log=log, timeout_s=3600) != 0:
        raise RuntimeError("real-return export failed")
    local_conv = root / "data" / f"pairs_{tag}"
    local_ret = root / "returns" / f"returns_{tag}.npz"
    if local_conv.exists() or local_ret.exists():
        raise Refused("converted output already exists locally (never reused)")
    sync(cfg, "cpu", remote_conv + ("/" if host(cfg, "cpu").get("ssh") else ""), "gpu", str(local_conv), log)
    sync(cfg, "cpu", remote_ret, "gpu", str(local_ret), log)
    sync(cfg, "cpu", remote_ret[:-4] + ".receipt.json", "gpu", str(local_ret.with_suffix(".receipt.json")), log)
    manifest = read_json(local_conv / "C4_EPISODES_MANIFEST.json")
    out = {**info, "pairs": str(local_conv), "returns": str(local_ret), "returns_sha256": sha256(local_ret),
           "episodes": len(manifest["episodes"]),
           "pair_groups": len({e.get("pair_key") for e in manifest["episodes"] if e.get("pair_key")}),
           "targets": read_json(local_ret.with_suffix(".receipt.json"))["targets"]}
    if cfg["collect"].get("sumocredit"):
        out["sumocredit"] = convert_sumocredit(cfg, state, work, tag, env, log, root)
    return out


def convert_sumocredit(cfg, state, work, tag, env, log, root) -> dict:
    """Gather this cycle's gate/credit records (cpu), then refit on ALL cycles so far (gpu host):
    the gate reliability table and the real-credit multipliers used by the next cycle."""
    h, g = host(cfg, "cpu"), host(cfg, "gpu")
    x = cfg["collect"]["sumocredit"]
    remote_credit, remote_gate = f"{work}/sumocredit_points_{tag}.jsonl", f"{work}/gate_records_{tag}.jsonl"
    gather = [h["python"], "-m", "wms.control.real_credit14", "gather", "--collect-root", f"{work}/collect",
              "--out-credit", remote_credit, "--out-gate", remote_gate]
    if run_cmd(h, gather, env=env, log=log, timeout_s=1800) != 0:
        raise RuntimeError("S-SUMOCREDIT gather failed")
    sc = root / "sumocredit"
    local_credit, local_gate = sc / f"points_{tag}.jsonl", sc / f"gate_{tag}.jsonl"
    if local_credit.exists() or local_gate.exists():
        raise Refused("S-SUMOCREDIT records already exist locally (never reused)")
    sync(cfg, "cpu", remote_credit, "gpu", str(local_credit), log)
    sync(cfg, "cpu", remote_gate, "gpu", str(local_gate), log)
    genv = {"PYTHONPATH": os.pathsep.join([str(Path(g["code_root"]) / "src"), g["code_root"]]),
            "C4_ALLOWED_ROOTS": os.pathsep.join([cfg["loop_root"]])}
    out = {"points_file": str(local_credit), "gate_file": str(local_gate)}
    credit_files = sorted(str(p) for p in sc.glob("points_*.jsonl") if p.stat().st_size > 0)
    if credit_files and int(x.get("credit_horizon", 0)):
        target = sc / f"REAL_CREDIT_{tag}.json"
        fit = [g["python"], "-m", "wms.control.real_credit14", "fit", "--points", *credit_files,
               "--out", str(target), "--cycle-budget-sim-s", str(collect_sim_s(cfg))]
        if run_cmd(g, fit, env=genv, log=log, timeout_s=1800) != 0:
            raise RuntimeError("S-SUMOCREDIT credit fit failed")
        doc = read_json(target)
        state["real_credit"] = str(target)
        out.update(real_credit=str(target), multipliers=doc["multipliers"], credit_points=doc["points"])
    gate_files = sorted(str(p) for p in sc.glob("gate_*.jsonl") if p.stat().st_size > 0)
    comparisons = [("reference", "gate_table", "GATE_TABLE")]
    if int(x.get("gate_candidates", 1)) > 1:
        comparisons.append(("mean", "gate_mean_table", "GATE_MEAN_TABLE"))
    for comparison, key, stem in (comparisons if gate_files and int(x.get("gate_horizon", 0)) else []):
        target = sc / f"{stem}_{tag}.json"
        fit = [g["python"], "-m", "wms.control.confident_gate", "--records", *gate_files, "--out", str(target),
               "--unit", x.get("gate_unit", "fleet"), "--reference", x.get("gate_reference", "native"),
               "--horizon", str(float(x["gate_horizon"])), "--state-source", x.get("gate_state", "belief"),
               "--comparison", comparison]
        code = run_cmd(g, fit, env=genv, log=log, timeout_s=1800)
        prefix = "gate" if comparison == "reference" else "gate_mean"
        if code == 0 and target.is_file():
            state[key] = str(target)
            meta = read_json(target)["meta"]
            out.update({key: str(target), f"{prefix}_n": meta.get("n"),
                        f"{prefix}_brier": meta.get("brier_in_sample"),
                        f"{prefix}_blocks_open": meta.get("blocks_at_or_above_threshold")})
        else:
            out[f"{key}_error"] = f"fit exit {code} (the gated arms keep the previous table, if any)"
    return out


def run_wm_ft(cfg, state, ph, root) -> dict:
    w, h = cfg["wm"], host(cfg, "gpu")
    k = state["cycles"][-1]["cycle"]
    conv = state["cycles"][-1]["phases"]["convert"]["receipt"]
    info = {"disk": disk_guard(cfg, h, cfg["loop_root"], float(w.get("need_gb", 0.3)))}
    if w["device"] == "cuda":
        info["gpu"] = gpu_guard(cfg, h)
    out = out_dir(root / "wm", f"c{k:02d}", ph["attempt"])
    sets = ["--set", f"model.init_from={json.dumps(state['accepted_wm']['checkpoint'])}",
            "--set", f"runtime.device={json.dumps(w['device'])}"]
    for item in w.get("extra_set", []):
        sets += ["--set", item]
    env = {"PYTHONPATH": os.pathsep.join([str(Path(h["code_root"]) / "src"), h["code_root"]]),
           "WMS_DATA_LOOP14_PAIRS": conv["pairs"], **w.get("env", {})}
    log = out.with_name(out.name + ".log")
    base = [h["python"], "-m", "wms", "train", "--config", w["config"], "--out", str(out), *sets]
    # Fail closed on the resolved config before training (``--mode preflight`` predates c4_base
    # validation sets and refuses every C4 recipe; validate_for_training is what run executes).
    validate = [h["python"], "-c", "import sys; from wms.config.load import load_train_config; "
                "c = load_train_config(sys.argv[1], sys.argv[2:]); c.validate_for_training(); "
                "assert c.physics.lane_change_secure_gap, 'D1 must be on'; print(c.name)",
                w["config"], *sets[1::2]]
    if run_cmd(h, validate, env=env, log=log, timeout_s=600) != 0:
        raise RuntimeError("WM fine-tune config validation failed (no training started)")
    budget = int(check_registration(cfg)["budget"]["wm_steps_per_cycle"])
    steps = validate_steps(h, env, w, sets, log)
    if steps > budget:
        raise Refused(f"WM fine-tune {steps} steps exceed the registered {budget} per cycle")
    code = run_cmd(h, [*base, "--mode", "run"], env=env, log=log, timeout_s=3600 * float(w["timeout_h"]))
    ckpt = out / "wm_s.pt"
    finite = True
    rows = 0
    if (out / "train_log.jsonl").is_file():
        for line in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows += 1
            loss = row.get("loss")
            if not isinstance(loss, (int, float)) or not (loss == loss and abs(loss) != float("inf")):
                finite = False
    if code != 0 or not ckpt.is_file() or not (out / "physics.json").is_file() or not finite or rows == 0:
        raise RuntimeError(f"WM fine-tune failed (exit {code}, checkpoint {ckpt.is_file()}, finite {finite}, rows {rows})")
    return {**info, "out": str(out), "candidate": str(ckpt), "candidate_sha256": sha256(ckpt),
            "init_from_sha256": state["accepted_wm"]["sha256"], "train_log_rows": rows, "exit": code}


def validate_steps(h, env, w, sets, log) -> int:
    """Total optimiser steps of the resolved fine-tune config (registered budget check)."""
    probe = [h["python"], "-c", "import sys; from wms.config.load import load_train_config; "
             "c = load_train_config(sys.argv[1], sys.argv[2:]); print('STEPS', sum(c.curriculum.stage_steps))",
             w["config"], *sets[1::2]]
    if h.get("ssh"):
        return int(w.get("declared_steps", 10 ** 9))
    proc = subprocess.run([str(a) for a in probe], cwd=h["code_root"], capture_output=True, text=True,
                          env={**os.environ, **{k: str(v) for k, v in env.items()}}, timeout=600, check=False)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(proc.stdout + proc.stderr)
    line = next((x for x in proc.stdout.splitlines() if x.startswith("STEPS ")), None)
    if proc.returncode != 0 or line is None:
        raise RuntimeError("could not resolve the fine-tune step count")
    return int(line.split()[1])


def run_check(cfg, state, ph, root) -> dict:
    ch, h = cfg["check"], host(cfg, "gpu")
    k = state["cycles"][-1]["cycle"]
    ft = state["cycles"][-1]["phases"]["wm_ft"]["receipt"]
    conv = state["cycles"][-1]["phases"]["convert"]["receipt"]
    out = out_dir(root / "checks", f"c{k:02d}", ph["attempt"])
    argv = [h["python"], "-B", str(Path(h["code_root"]) / "tools/loop14_checks/accept.py"),
            "--candidate", ft["candidate"], "--reference", state["accepted_wm"]["checkpoint"],
            "--config", cfg["wm"]["config"], "--roots", *ch["roots"], "--pairs", conv["pairs"],
            "--panel", str(root / "panel.json"), "--out", str(out),
            "--onestep-windows", str(ch["onestep_windows"]), "--rollout-windows", str(ch["rollout_windows"]),
            "--pair-groups", str(ch["pair_groups"]), "--stride-eps", str(ch.get("stride_eps", 10)),
            "--workers", str(ch.get("workers", 1))]
    env = {"PYTHONPATH": os.pathsep.join([str(Path(h["code_root"]) / "src"), h["code_root"]]),
           "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": 1, "MKL_NUM_THREADS": 1, **ch.get("env", {})}
    if ch.get("nice", True) and os.name == "posix" and not h.get("ssh"):
        argv = ["nice", "-n", "19", *argv]
    code = run_cmd(h, argv, env=env, log=out.with_name(out.name + ".log"), timeout_s=3600 * float(ch["timeout_h"]))
    if not (out / "ACCEPT.json").is_file():
        raise RuntimeError(f"acceptance produced no verdict (exit {code})")
    verdict = read_json(out / "ACCEPT.json")
    accepted = code == 0 and verdict.get("decision") == "ACCEPT"
    previous = dict(state["accepted_wm"])
    if accepted:
        state["accepted_wm"] = {"checkpoint": ft["candidate"], "config": cfg["wm"]["config"],
                                "sha256": ft["candidate_sha256"], "source": f"cycle {k}"}
        state["consecutive_rejects"] = 0
    else:
        state["consecutive_rejects"] += 1
    roll = verdict.get("rollout", {})
    return {"out": str(out), "exit": code, "decision": "ACCEPT" if accepted else "REJECT",
            "reasons": verdict.get("reasons"), "error": verdict.get("error"),
            "previous_wm_sha256": previous["sha256"], "accepted_wm_sha256": state["accepted_wm"]["sha256"],
            "onestep_mae": (verdict.get("one_step") or {}).get("mae"),
            "hard_brake_share": {a: (roll.get(a) or {}).get("hard_brake_share") for a in ("candidate", "reference")},
            "paired_difference": {a: {kk: v for kk, v in d.items() if kk != "rows"}
                                  for a, d in (verdict.get("paired_difference") or {}).items()}}


RUNNERS = {"policy": run_policy, "collect": run_collect, "convert": run_convert,
           "wm_ft": run_wm_ft, "check": run_check}


def run_phase(cfg, name, *, dry_run=False) -> int:
    state = load_state(cfg)
    root = Path(cfg["loop_root"])
    if state.get("stopped"):
        raise Refused(f"loop stopped: {state['stopped']}")
    nxt = _phase(state)
    if nxt["name"] is None:
        raise Refused("cycle complete; nothing to run")
    if name != nxt["name"]:
        raise Refused(f"phase order: the next phase of cycle {state['cycles'][-1]['cycle']} is {nxt['name']}, not {name}")
    ph = state["cycles"][-1]["phases"][name]
    if ph["status"] == "failed":
        raise Refused(f"{name} failed; inspect, then `retry {name}` (new attempt directory)")
    if ph["status"] == "running":
        raise Refused(f"{name} is marked running (pid {ph.get('pid')}); if it died, `retry {name}`")
    if dry_run:
        print(json.dumps({"cycle": state["cycles"][-1]["cycle"], "next": name, "accepted_wm": state["accepted_wm"],
                          "policy": state["policy"], "sim_s_this_collect": collect_sim_s(cfg)}, indent=1))
        if name == "policy":
            argv, env = policy_cmd(cfg, state, Path(cfg["loop_root"]) / "policy" / "<cycle>")
            print("env", json.dumps({k: str(v) for k, v in env.items()}))
            print(" ".join(map(shlex.quote, map(str, argv))))
        return 0
    ph.update(status="running", attempt=ph["attempt"] + 1, started=now(), pid=os.getpid())
    save_state(cfg, state)
    k = state["cycles"][-1]["cycle"]
    try:
        receipt = RUNNERS[name](cfg, state, ph, root)
    except GpuBusy as error:
        ph.update(status="pending", attempt=ph["attempt"] - 1, refused=str(error))
        save_state(cfg, state)
        append(root / "cycle_ledger.jsonl", {"cycle": k, "phase": name, "event": "gpu_busy", "detail": str(error)})
        print(f"GPU BUSY: {error}", file=sys.stderr)
        return 5
    except Refused as error:
        ph.update(status="pending", attempt=ph["attempt"] - 1, refused=str(error))
        save_state(cfg, state)
        append(root / "cycle_ledger.jsonl", {"cycle": k, "phase": name, "event": "refused", "detail": str(error)})
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # noqa: BLE001 - recorded, never silently retried
        ph.update(status="failed", finished=now(), error=f"{type(error).__name__}: {error}")
        save_state(cfg, state)
        append(root / "cycle_ledger.jsonl", {"cycle": k, "phase": name, "event": "failed", "error": ph["error"]})
        print(f"FAILED: {ph['error']}", file=sys.stderr)
        return 3
    ph.update(status="done", finished=now(), receipt=receipt)
    append(root / "cycle_ledger.jsonl", {"cycle": k, "phase": name, "event": "done", "attempt": ph["attempt"],
                                         "receipt": receipt})
    code = 0
    if name == "check":
        code = 0 if receipt["decision"] == "ACCEPT" else 4
        reg = check_registration(cfg)
        decision = {"cycle": k, "event": "cycle_decision", "wm_update": receipt["decision"],
                    "accepted_wm": state["accepted_wm"], "policy": state["policy"],
                    "real_returns_next": state["cycles"][-1]["phases"]["convert"]["receipt"]["returns"],
                    "sim_s_charged_total": state["sim_s_charged"]}
        # The policy continues on the ACCEPTED WM with this cycle's current-policy SUMO returns.
        state["real_returns"] = decision["real_returns_next"]
        stops = []
        if state["consecutive_rejects"] >= int(reg["stop"].get("consecutive_wm_rejects", 2)):
            stops.append(f"{state['consecutive_rejects']} consecutive WM rejections")
        if k + 1 >= int(reg["budget"]["max_cycles"]):
            stops.append("registered max_cycles reached")
        if state["sim_s_charged"] + collect_sim_s(cfg) > reg["budget"]["sim_s_total"]:
            stops.append("registered SUMO budget cannot fund another collection")
        if stops:
            state["stopped"] = "; ".join(stops)
        else:
            state["cycles"].append(new_cycle(k + 1))
        decision["stopped"] = state["stopped"]
        append(root / "cycle_ledger.jsonl", decision)
    if name == "collect":
        bad = (receipt.get("branch_collisions") or 0) + (receipt.get("branch_teleports") or 0)
        limit = int(check_registration(cfg)["stop"].get("max_branch_incidents_per_cycle", 0))
        if bad > limit:
            state["stopped"] = f"SUMO safety: {bad} branch collisions/teleports > registered {limit}"
            append(root / "cycle_ledger.jsonl", {"cycle": k, "event": "stop", "reason": state["stopped"]})
    save_state(cfg, state)
    print(json.dumps({"cycle": k, "phase": name, "status": "done", **({"decision": receipt["decision"]}
                                                                         if name == "check" else {})}))
    return code


def retry(cfg, name) -> int:
    state = load_state(cfg)
    ph = state["cycles"][-1]["phases"][name]
    if ph["status"] not in ("failed", "running"):
        raise Refused(f"{name} is {ph['status']}; only a failed/dead phase can be retried")
    ph.update(status="pending", previous_error=ph.get("error"))
    save_state(cfg, state)
    append(Path(cfg["loop_root"]) / "cycle_ledger.jsonl",
           {"cycle": state["cycles"][-1]["cycle"], "phase": name, "event": "retry_armed", "next_attempt": ph["attempt"] + 1})
    return run_phase(cfg, name)


def status(cfg) -> int:
    state = load_state(cfg)
    cyc = state["cycles"][-1]
    print(json.dumps({"loop_id": state["loop_id"], "cycle": cyc["cycle"], "next": _phase(state)["name"],
                      "phases": {p: v["status"] for p, v in cyc["phases"].items()},
                      "accepted_wm": state["accepted_wm"], "policy": state["policy"],
                      "sim_s_charged": state["sim_s_charged"], "seeds_used": len(state["seeds_used"]),
                      "consecutive_rejects": state["consecutive_rejects"], "stopped": state["stopped"]}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("init", "status", "plan", "next", "run", "retry"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "init":
            p.add_argument("--adopt-policy", default=None, help="existing POLICY14 checkpoint for cycle 0")
        if name in ("run", "retry"):
            p.add_argument("phase", choices=PHASES)
    args = ap.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.cmd == "init":
            return init(cfg, args)
        if args.cmd == "status":
            return status(cfg)
        if args.cmd == "plan":
            return run_phase(cfg, _phase(load_state(cfg))["name"], dry_run=True)
        if args.cmd == "next":
            return run_phase(cfg, _phase(load_state(cfg))["name"])
        if args.cmd == "run":
            return run_phase(cfg, args.phase)
        return retry(cfg, args.phase)
    except GpuBusy as error:
        print(f"GPU BUSY: {error}", file=sys.stderr)
        return 5
    except Refused as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""N7 launcher: speed-only MAPPO on wz3to2 (plan -> check -> ack -> smoke -> run -> status/stop/bind).

    python tools/nm_mappo/launch.py plan  --option full169_u0 --host cpuQ --run-dir RUN --init U0.pt \
                                          [--python PY] [--sumo-home DIR] [--workers N] [--allow-stub]
    python tools/nm_mappo/launch.py check --run-dir RUN          # CPU fake-env smoke of the exact config
    python tools/nm_mappo/launch.py ack   --run-dir RUN --registration REGISTRATION.md
    python tools/nm_mappo/launch.py smoke --run-dir RUN [--dry-run]   # SUMO: 1 it x 2 ep x (300+30) s
    python tools/nm_mappo/launch.py run   --run-dir RUN [--dry-run]   # detached production (Linux)
    python tools/nm_mappo/launch.py resume --run-dir RUN [--dry-run]  # same plan, after a box restart
    python tools/nm_mappo/launch.py status --run-dir RUN
    python tools/nm_mappo/launch.py stop  --run-dir RUN
    python tools/nm_mappo/launch.py bind  --run-dir RUN --iteration 169   # comparator binding receipt

Gates (mechanical, never bypassed by this tool):
* ``plan`` writes ``PLAN.json`` (exact argv, config/init sha256, map binding, budget, wall cap).
  A STUB map binding (N1 not landed) gives ``sumo_allowed = false``: smoke/run/resume refuse.
* ``ack`` writes ``REGISTERED.json`` only if the registration file already contains the PLAN
  sha256 (registered BEFORE any SUMO, AGENTS.md); smoke/run refuse without a matching receipt.
* ``run`` additionally requires a PASSED smoke (completed, 2 episodes, 0 failures / collisions /
  teleports / lane commands / policy lane changes).
* SUMO stages start ``tools/nm_mappo/supervise.py`` detached (setsid nohup nice -n 19); the
  supervisor re-verifies everything, enforces the wall cap and records its own PIDs only.
No ssh: run this ON the training box from the staged code root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
for _p in (CODE / "src", CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_sibling(name: str):
    """Import tools/nm_mappo/<name>.py under a unique module name (no sys.path pollution)."""
    import importlib.util
    key = f"nm_mappo_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


Q = load_sibling("quote")

CONFIG = CODE / "configs" / "c4_mappo_speed_wz3to2.toml"
U0_SHA256 = "d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca"
SNAPSHOTS = (24, 42, 84)          # pre-registered nested reduced-cost snapshot iterations
SMOKE = {"workers": 2, "iterations": 1, "episodes": 2, "horizon_s": 30, "max_sumo_seconds": 60}
STAGES = ("smoke", "production")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text_lf(path) -> str:
    """sha256 of a text file with CRLF normalised (Windows checkout == Linux bundle)."""
    return sha256_bytes(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def write_json_new(path: Path, doc) -> None:
    with open(path, "x", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
        fh.flush()
        os.fsync(fh.fileno())


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=CODE,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_binding(config: Path):
    from wms.control.mappo.mapbind import resolve_binding as _resolve
    from wms.core.toml import loads
    cfg = loads(config.read_text(encoding="utf-8"))
    return _resolve(cfg["map"]), cfg


def train_argv(python: str, config: Path, out: Path, *, init: str | None, workers: int,
               iterations: int, episodes: int, max_sumo_seconds: float,
               horizon_s: float | None = None, update_threads: int = 16) -> list[str]:
    argv = [python, "-u", "-m", "wms.control.mappo.train", "--config", str(config),
            "--out", str(out)]
    if init:
        argv += ["--init-from", str(init)]
    argv += ["--update-threads", str(update_threads), "--workers", str(workers),
             "--iterations", str(iterations), "--episodes-per-iteration", str(episodes)]
    if horizon_s is not None:
        argv += ["--horizon-s", str(int(horizon_s))]
    argv += ["--max-sumo-seconds", str(int(max_sumo_seconds))]
    return argv


def make_plan(*, option: str, host: str, run_dir: Path, init: str | None, python: str,
              sumo_home: str | None, workers: int | None = None, config: Path = CONFIG,
              allow_stub: bool = False) -> dict:
    if option not in Q.OPTIONS:
        raise SystemExit(f"unknown option {option!r}; choose from {sorted(Q.OPTIONS)}")
    if host not in Q.HOSTS:
        raise SystemExit(f"unknown host {host!r}; choose from {sorted(Q.HOSTS)}")
    q = Q.option_quote(option)
    binding, cfg = resolve_binding(config)
    if binding.is_stub and not allow_stub:
        raise SystemExit("map binding is the STUB descriptor (N1 not landed / [map] api unset); "
                         "use --allow-stub only for a dry plan (SUMO stages will refuse)")
    rollout = cfg["rollout"]
    if q["init"] == "u0":
        if not init:
            raise SystemExit("--init <MAPPO u0 policy_00169.pt> is required for a u0 warm start")
        init_record = {"kind": "u0", "path": str(init), "sha256": U0_SHA256}
    else:
        init, init_record = None, {"kind": "scratch", "path": None, "sha256": None}
    workers = int(workers or Q.HOSTS[host]["workers"])
    wall_h = (q["wall_h"][host] if workers == Q.HOSTS[host]["workers"]
              else Q.wall_h_for_workers(q["iterations"], workers))
    high_h = wall_h[1]
    wall_cap_s = int(max(2.0, 1.5 * high_h) * 3600)
    production = train_argv(python, config, run_dir / "production", init=init, workers=workers,
                            iterations=q["iterations"], episodes=Q.EPISODES,
                            max_sumo_seconds=q["controlled_s"],
                            update_threads=int(rollout.get("update_threads", 16)))
    smoke = train_argv(python, config, run_dir / "smoke", init=init, workers=SMOKE["workers"],
                       iterations=SMOKE["iterations"], episodes=SMOKE["episodes"],
                       max_sumo_seconds=SMOKE["max_sumo_seconds"], horizon_s=SMOKE["horizon_s"],
                       update_threads=int(rollout.get("update_threads", 16)))
    return {
        "schema": "nm_mappo.plan.v1", "task": "N7", "option": option, "host": host,
        "run_dir": str(run_dir), "code_root": str(CODE), "source_commit": git_head(),
        "config": {"path": str(config), "sha256_lf": sha256_text_lf(config)},
        "init": init_record, "binding": binding.record(), "sumo_allowed": not binding.is_stub,
        "seeds": rollout["seeds"], "workers": workers,
        "iterations": q["iterations"], "episodes_per_iteration": Q.EPISODES,
        "max_sumo_seconds_controlled": q["controlled_s"],
        "budget_sim_s": {"planned": q["planned_sim_s"], "smoke": q["smoke_sim_s"],
                         "planned_total": q["planned_total_sim_s"], "hard_cap": q["hard_cap_sim_s"]},
        "quote_wall_h": wall_h, "wall_cap_s": wall_cap_s,
        "snapshots_preregistered": [s for s in SNAPSHOTS if s <= q["iterations"]],
        "python": python, "sumo_home": sumo_home,
        "argv": {"smoke": smoke, "production": production},
        "env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
        "rules": ["no HELD-OUT 989600-989699 seed", "never 964000-964199/962xxx/977xxx",
                  "own PIDs only", "no checkpoint selection by evaluation outcomes"],
    }


def load_plan(run_dir: Path) -> tuple[dict, str]:
    path = run_dir / "PLAN.json"
    return json.loads(path.read_text()), sha256_file(path)


def verify_plan(run_dir: Path, *, check_init: bool = True) -> dict:
    """PLAN + REGISTERED receipt + config/init hashes; raises SystemExit on any mismatch."""
    plan, digest = load_plan(run_dir)
    if not plan.get("sumo_allowed"):
        raise SystemExit("PLAN.sumo_allowed is false (stub map binding): SUMO stages refused")
    reg_path = run_dir / "REGISTERED.json"
    if not reg_path.exists():
        raise SystemExit("REGISTERED.json missing: register the PLAN sha256 first (launch.py ack)")
    reg = json.loads(reg_path.read_text())
    if reg.get("plan_sha256") != digest:
        raise SystemExit("REGISTERED.json does not bind this PLAN.json")
    if sha256_text_lf(plan["config"]["path"]) != plan["config"]["sha256_lf"]:
        raise SystemExit("config changed since the plan")
    if check_init and plan["init"]["kind"] == "u0":
        if sha256_file(plan["init"]["path"]) != plan["init"]["sha256"]:
            raise SystemExit("init checkpoint sha256 != MAPPO u0 d9042cc5...")
    return {"plan": plan, "plan_sha256": digest, "registered": reg}


def smoke_passed(run_dir: Path) -> tuple[bool, str]:
    smoke = run_dir / "smoke"
    try:
        summary = json.loads((smoke / "RUN_SUMMARY.json").read_text())
        rows = [json.loads(s) for s in (smoke / "log.jsonl").read_text().splitlines() if s.strip()]
    except (OSError, ValueError) as exc:
        return False, f"smoke evidence unreadable: {exc}"
    if summary.get("status") != "completed":
        return False, f"smoke status {summary.get('status')!r}"
    its = [r for r in rows if r.get("type") == "iter"]
    if len(its) != SMOKE["iterations"] or its[0].get("episodes") != SMOKE["episodes"]:
        return False, "smoke must have exactly 1 iteration of 2 episodes"
    it = its[0]
    for key in ("failed_episodes", "collisions", "teleports", "lane_commands_issued",
                "policy_lane_change_rate"):
        if (it.get(key) or 0) != 0:
            return False, f"smoke {key} = {it.get(key)}"
    if any(r.get("type") == "failure" for r in rows):
        return False, "smoke has failure rows"
    eps = [r for r in rows if r.get("type") == "episode"]
    if any((r.get("dropped_vehicle_steps") or 0) != 0 for r in eps):
        return False, "smoke dropped vehicle steps (K-slot overflow on this map)"
    return True, "PASS"


def cmd_plan(a) -> int:
    run_dir = Path(a.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = make_plan(option=a.option, host=a.host, run_dir=run_dir, init=a.init,
                     python=a.python, sumo_home=a.sumo_home, workers=a.workers,
                     allow_stub=a.allow_stub)
    write_json_new(run_dir / "PLAN.json", plan)
    print(json.dumps({"plan": str(run_dir / "PLAN.json"),
                      "plan_sha256": sha256_file(run_dir / "PLAN.json"),
                      "sumo_allowed": plan["sumo_allowed"], "budget_sim_s": plan["budget_sim_s"],
                      "quote_wall_h": plan["quote_wall_h"], "sim_s": 0}))
    return 0


def cmd_check(a) -> int:
    """CPU fake-env smoke of the exact config (``--smoke``) incl. the strict u0 warm-start load
    (lane head absent, stored bounds, observation layout unchanged on this tree); zero SUMO."""
    run_dir = Path(a.run_dir)
    plan, _ = load_plan(run_dir)
    out = run_dir / f"check_{int(time.time())}"
    argv = [sys.executable, "-m", "wms.control.mappo.train", "--config", plan["config"]["path"],
            "--smoke", "--workers", "1", "--out", str(out)]
    if plan["init"]["kind"] == "u0":
        if sha256_file(plan["init"]["path"]) != plan["init"]["sha256"]:
            raise SystemExit("init checkpoint sha256 != MAPPO u0 d9042cc5...")
        argv += ["--init-from", plan["init"]["path"]]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(CODE / "src"), str(CODE)]))
    rc = subprocess.call(argv, cwd=CODE, env=env)
    print(json.dumps({"check": str(out), "rc": rc, "sim_s": 0}))
    return rc


def cmd_ack(a) -> int:
    run_dir = Path(a.run_dir)
    _, digest = load_plan(run_dir)
    text = Path(a.registration).read_text(encoding="utf-8", errors="replace")
    if digest not in text:
        raise SystemExit(f"registration file does not contain the PLAN sha256 {digest}")
    write_json_new(run_dir / "REGISTERED.json", {
        "plan_sha256": digest, "registration": str(a.registration),
        "registration_sha256": sha256_file(a.registration),
        "acked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print(json.dumps({"registered": True, "plan_sha256": digest}))
    return 0


def resume_argv(argv: list[str]) -> list[str]:
    """Production argv for an explicit resume: no --init-from (the trainer refuses it once
    checkpoints exist), plus --resume (requires a valid checkpoint)."""
    out, skip = [], False
    for item in argv:
        if skip:
            skip = False
            continue
        if item == "--init-from":
            skip = True
            continue
        out.append(item)
    return out + ["--resume"]


def supervisor_argv(plan: dict, stage: str, run_dir: Path) -> list[str]:
    return ["setsid", "nohup", "nice", "-n", "19", plan["python"], "-u",
            str(CODE / "tools" / "nm_mappo" / "supervise.py"), stage, str(run_dir)]


def cmd_stage(a, stage: str) -> int:
    run_dir = Path(a.run_dir)
    checked = verify_plan(run_dir, check_init=not a.dry_run)
    target = "production" if stage in ("run", "resume") else "smoke"
    if target == "production":
        ok, why = smoke_passed(run_dir)
        if not ok:
            raise SystemExit(f"smoke gate: {why}")
    if stage == "resume" and not (run_dir / "production" / "checkpoints").is_dir():
        raise SystemExit("resume: no production checkpoints")
    if stage != "resume" and (run_dir / target).exists():
        raise SystemExit(f"{run_dir / target} exists: fresh stage only (use resume)")
    argv = supervisor_argv(checked["plan"], "resume" if stage == "resume" else target, run_dir)
    if a.dry_run:
        trainer = checked["plan"]["argv"][target]
        if stage == "resume":
            trainer = resume_argv(trainer)
        print(json.dumps({"dry_run": True, "supervisor": argv, "trainer": trainer, "sim_s": 0}))
        return 0
    if platform.system() != "Linux" or shutil.which("setsid") is None:
        raise SystemExit("SUMO stages run on the Linux training box only")
    log = open(run_dir / f"supervisor_{target}_{int(time.time())}.log", "x")
    proc = subprocess.Popen(argv, cwd=CODE, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({"started": target, "supervisor_pid": proc.pid, "log": log.name}))
    return 0


def last_rows(path: Path, kind: str = "iter", n: int = 1) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    return [r for r in rows if r.get("type") == kind][-n:]


def cmd_status(a) -> int:
    run_dir = Path(a.run_dir)
    plan, digest = load_plan(run_dir)
    out = {"plan_sha256": digest, "option": plan["option"], "host": plan["host"],
           "registered": (run_dir / "REGISTERED.json").exists(),
           "smoke": smoke_passed(run_dir) if (run_dir / "smoke").exists() else None}
    prod = run_dir / "production"
    if (prod / "status.json").exists():
        out["status.json"] = json.loads((prod / "status.json").read_text())
    for pattern in ("OWNERSHIP_*.json", "EXIT_*.json"):
        found = sorted(prod.glob(pattern)) if prod.is_dir() else []
        if found:
            out[found[-1].name] = json.loads(found[-1].read_text())
    its = last_rows(prod / "log.jsonl", n=5)
    if its:
        walls = [r.get("iteration_wall_s") or r.get("wall_s") for r in its]
        walls = [w for w in walls if w]
        done = its[-1].get("iteration", 0)
        out["last_iter"] = {k: its[-1].get(k) for k in ("iteration", "episodes", "mean_episode_return",
                                                         "collisions", "teleports", "failed_episodes",
                                                         "lane_commands_issued")}
        if walls:
            out["eta_h"] = round((plan["iterations"] - done) * sum(walls) / len(walls) / 3600.0, 2)
    print(json.dumps(out, indent=1, default=str))
    return 0


def cmd_stop(a) -> int:
    run_dir = Path(a.run_dir)
    (run_dir / "STOP").write_text("launch.py stop\n")
    for stage in STAGES:
        if (run_dir / stage).is_dir():
            with open(run_dir / stage / "STOP_REQUESTED", "a") as fh:
                fh.write("launch.py stop\n")
    print(json.dumps({"stop_requested": True}))
    return 0


def cmd_bind(a) -> int:
    """Comparator binding receipt for one exported iteration (N4 / final evaluation)."""
    import torch
    run_dir = Path(a.run_dir)
    plan, digest = load_plan(run_dir)
    path = run_dir / "production" / f"policy_{int(a.iteration):05d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("lane_control") != "native" or any(k.startswith("lane_head.") for k in payload["policy"]):
        raise SystemExit(f"{path}: not a speed-only (native-lane) actor")
    receipt = {"path": str(path), "sha256": sha256_file(path), "iteration": int(a.iteration),
               "lane_control": "native", "plan_sha256": digest, "option": plan["option"],
               "preregistered_snapshot": int(a.iteration) in plan["snapshots_preregistered"]
               or int(a.iteration) == plan["iterations"],
               "bound_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json_new(run_dir / f"BOUND_{int(a.iteration):05d}.json", receipt)
    print(json.dumps(receipt))
    return 0


def parse(argv=None):
    ap = argparse.ArgumentParser(description="N7 wz3to2 speed-only MAPPO launcher")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--option", default="full169_u0")
    p.add_argument("--host", default="cpuQ")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--init", default=None)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--sumo-home", default=os.environ.get("SUMO_HOME"))
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--allow-stub", action="store_true")
    for name in ("check", "status", "stop"):
        sub.add_parser(name).add_argument("--run-dir", required=True)
    p = sub.add_parser("ack")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--registration", required=True)
    for name in ("smoke", "run", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--run-dir", required=True)
        p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("bind")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--iteration", type=int, required=True)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    if a.cmd == "plan":
        return cmd_plan(a)
    if a.cmd == "check":
        return cmd_check(a)
    if a.cmd == "ack":
        return cmd_ack(a)
    if a.cmd in ("smoke", "run", "resume"):
        return cmd_stage(a, a.cmd)
    if a.cmd == "status":
        return cmd_status(a)
    if a.cmd == "stop":
        return cmd_stop(a)
    return cmd_bind(a)


if __name__ == "__main__":
    raise SystemExit(main())

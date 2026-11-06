"""WM-E runner: prepare, smoke, train (resumable), status and checkpoint manifest of the wz3to2 WM fine-tune.

    python -m tools.wzwm.run prepare --root R --init WM2.pt --init-sha b8797242... \
        --map-spec MAP_SPEC.json --source wz_native=/data/src_native --source wz_mappo_u0=/data/src_mappo
    python -m tools.wzwm.run probe  --root R      # one real batch per split through the truth guard (CPU)
    python -m tools.wzwm.run train  --root R      # smoke -> phase A -> phase B; re-run = resume
    python -m tools.wzwm.run status --root R

Layout of ``R``: PLAN.json (write-once), INIT_WZ.pt + REINIT_RECEIPT.json, WORLDS.json,
smoke/aN, phases/{A,B}/aN (each a fresh ``wms train`` out dir), phases/P/DONE.json,
ESTIMATE.json, STATUS.json, EVENTS.jsonl, CHECKPOINT_MANIFEST.json, RUN.lock.
Training is the unchanged ``python -m wms train`` (same code as WM1-WM3); this module only
sequences it.  Child processes are the runner's own; it never signals anything else.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from . import guard, plan, reinit, worlds

WM2_SHA256 = "b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436"
WM2_GPUA = "/root/autodl-tmp/c4/restart20261026/arm/loop/wm/c02/wm_s.pt"
MANIFEST_NAME = "C4_EPISODES_MANIFEST.json"
REPO = Path(__file__).resolve().parents[2]
_CHILD: Optional[subprocess.Popen] = None


# ------------------------------------------------------------------ small io ---
def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read(p) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write(p, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(p)


def event(root: Path, kind: str, **kw) -> None:
    with (Path(root) / "EVENTS.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": now(), "event": kind, **kw}, default=str) + "\n")


def status(root: Path, **kw) -> None:
    p = Path(root) / "STATUS.json"
    old = read(p) if p.is_file() else {}
    write(p, {**old, **kw, "updated": now()})


def _alive(pid: int) -> bool:
    if os.name == "nt":          # os.kill(pid, 0) would TERMINATE on Windows: stay conservative
        return True
    return Path(f"/proc/{int(pid)}").exists()


class RunLock:
    """Exclusive runner lock (O_EXCL pid file).  A stale lock (dead pid) is renamed, never deleted."""

    def __init__(self, root: Path):
        self.path = Path(root) / "RUN.lock"

    def __enter__(self):
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(json.loads(self.path.read_text())["pid"])
                except Exception:
                    pid = -1
                if pid > 0 and _alive(pid):
                    raise RuntimeError(f"{self.path}: runner pid {pid} is alive")
                self.path.rename(self.path.with_name(f"RUN.lock.stale.{int(time.time())}"))
                continue
            with os.fdopen(fd, "w") as fh:
                fh.write(json.dumps({"pid": os.getpid(), "t": now()}))
            return self
        raise RuntimeError("could not take the runner lock")

    def __exit__(self, *exc):
        try:
            if int(json.loads(self.path.read_text())["pid"]) == os.getpid():
                self.path.unlink()
        except Exception:
            pass


# ------------------------------------------------------------------ sources ---
def read_manifest(src: str) -> dict:
    guard.check_path(src)
    p = Path(src) / MANIFEST_NAME
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing (convert with python -m wms.data.c4_episodes)")
    doc = read(p)
    if doc.get("schema") != "wms_c4_episode_source/1":
        raise ValueError(f"{p}: schema {doc.get('schema')!r}")
    return doc


def split_by_arm(src: str, out_root: Path) -> list[tuple[str, str]]:
    """Derive one source per arm from a single converted source (absolute episode dirs, no copy)."""
    doc = read_manifest(src)
    arms: dict[str, list] = {}
    for e in doc["episodes"]:
        arms.setdefault(str(e.get("arm") or "policy"), []).append(e)
    out = []
    for arm, eps in sorted(arms.items()):
        d = Path(out_root) / f"src_{arm}"
        d.mkdir(parents=True, exist_ok=False)
        entries = [{**e, "dir": str((Path(src) / e["dir"]).resolve())} for e in eps]
        write(d / MANIFEST_NAME, {**doc, "episodes": entries, "n_episodes": len(entries),
                                  "n_train": sum(e.get("split_role") == "train" for e in entries),
                                  "n_val": sum(e.get("split_role") == "val" for e in entries),
                                  "derived_from": str(src), "derived_arm": arm})
        out.append((f"wz_{arm}", str(d)))
    return out


# ------------------------------------------------------------------ config ---
def load_cfg(config: str, overrides):
    from wms.config.load import load_train_config
    path = Path(config) if Path(config).is_absolute() else REPO / config
    cfg = load_train_config(str(path), list(overrides))
    cfg.validate_for_training()
    return cfg


def cfg_mapping(cfg) -> dict:
    from wms.config.schema import to_mapping
    return to_mapping(cfg)


def road_record(map_spec: Optional[str]) -> dict:
    """The WM-A road for ``map_spec`` (``wms.core.road``), or {'available': False}."""
    try:
        from wms.core import road as road_mod
    except Exception as error:
        return {"available": False, "error": repr(error)}
    try:
        road = road_mod.load_road(map_spec) if map_spec else road_mod.current_road()
    except Exception as error:
        return {"available": True, "error": repr(error)}
    rec = road.record() if hasattr(road, "record") else {}
    return {"available": True, "name": getattr(road, "name", None),
            "road_end_m": getattr(road, "road_end_m", None),
            "digest": getattr(road, "digest", None), "record": rec}


def sha(p) -> str:
    return reinit.sha256_file(p)


def git_head() -> Optional[str]:
    """Deployed CODE_SHA (gpuA code dirs have no .git), else ``git rev-parse HEAD``."""
    if (REPO / "CODE_SHA").is_file():
        return (REPO / "CODE_SHA").read_text().strip() or None
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except Exception:
        return None


# ------------------------------------------------------------------ prepare ---
def prepare(args) -> dict:
    root = Path(guard.check_path(args.root))
    if (root / "PLAN.json").exists():
        raise FileExistsError(f"{root}/PLAN.json exists (write-once; use a new root)")
    root.mkdir(parents=True, exist_ok=True)
    specs = [plan.parse_source(s) for s in args.source]
    if args.split_by_arm:
        if len(specs) != 1:
            raise ValueError("--split-by-arm takes exactly one --source")
        specs = [(n, p, None) for n, p in split_by_arm(specs[0][1], root / "sources")]
    docs = {name: read_manifest(path) for name, path, _ in specs}
    wm = worlds.worlds_manifest(docs, map_name=args.map_name)
    hit = worlds.overlap(wm)
    if hit:
        raise worlds.WorldError(f"reserved qualification worlds in the data: {hit}")
    write(root / "WORLDS.json", wm)
    train_eps = {n: wm["sources"][n]["train_episodes"] for n in docs}
    sources, base = plan.mix_sources(specs, train_eps, base=args.base)
    for s in sources:
        s.path = str(Path(s.path).resolve()) if not Path(s.path).is_absolute() else s.path
        s.paired = bool(any(e.get("pair_key") for e in docs[s.name]["episodes"])) and args.paired
    road = road_record(args.map_spec)
    if not road.get("available") or road.get("error") or road.get("name") != args.map_name:
        if not args.allow_no_road:
            raise RuntimeError(f"WM road binding for {args.map_name} unavailable: {road}")
    # Initial checkpoint: verify WM2, re-initialise the geometry tensors from a fresh model.
    overrides_a = plan.phase_overrides("A", init_from="INIT_WZ.pt", sources=sources, base=base,
                                       device=args.device, profile=args.profile)
    env_road = {"WMS_MAP_SPEC": args.map_spec} if args.map_spec else {}
    os.environ.update(env_road)
    rec = reinit.reinit_checkpoint(
        Path(args.init), root / "INIT_WZ.pt", expected_sha256=args.init_sha,
        fresh=lambda: reinit.fresh_state_from_config(plan.CONFIGS["A"], overrides_a, repo=REPO),
        receipt_path=root / "REINIT_RECEIPT.json", config=plan.CONFIGS["A"])
    phases = {}
    for ph in plan.PHASES:
        init = str(root / "INIT_WZ.pt") if ph == "A" else "<A_FINAL>"
        sets = plan.phase_overrides(ph, init_from=init, sources=sources, base=base,
                                    device=args.device, profile=args.profile)
        cfg = load_cfg(plan.CONFIGS[ph], [s if not s.startswith("model.init_from=") else
                                          'model.init_from="INIT.pt"' for s in sets])
        mapping = cfg_mapping(cfg)
        guard.check_config(mapping)
        if (road.get("name") == args.map_name and road.get("road_end_m") is not None
                and abs(cfg.loss.agg_delay.road_end_m - road["road_end_m"]) > 1e-6):
            raise RuntimeError(f"phase {ph}: loss.agg_delay.road_end_m {cfg.loss.agg_delay.road_end_m} "
                               f"!= road {road['road_end_m']}")
        phases[ph] = {"config": plan.CONFIGS[ph], "config_sha256": sha(REPO / plan.CONFIGS[ph]),
                      "overrides": sets, "fingerprint": cfg.fingerprint(),
                      "curriculum": {k: list(getattr(cfg.curriculum, k)) for k in plan.STAGE_LISTS},
                      "optim": {k: list(getattr(cfg.optim, k)) for k in plan.OPTIM_LISTS
                                if getattr(cfg.optim, k) is not None},
                      "steps": plan.total_steps({"stage_steps": cfg.curriculum.stage_steps})}
    body = {
        "schema": "wzwm.plan/1", "created": now(), "root": str(root), "map": args.map_name,
        "map_spec": args.map_spec, "road": road, "profile": args.profile, "device": args.device,
        "init": {"wm2": str(args.init), "wm2_sha256": args.init_sha, "init_wz": str(root / "INIT_WZ.pt"),
                 "init_wz_sha256": rec["out_sha256"], "reinit_prefixes": rec["prefixes"],
                 "n_reinit": rec["n_reinit"], "n_kept": rec["n_kept"],
                 "reinit_receipt_sha256": sha(root / "REINIT_RECEIPT.json")},
        "sources": [vars(s) for s in sources], "base_source": base,
        "worlds_sha256": wm["sha256"], "n_train_worlds": wm["n_train_worlds"],
        "n_val_worlds": wm["n_val_worlds"], "phases": phases,
        "smoke": {"enabled": not args.no_smoke, "steps": 2},
        "env": {"WMS_LATEST_EVERY": str(args.latest_every), "WMS_GPU_MEM_FRACTION": str(args.gpu_mem_fraction),
                "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", **env_road},
        "python": args.python or sys.executable, "code_commit": git_head(), "repo": str(REPO),
        "max_retries": args.max_retries,
    }
    write(root / "PLAN.json", body)
    status(root, state="prepared", phase=None)
    event(root, "prepared", worlds=wm["sha256"], init_wz=rec["out_sha256"])
    return body


# ------------------------------------------------------------------ init (data-independent pre-stage) ---
def init_only(args) -> dict:
    """WM2 sha check + geometry re-init + strict load into the phase A and B models (no data)."""
    out = Path(guard.check_path(args.out_dir))
    out.mkdir(parents=True, exist_ok=True)
    if args.map_spec:
        os.environ["WMS_MAP_SPEC"] = args.map_spec
    srcs = [plan.Source("wz_native", "/placeholder/wz_native", 0.5),
            plan.Source("wz_mappo_u0", "/placeholder/wz_mappo_u0", 0.5)]
    sets = {ph: plan.phase_overrides(ph, init_from="INIT.pt", sources=srcs, base="wz_native",
                                     device="cpu", profile="night") for ph in plan.PHASES}
    rec = reinit.reinit_checkpoint(
        Path(args.init), out / "INIT_WZ.pt", expected_sha256=args.init_sha,
        fresh=lambda: reinit.fresh_state_from_config(plan.CONFIGS["A"], sets["A"], repo=REPO),
        receipt_path=out / "REINIT_RECEIPT.json", config=plan.CONFIGS["A"])
    import torch
    from wms.cli.assemble import build_model, build_physics
    state = torch.load(out / "INIT_WZ.pt", map_location="cpu", weights_only=True)["model"]
    loads = {}
    for ph in plan.PHASES:
        cfg = load_cfg(plan.CONFIGS[ph], sets[ph])
        guard.check_config(cfg_mapping(cfg))
        model = build_model(cfg, build_physics(cfg))
        model.load_state_dict(state, strict=True)
        loads[ph] = len(model.state_dict())
    body = {"t": now(), "init_wz_sha256": rec["out_sha256"], "wm2_sha256": rec["source_sha256"],
            "n_reinit": rec["n_reinit"], "n_kept": rec["n_kept"], "strict_load_tensors": loads,
            "road": road_record(args.map_spec), "code_commit": git_head()}
    write(out / "INIT_CHECK.json", body)
    return body


# ------------------------------------------------------------------ probe (truth guard on a real batch) ---
def probe(args) -> dict:
    root = Path(args.root)
    p = read(root / "PLAN.json")
    os.environ.update({k: v for k, v in p["env"].items() if k == "WMS_MAP_SPEC"})
    from wms.cli.assemble import assemble
    ph = p["phases"]["A"]
    sets = [s if not s.startswith("runtime.device=") else 'runtime.device="cpu"' for s in ph["overrides"]]
    cfg = load_cfg(ph["config"], sets)
    out = root / "probe"
    out.mkdir(exist_ok=True)
    inputs = assemble(cfg, out)
    report = {"t": now(), "train_windows": len(inputs.train),
              "val_windows": len(inputs.val) if inputs.val is not None else 0, "keys": {}}
    for name, ds in (("train", inputs.train), ("val", inputs.val)):
        if ds is None or not len(ds):
            raise guard.GuardError(f"no {name} windows")
        rec = ds[0]
        guard.check_batch(rec)
        report["keys"][name] = sorted(rec)
    report["pass"] = True
    write(root / "BATCH_GUARD.json", report)
    return report


# ------------------------------------------------------------------ training ---
def _env(p: dict) -> dict:
    env = dict(os.environ)
    env.update(p["env"])
    src = str(Path(p["repo"]) / "src")
    env["PYTHONPATH"] = os.pathsep.join([src, p["repo"]] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return env


def _attempts(d: Path) -> list[Path]:
    return sorted((x for x in d.glob("a*") if x.is_dir() and x.name[1:].isdigit()), key=lambda x: int(x.name[1:]))


def _ckpt_step(path: Path) -> Optional[int]:
    import torch
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        step = payload.get("step")
        return None if step is None else int(step)
    except Exception:
        return None


def resume_point(phase_dir: Path) -> tuple[int, Optional[Path]]:
    """(optimizer steps completed in this phase, newest weights checkpoint) over all attempts."""
    best = (0, None)
    for a in _attempts(phase_dir):
        meta = read(a / "ATTEMPT.json") if (a / "ATTEMPT.json").is_file() else {}
        before = int(meta.get("done_before", 0))
        for c in [a / "run" / "wm_s_latest.pt", *sorted((a / "run").glob("wm_s_stage*.pt"))]:
            if c.is_file():
                s = _ckpt_step(c)
                if s is not None and before + s > best[0]:
                    best = (before + s, c)
    return best


def check_run(out: Path, *, expect_init_sha: Optional[str] = None, n_tensors: Optional[int] = None) -> list[str]:
    """Problems of a finished ``wms train`` out dir (empty = complete and finite)."""
    probs = []
    for name in ("wm_s.pt", "physics.json", "RUN_SUMMARY.json", "train_log.jsonl", "INIT_FROM.json"):
        if not (out / name).is_file():
            probs.append(f"missing {name}")
    if (out / "train_log.jsonl").is_file():
        rows = [json.loads(x) for x in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows:
            probs.append("empty train_log")
        for r in rows:
            loss = r.get("loss")
            if not isinstance(loss, (int, float)) or not math.isfinite(loss):
                probs.append(f"non-finite loss at step {r.get('step')}")
                break
    if expect_init_sha and (out / "INIT_FROM.json").is_file():
        probs += reinit.verify_init_report(read(out / "INIT_FROM.json"), expected_sha256=expect_init_sha,
                                           n_model_tensors=n_tensors or len(read(out / "INIT_FROM.json").get("loaded", [])))
    return probs


def _run_child(cmd: list, env: dict, log: Path, root: Path, cwd: str) -> int:
    global _CHILD
    with open(log, "ab") as fh:
        _CHILD = subprocess.Popen(cmd, cwd=cwd, env=env,
                                  stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        status(root, child_pid=_CHILD.pid, child_cmd=cmd[:6])
        event(root, "child_start", pid=_CHILD.pid, log=str(log))
        code = _CHILD.wait()
    event(root, "child_exit", pid=_CHILD.pid, code=code)
    _CHILD = None
    return code


def _stop(signum, _frame):
    if _CHILD is not None and _CHILD.poll() is None:
        _CHILD.terminate()          # own child only
    raise SystemExit(128 + int(signum))


def run_phase(root: Path, p: dict, phase: str, init: str, init_sha: str, *, smoke: bool = False) -> dict:
    """Run (or resume) one phase until DONE; returns the DONE record."""
    spec = p["phases"]["A" if smoke else phase]
    pdir = root / ("smoke" if smoke else f"phases/{phase}")
    pdir.mkdir(parents=True, exist_ok=True)
    if (pdir / "DONE.json").is_file():
        done = read(pdir / "DONE.json")
        if sha(done["final"]) != done["final_sha256"]:
            raise RuntimeError(f"{pdir}/DONE.json final checkpoint hash changed")
        return done
    env = _env(p)
    sets_base = [s for s in spec["overrides"] if not s.startswith("model.init_from=")]
    curriculum, optim = dict(spec["curriculum"]), dict(spec["optim"])
    if smoke:
        sets_base = [s for s in sets_base if not s.startswith(("curriculum.stage_steps=", "selection.val_max_windows="))]
        sets_base += ['curriculum.stage_steps=[2]', "selection.val_max_windows=2"]
        curriculum = {**curriculum, "stage_steps": [2]}
    total = plan.total_steps(curriculum)
    retries = 0
    while True:
        done_steps, ckpt = (0, None) if smoke else resume_point(pdir)
        if done_steps >= total and ckpt is not None:
            # trained through but the final save/validation did not finish: adopt the last weights
            rec = {"phase": phase, "final": str(ckpt), "final_sha256": sha(ckpt), "best": None,
                   "steps": done_steps, "note": "final validation not completed; last full-step weights"}
            write(pdir / "DONE.json", rec)
            return rec
        n = len(_attempts(pdir)) + 1
        out = pdir / f"a{n}"
        start_init = str(ckpt) if ckpt is not None else init
        sets = [f"model.init_from={json.dumps(start_init)}", *sets_base]
        if done_steps:
            sets += plan.resume_overrides(curriculum, optim, done_steps)
        out.mkdir(parents=True)
        write(out / "ATTEMPT.json", {"phase": phase, "smoke": smoke, "attempt": n, "done_before": done_steps,
                                     "init": start_init, "init_sha256": sha(start_init), "overrides": sets,
                                     "resumed": bool(done_steps), "started": now(),
                                     "note": ("resumed from weights; Adam moments and sampler restart"
                                              if done_steps else None)})
        # wms train needs a fresh/empty out dir: train into out/run, keep ATTEMPT.json beside it.
        run_dir = out / "run"
        cmd = [p["python"], "-m", "wms", "train", "--config", str(Path(p["repo"]) / spec["config"]),
               "--out", str(run_dir), "--mode", "run"]
        for s in sets:
            cmd += ["--set", s]
        status(root, state="smoke" if smoke else "training", phase=phase, attempt=n, done_before=done_steps)
        t0 = time.time()
        code = _run_child(cmd, env, out / "train.log", root, p["repo"])
        probs = [f"exit {code}"] if code else []
        probs += check_run(run_dir, expect_init_sha=sha(start_init))
        write(out / "RESULT.json", {"exit": code, "problems": probs, "wall_s": time.time() - t0, "ended": now()})
        if not probs:
            rec = {"phase": "smoke" if smoke else phase, "final": str(run_dir / "wm_s.pt"),
                   "final_sha256": sha(run_dir / "wm_s.pt"),
                   "best": str(run_dir / "wm_s_best.pt") if (run_dir / "wm_s_best.pt").is_file() else None,
                   "best_sha256": sha(run_dir / "wm_s_best.pt") if (run_dir / "wm_s_best.pt").is_file() else None,
                   "attempts": n, "steps": total, "init_sha256": init_sha, "ended": now()}
            write(pdir / "DONE.json", rec)
            event(root, "phase_done", phase=rec["phase"], final=rec["final_sha256"])
            return rec
        event(root, "attempt_failed", phase=phase, attempt=n, problems=probs[:5])
        new_steps, _ = (0, None) if smoke else resume_point(pdir)
        retries += 1
        if smoke or retries > int(p.get("max_retries", 2)) or new_steps <= done_steps:
            status(root, state="failed", phase=phase, problems=probs[:5])
            raise RuntimeError(f"phase {phase} attempt a{n} failed: {probs[:5]}")


def estimate_from_smoke(root: Path, p: dict) -> dict:
    from .estimate import estimate
    run_dir = _attempts(root / "smoke")[-1] / "run"
    rows = [json.loads(x) for x in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    est = estimate(p, measured_s_per_step_120=max(r.get("seconds", 0.0) for r in rows),
                   measured_peak_mb=max(float(r.get("cuda_peak_mb") or 0.0) for r in rows))
    write(root / "ESTIMATE.json", est)
    return est


def manifest(root: Path, p: dict, done_a: dict, done_b: dict) -> dict:
    wm = read(root / "WORLDS.json")
    hit = worlds.overlap(wm)
    if hit:
        raise worlds.WorldError(f"reserved qualification worlds in training: {hit}")
    body = {
        "schema": "wzwm.checkpoint_manifest/1", "created": now(), "map": p["map"], "road": p["road"],
        "candidate_final": done_b["final"], "candidate_final_sha256": done_b["final_sha256"],
        "candidate_best": done_b.get("best"), "candidate_best_sha256": done_b.get("best_sha256"),
        "phase_a_final": done_a["final"], "phase_a_final_sha256": done_a["final_sha256"],
        "init_wm2_sha256": p["init"]["wm2_sha256"], "init_wz_sha256": p["init"]["init_wz_sha256"],
        "reinit_prefixes": p["init"]["reinit_prefixes"], "reinit_receipt_sha256": p["init"]["reinit_receipt_sha256"],
        "train_worlds": wm["train_worlds"], "val_worlds": wm["val_worlds"],
        "train_seeds": wm["train_seeds"], "val_seeds": wm["val_seeds"], "worlds_sha256": wm["sha256"],
        "reserved_qualification_overlap": hit, "sources": p["sources"],
        "phases": {k: {"config": v["config"], "config_sha256": v["config_sha256"], "overrides": v["overrides"]}
                   for k, v in p["phases"].items()},
        "code_commit": p["code_commit"], "plan_sha256": sha(root / "PLAN.json"),
    }
    write(root / "CHECKPOINT_MANIFEST.json", body)
    return body


def train(args) -> int:
    root = Path(args.root)
    p = read(root / "PLAN.json")
    signal.signal(signal.SIGTERM, _stop)
    with RunLock(root):
        status(root, state="starting", runner_pid=os.getpid())
        event(root, "runner_start", pid=os.getpid())
        init_wz, init_sha = p["init"]["init_wz"], p["init"]["init_wz_sha256"]
        if sha(init_wz) != init_sha:
            raise RuntimeError("INIT_WZ.pt hash changed")
        if p["smoke"]["enabled"]:
            run_phase(root, p, "A", init_wz, init_sha, smoke=True)
            if not (root / "ESTIMATE.json").is_file():
                est = estimate_from_smoke(root, p)
                event(root, "estimate", total_h=est.get("total_h"))
        done_a = run_phase(root, p, "A", init_wz, init_sha)
        done_b = run_phase(root, p, "B", done_a["final"], done_a["final_sha256"])
        body = manifest(root, p, done_a, done_b)
        status(root, state="done", phase=None, candidate=body["candidate_final_sha256"])
        event(root, "done", candidate=body["candidate_final_sha256"])
    return 0


def show(args) -> int:
    root = Path(args.root)
    st = read(root / "STATUS.json") if (root / "STATUS.json").is_file() else {}
    tail = {}
    for d in [root / "smoke", root / "phases/A", root / "phases/B"]:
        for a in _attempts(d) if d.is_dir() else []:
            log = a / "run" / "train_log.jsonl"
            if log.is_file():
                lines = [x for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
                if lines:
                    r = json.loads(lines[-1])
                    tail[f"{d.name}/{a.name}"] = {k: r.get(k) for k in ("step", "stage_s", "loss", "seconds", "cuda_peak_mb")}
    print(json.dumps({"status": st, "last_rows": tail}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools.wzwm.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--root", required=True)
    pr.add_argument("--init", default=WM2_GPUA)
    pr.add_argument("--init-sha", default=WM2_SHA256)
    pr.add_argument("--source", action="append", required=True, help="NAME=PATH[@MIX] (repeat)")
    pr.add_argument("--split-by-arm", action="store_true")
    pr.add_argument("--base", default=None)
    pr.add_argument("--paired", action="store_true", help="enable paired_diff if a source has pair keys")
    pr.add_argument("--map-spec", default=os.environ.get("WMS_MAP_SPEC"))
    pr.add_argument("--map-name", default="wz3to2")
    pr.add_argument("--allow-no-road", action="store_true", help="dry runs/tests only")
    pr.add_argument("--profile", default="night", choices=sorted(plan.PROFILES))
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--python", default=None)
    pr.add_argument("--latest-every", type=int, default=5)
    pr.add_argument("--gpu-mem-fraction", type=float, default=0.5)
    pr.add_argument("--max-retries", type=int, default=2)
    pr.add_argument("--no-smoke", action="store_true")
    ini = sub.add_parser("init")
    ini.add_argument("--out-dir", required=True)
    ini.add_argument("--init", default=WM2_GPUA)
    ini.add_argument("--init-sha", default=WM2_SHA256)
    ini.add_argument("--map-spec", default=os.environ.get("WMS_MAP_SPEC"))
    for name in ("probe", "train", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("--root", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        body = prepare(args)
        print(json.dumps({k: body[k] for k in ("root", "n_train_worlds", "n_val_worlds", "worlds_sha256")}))
        return 0
    if args.cmd == "init":
        print(json.dumps(init_only(args), default=str))
        return 0
    if args.cmd == "probe":
        print(json.dumps(probe(args)["keys"]))
        return 0
    if args.cmd == "train":
        return train(args)
    return show(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""MAIN14 gpuR-resident fail-closed waiter and four-cycle driver. Never signals a PID."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import fcntl
except ImportError:  # allow pure helper tests on the Windows integration host
    fcntl = None

import loop14 as loop

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.main14.verdict_snapshot import digest  # noqa: E402

TESTS = ["test_coopw.py", "test_sumocredit.py", "test_sumocredit_kcheck.py",
         "test_gate_panel.py", "test_loop14.py", "test_loop14c.py", "test_main14.py", "test_native_lane.py",
         "train/test_gpu_memory_cap.py",
         "test_policy14_integration.py", "test_policy14_critic.py", "test_policy14_lanes.py",
         "test_policy14e.py", "test_policy14_r13e.py", "test_native_adoption.py"]


def say(event, **fields):
    print(json.dumps({"t": loop.now(), "event": event, **fields}), flush=True)


def proc_identity(pid):
    """pid reuse and zombies count as exited; /proc read only."""
    try:
        fields = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def chain_finished(arm, summary):
    return (proc_identity(636520) != arm["round14b_start_ticks"] and Path(summary).is_file())


def timely_verdict(verdict, deadline):
    return bool(verdict.get("ready") and verdict["verdict_mtime"] <= deadline)


def verify_source(arm):
    for rel, expected in arm["source_files"].items():
        target = ROOT / rel
        loop.guard(target)
        if not target.resolve().is_relative_to(ROOT) or digest(target) != expected:
            raise RuntimeError(f"deployed source changed: {rel}")
    if digest(Path(arm["registration"])) != arm["registration_sha256"]:
        raise RuntimeError("registration changed since arm")
    if digest(Path(arm["config"])) != arm["config_sha256"]:
        raise RuntimeError("registered config changed since arm")


def environment(cfg, arm_dir):
    env = {**os.environ, **cfg["policy"].get("env", {}), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "1"}
    env["C4_ALLOWED_ROOTS"] = ":".join([*cfg["policy"]["allowed_roots"], cfg["loop_root"], str(arm_dir)])
    return env


def run(argv, cfg, arm_dir, label, timeout=7200, env=None):
    if (arm_dir / "STOP_ARM").exists():
        raise RuntimeError("STOP_ARM requested")
    say("command", label=label, argv=list(map(str, argv)))
    with (arm_dir / f"{label}.log").open("ab") as stream:
        result = subprocess.run(list(map(str, argv)), cwd=ROOT,
                                env=env or environment(cfg, arm_dir), stdin=subprocess.DEVNULL,
                                stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{label} failed, exit {result.returncode}; owner intervention required")
    say("passed", label=label)


def fetch_verdict(cfg):
    cpu = cfg["hosts"]["cpu"]
    command = f"{cpu['python']} {cpu['code_root']}/tools/main14/verdict_snapshot.py"
    if loop.fast_enabled(cfg, "base_three"):
        command += " --three-spec " + shlex.quote(json.dumps(cfg["fastloop"]["verdict_candidates"]))
    argv = [*cpu["ssh"], command]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=90, check=False)
        if proc.returncode:
            return {"ready": False, "reason": f"cpuX verdict query exit {proc.returncode}: {proc.stderr[-400:]}"}
        return json.loads(proc.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"ready": False, "reason": f"cpuX verdict temporarily unavailable: {error}"}


def prepare_horizon(cfg, state, arm_dir):
    """Each accepted WM needs its own receipt; never reuse the threshold from a new WM."""
    acc = state["accepted_wm"]
    target = arm_dir / f"horizon_{acc['sha256']}.json"
    registered = ROOT / "configs/coopw_horizon_gate_13E_b2.json"
    doc = loop.read_json(registered)
    if not target.exists():
        if doc["wm_sha256"] == acc["sha256"]:
            shutil.copyfile(registered, target)
        else:
            argv = [sys.executable, "tools/coopw/horizon_gate.py", "--wm-ckpt", acc["checkpoint"],
                    "--wm-config", acc["config"], "--pairs", cfg["wm"]["env"]["WMS_DATA_PAIRS300"],
                    "--groups", "96", "--horizons", "10", "30", "60", "--threshold", str(registered),
                    "--out", str(target)]
            try:
                run(argv, cfg, arm_dir, f"horizon_{acc['sha256'][:12]}", timeout=7200,
                    env={**environment(cfg, arm_dir), "CUDA_VISIBLE_DEVICES": ""})
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                # A measurement failure cannot authorise a longer horizon.
                loop.write_atomic(target, {"schema": "coopw_horizon_gate/1", "wm_sha256": acc["sha256"],
                                           "horizons": {}, "measurement_error": str(error)})
                say("horizon_capped_at_10s", error=str(error))
    loop.write_atomic(arm_dir / "horizon_gate_current.json", loop.read_json(target))


def retire_pair_cache(cfg, completed, arm_dir):
    """Only own completed-cycle cache; exact hashes must match the retained cpuX copy first."""
    receipt = completed["phases"]["convert"]["receipt"]
    local = Path(receipt["pairs"])
    data_root = Path(cfg["loop_root"]) / "data"
    if local.is_symlink() or local.parent.resolve() != data_root.resolve() or not local.name.startswith("pairs_c"):
        raise RuntimeError("pair cache is not an owned MAIN14 output")
    if not local.exists():
        return
    if loop.fast_enabled(cfg, "two_box"):
        worlds = completed["phases"]["collect"]["receipt"]["worlds"]
        if not all(loop.fast_module().artifact_valid(w["artifact"]) for w in worlds):
            raise RuntimeError("durable world archive missing/corrupt")
        loop.write_atomic(arm_dir / f"CACHE_ARCHIVE_c{completed['cycle']:02d}.json",
                          {"local": str(local), "retained_worlds": worlds, "verified_at": loop.now()})
        shutil.rmtree(local)
        return
    files = {}
    for p in local.rglob("*"):
        loop.guard(p)
        if p.is_symlink():
            raise RuntimeError("pair cache contains a symlink")
        if p.is_file():
            files[p.relative_to(local).as_posix()] = digest(p)
    if not files:
        raise RuntimeError("pair cache empty")
    col = completed["phases"]["collect"]["receipt"]
    remote = col["work"] + "/" + local.name
    cpu = cfg["hosts"]["cpu"]
    cmd = [*cpu["ssh"], f"{cpu['python']} {cpu['code_root']}/tools/main14/verify_cache.py --root {remote}"]
    result = subprocess.run(cmd, input=json.dumps(files), capture_output=True, text=True, timeout=1200, check=False)
    if result.returncode or json.loads(result.stdout)["verified"] != len(files):
        raise RuntimeError(f"cpuX cache archive verification failed: {result.stderr[-400:]}")
    loop.write_atomic(arm_dir / f"CACHE_ARCHIVE_c{completed['cycle']:02d}.json",
                      {"local": str(local), "retained_cpuX": remote, "files": files, "verified_at": loop.now()})
    shutil.rmtree(local)
    say("own_completed_pair_cache_retired", cycle=completed["cycle"], files=len(files), cpuX=remote)


def drive(cfg, arm_dir):
    if loop.fast_enabled(cfg, "overlap_policy"):
        return loop.fast_module().drive(loop, sys.modules[__name__], cfg, arm_dir)
    while True:
        if (arm_dir / "STOP_ARM").exists() or (Path(cfg["loop_root"]) / "STOP_DRIVER").exists():
            say("graceful_stop")
            return
        state = loop.load_state(cfg)
        if state.get("stopped"):
            say("loop_stopped", reason=state["stopped"], sim_s=state["sim_s_charged"])
            return
        phase = loop._phase(state)["name"]
        if phase == "policy":
            prepare_horizon(cfg, state, arm_dir)
        say("phase_start", cycle=state["cycles"][-1]["cycle"], phase=phase)
        code = loop.run_phase(cfg, phase)
        if code == 5:
            say("gpu_busy_wait", seconds=120)
            time.sleep(120)
            continue
        if code not in (0, 4):
            raise RuntimeError(f"phase {phase} exit {code}; no automatic retry")
        if phase == "check":
            after = loop.load_state(cfg)
            completed = next(c for c in after["cycles"] if c["cycle"] == state["cycles"][-1]["cycle"])
            retire_pair_cache(cfg, completed, arm_dir)


def launch(cfg, arm, verdict, arm_dir):
    verify_source(arm)
    v5 = arm_dir.parent
    fallback = cfg["wm"]["initial_wm"]
    fallback_sha = loop.read_json(cfg["registration"])["sources"]["fallback_wm_sha256"]
    if digest(Path(fallback)) != fallback_sha or verdict["reference_checkpoint_sha256"] != fallback_sha:
        raise RuntimeError("13E b2 FINAL sha mismatch")
    if loop.fast_enabled(cfg, "base_three"):
        for label, model in cfg["fastloop"]["base_models"].items():
            if digest(Path(model["checkpoint"])) != verdict["checkpoint_sha256"][label]:
                raise RuntimeError(f"{label} scored checkpoint does not match deployed FINAL")
        selected = cfg["fastloop"]["base_models"][verdict["base"]]
        cfg["wm"]["initial_wm"] = selected["checkpoint"]
        cfg["wm"]["initial_wm_config"] = selected["config"]
    elif verdict["base"] == "14b":
        cfg["wm"]["initial_wm"] = str(v5 / "runs/c4_wm_v10_round14_b/wm_s.pt")
        cfg["wm"]["initial_wm_config"] = str(v5 / "runs/c4_wm_v10_round14_b/config.resolved.json")
    if not loop.fast_enabled(cfg, "base_three") and digest(v5 / "runs/c4_wm_v10_round14_b/wm_s.pt") != verdict["scored_checkpoint_sha256"]:
        raise RuntimeError("14b scored checkpoint does not match gpuR FINAL")
    loop.write_atomic(arm_dir / "BASE_SELECTION.json", verdict)
    launch_config = arm_dir / "launch_config.json" if cfg.get("fastloop") else v5 / "main14_launch_config.json"
    loop.write_atomic(launch_config, cfg)
    test_tmp = str(arm_dir / "pytest_launch") if cfg.get("fastloop") else "/tmp/main14_pytest_launch"
    run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", test_tmp,
         *[f"tests/unit/{t}" for t in TESTS],
         *(["tests/unit/test_fastloop.py", "tests/test_speed2_flags.py"] if cfg.get("fastloop") else [])], cfg, arm_dir, "unit_tests",
        env={**environment(cfg, arm_dir), "CUDA_VISIBLE_DEVICES": ""})
    base_args = [sys.executable, "-m", "wms.control.policy14_cli", "--config", cfg["policy"]["config"],
                 "--wm-ckpt", cfg["wm"]["initial_wm"], "--wm-config", cfg["wm"]["initial_wm_config"],
                 "--real-returns", cfg["policy"]["initial_real_returns"], "--device", "cpu", "--compact"]
    run([*base_args, "--smoke", "--max-updates", "2", "--output-dir", str(arm_dir / "smoke")],
        cfg, arm_dir, "smoke", env={**environment(cfg, arm_dir), "CUDA_VISIBLE_DEVICES": ""})
    if not (arm_dir / "gate_panel.json").exists():
        run([*base_args, "--max-updates", "1", "--output-dir", str(arm_dir / "panel_build"),
             "--gate-panel-dump", str(arm_dir / "gate_panel.json")], cfg, arm_dir, "panel_build")
    if "gate_panel_sha256" in arm and digest(arm_dir / "gate_panel.json") != arm["gate_panel_sha256"]:
        raise RuntimeError("frozen gate panel changed since arm")
    verify_source(arm)
    loop.gpu_guard(cfg, loop.host(cfg, "gpu"))
    loop.init(cfg, argparse.Namespace(adopt_policy=None))
    loop.write_atomic(arm_dir / "LAUNCHED.json", {"t": loop.now(), "pid": os.getpid(),
                      "config": str(launch_config), "config_sha256": digest(launch_config),
                      "registration_sha256": arm["registration_sha256"], "base": verdict["base"],
                      "wm_sha256": digest(Path(cfg["wm"]["initial_wm"])),
                      "gate_panel_sha256": digest(arm_dir / "gate_panel.json")})
    say("launch_checks_passed", base=verdict["base"])
    drive(cfg, arm_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--check-only", action="store_true")
    args = p.parse_args()
    arm_path = Path(args.arm)
    arm_dir = arm_path.parent
    arm = loop.read_json(arm_path)
    cfg = loop.load_config(arm["config"])
    verify_source(arm)
    loop.check_registration(cfg)
    if args.check_only:
        print(json.dumps({"verified": True, "code_sha": arm["code_sha"],
                          "source_files": len(arm["source_files"]), "sim_s_per_cycle": loop.collect_sim_s(cfg),
                          **fetch_verdict(cfg)}))
        return 0
    if fcntl is None:
        raise RuntimeError("waiter requires Linux flock")
    with (arm_dir / "waiter.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (arm_dir / "waiter.pid").write_text(str(os.getpid()) + "\n")
        say("armed", pid=os.getpid(), code_sha=arm["code_sha"], deadline=arm["verdict_deadline"])
        deadline = datetime.fromisoformat(arm["verdict_deadline"]).timestamp()
        observed = None
        try:
            while True:
                if (arm_dir / "STOP_ARM").exists():
                    say("disarmed_by_marker")
                    return 0
                verdict = observed or fetch_verdict(cfg)
                if timely_verdict(verdict, deadline):
                    observed = verdict
                    loop.write_atomic(arm_dir / "VERDICT_OBSERVED.json", verdict)
                elif time.time() >= deadline:
                    say("expired_no_timely_verdict", reason=verdict.get("reason"))
                    loop.write_atomic(arm_dir / "EXPIRED.json", {"t": loop.now(), "verdict": verdict})
                    return 2
                if not chain_finished(arm, arm["round14b_summary"]):
                    say("waiting_round14b_chain", verdict_ready=verdict.get("ready", False),
                        verdict_reason=verdict.get("reason"))
                elif not timely_verdict(verdict, deadline):
                    say("waiting_qualification", reason=verdict.get("reason"))
                else:
                    try:
                        loop.gpu_guard(cfg, loop.host(cfg, "gpu"))
                    except loop.GpuBusy as error:
                        say("waiting_gpu", reason=str(error))
                    else:
                        launch(cfg, arm, verdict, arm_dir)
                        return 0
                time.sleep(120)
        except Exception as error:  # fail closed; no autonomous retry of tests/smoke/failed phases
            loop.write_atomic(arm_dir / "WAIT_OWNER.json", {"t": loop.now(), "error": f"{type(error).__name__}: {error}"})
            say("wait_owner", error=str(error))
            return 3


if __name__ == "__main__":
    raise SystemExit(main())

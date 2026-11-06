#!/usr/bin/env python3
"""Standalone R13C measurement waiter; never launches/retries training.

Arm before the production exec, retaining that exec's PID/start ticks. SIGTERM
requires the same Linux process identity AND the exact production --out argv.
The legacy R12 verdicts are context; only the registered R13C paired rules stop
training (or a measurement error, fail closed). TRAIN here means the 963000..
963959 corpus, including its internal validation-role diagnostic states.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "hetdata_truth")
SOURCES = ("v5_native", "genrand_plain")
EMISSIONS = ("temporal_accel_off", "residual_accel_off", "follower_courtesy_off")
WORKERS = 8  # one pool at a time, shared lock across both waiter modes
POLL = 10


def safe_path(path):
    path = Path(path)
    for candidate in (str(path), str(path.resolve())):
        if any(token in candidate.lower() for token in FORBIDDEN):
            raise ValueError(f"restricted path: {path}")
    return path


def read_json(path):
    return json.loads(safe_path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with safe_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def proc_stat(pid):
    """Return (start ticks, state); comm may itself contain spaces and ')'."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = text.rsplit(")", 1)[1].split()
    return int(fields[19]), fields[0]


def _pidfd_syscall(number, *args):
    """Raw pidfd fallback for the Linux x86-64 syscall ABI only."""
    if (sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64")
            or ctypes.sizeof(ctypes.c_long) != 8 or ctypes.sizeof(ctypes.c_void_p) != 8):
        raise OSError(errno.ENOSYS, "pidfd syscall fallback requires Linux x86-64")
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.restype = ctypes.c_long
    syscall.argtypes = [ctypes.c_long]  # remaining variadic arguments are explicitly typed
    result = syscall(ctypes.c_long(number), *args)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_open(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_send_signal(fd, sig):
    if hasattr(signal, "pidfd_send_signal"):
        return signal.pidfd_send_signal(fd, sig)
    return _pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig),
                          ctypes.c_void_p(None), ctypes.c_uint(0))


class ProductionProcess:
    def __init__(self, pid, run, start_ticks=None):
        if pid <= 1 or pid == os.getpid():
            raise ValueError("--pid must be the separate production process")
        self.pid, self.run = pid, Path(run).resolve()
        current = proc_stat(pid)
        self.start_ticks = start_ticks if start_ticks is not None else (current[0] if current else None)
        if self.start_ticks is None:
            raise ValueError("PID already gone: supply its captured --start-ticks")

    def alive(self):
        current = proc_stat(self.pid)
        return bool(current and current[0] == self.start_ticks and current[1] not in ("Z", "X"))

    def is_production(self):
        if not self.alive():
            return False
        try:
            args = Path(f"/proc/{self.pid}/cmdline").read_bytes().decode().split("\0")
        except FileNotFoundError:
            return False
        # The gate execs python -m wms train ... --mode run --out ABSOLUTE_RUN.
        def option(name):
            return args[args.index(name) + 1] if name in args and args.index(name) + 1 < len(args) else None
        return (any(args[i:i + 3] == ["-m", "wms", "train"] for i in range(len(args)))
                and option("--mode") == "run" and option("--out") == str(self.run))

    def terminate(self):
        """A pidfd pins identity across the check/signal race; never signal a PID group."""
        if not self.is_production():
            return "not sent: production PID/start identity/argv no longer matches"
        try:
            fd = pidfd_open(self.pid)
            try:
                if not self.is_production():
                    return "not sent: identity changed before signal"
                pidfd_send_signal(fd, signal.SIGTERM)
            finally:
                os.close(fd)
        except ProcessLookupError:
            return "not sent: production process exited"
        return "SIGTERM sent to verified production pidfd"


def stop_reasons(candidate_share, baseline_share, candidate_mae, baseline_mae):
    vals = (candidate_share, baseline_share, candidate_mae, baseline_mae)
    if not all(math.isfinite(x) and x >= 0 for x in vals) or max(vals[:2]) > 1:
        raise ValueError("invalid/nonfinite paired measurements")
    reasons = []
    if candidate_share > 2 * baseline_share:
        reasons.append("full-road lane1 hard-brake share > 2 * SAME-panel S-R12C share")
    if candidate_mae > baseline_mae + 0.1:
        reasons.append("all-lane1 one-step MAE > SAME-panel S-R12C MAE + 0.1 m/s^2")
    return reasons


def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 963000 <= seed < 963960:
        raise ValueError(f"not a registered TRAIN-corpus seed: {seed!r}")


def check_strings(value):
    """Reject restricted references before any referenced data can be opened."""
    if isinstance(value, str):
        if any(t in value.lower() for t in FORBIDDEN):
            raise ValueError("restricted reference in provenance")
    elif isinstance(value, dict):
        for k, v in value.items():
            check_strings(k)
            check_strings(v)
            if k == "seed" and isinstance(v, int) and 964000 <= v <= 964199:
                raise ValueError("prohibited evaluation seed")
    elif isinstance(value, list):
        for v in value:
            check_strings(v)


def train_panel(root, work):
    """Validate both entire one-step roots, then materialize a fixed 24-state panel.

    The old one-step reader promotes all manifest worlds to train: do not trust
    that promotion as provenance. Internal train/val roles are explicitly recorded.
    No source paths or auxiliary/assignment files from manifests are opened.
    """
    manifests, receipt = {}, []
    for source in SOURCES:
        path = safe_path(root / "data" / source / "C4_EPISODES_MANIFEST.json")
        doc = read_json(path)
        entries = doc["episodes"]
        index = {}
        for e in entries:
            check_strings(e)
            validate_seed(e["seed"])
            if e.get("split_role") not in ("train", "val"):
                raise ValueError("unregistered manifest split role")
            directory = safe_path(root / "data" / source / e["dir"])
            directory.resolve().relative_to((root / "data" / source / "episodes").resolve())
            if e["episode"] in index:
                raise ValueError("duplicate manifest episode")
            index[e["episode"]] = e
        manifests[source] = index
        receipt.append({"source": source, "sha256": digest(path), "episodes": len(entries)})
    source_path = safe_path(root / "chk_r10b" / "states192.json.gz")
    with gzip.open(source_path, "rt", encoding="utf-8") as stream:
        states = json.load(stream)
    if len(states) != 192:
        raise ValueError("expected registered 192-state source")
    # Validate the whole source too: a prohibited record is never silently skipped.
    for r in states:
        check_strings(r)
        validate_seed(r["seed"])
    selected, metadata = states[::8], []
    for r in selected:
        check_strings(r)
        validate_seed(r["seed"])
        entry = manifests.get(r["source"], {}).get(r["episode"])
        if entry is None or entry["seed"] != r["seed"]:
            raise ValueError("state not matched to TRAIN manifest")
        if r.get("dt_s") != 0.5 or r.get("recorded_horizon_steps", {}).get("native", 0) < 120:
            raise ValueError("state lacks 60 s native replay at dt=0.5")
        metadata.append({k: r[k] for k in ("key", "source", "episode", "seed", "t0_s")}
                       | {"split_role": entry["split_role"]})
    if len({r["key"] for r in selected}) != 24:
        raise ValueError("need 24 distinct TRAIN states")
    target = work / "states24.json.gz"
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        json.dump(selected, stream)
    return target, {"corpus": "TRAIN 963000..963959 (internal train/val roles)",
                    "manifests": receipt, "states_source_sha256": digest(source_path),
                    "states": metadata, "panel_sha256": digest(target)}


def paired_mae(os_dir, arms):
    import numpy as np
    reference, result = None, {}
    for arm in arms:
        with np.load(os_dir / f"onestep_{arm}.npz", allow_pickle=False) as z:
            cols, rows = list(z["cols"]), z["rows"]
            identity = rows[:, [cols.index(c) for c in ("win", "t", "slot", "lane", "rec")]]
            if reference is not None and not np.array_equal(reference, identity, equal_nan=True):
                raise ValueError("one-step arms do not share identical supervision rows")
            reference = identity
            err = rows[:, cols.index("fin")].astype(float) - rows[:, cols.index("rec")]
            sel = rows[:, cols.index("lane")] == 1
            if sel.sum() < 5 or not np.isfinite(err[sel]).all():
                raise ValueError(f"missing/nonfinite lane1 one-step rows: {arm}")
            result[arm] = {"n": int(sel.sum()), "mae": float(np.abs(err[sel]).mean())}
    return result


def hard_brakes(roll_dir, arms):
    """All road positions, every 0.5 s transition; never the zone-only 'rows'."""
    import numpy as np
    out = {}
    expected = [f"s{i:03d}_k0.npz" for i in range(24)]
    for arm in arms:
        directory = roll_dir / arm
        if sorted(p.name for p in directory.glob("*.npz")) != expected:
            raise ValueError(f"incomplete 24-state rollout: {arm}")
        n = hard = 0
        for name in expected:
            with np.load(directory / name, allow_pickle=False) as z:
                v, lane, mask, cav = (z[k] for k in ("d_v", "d_lane", "d_mask", "d_cav"))
                if v.ndim != 2 or v.shape[0] != 121 or lane.shape != v.shape or mask.shape != v.shape:
                    raise ValueError("expected full 60 s trajectory arrays")
                sel = (mask[:-1] & mask[1:] & (lane[:-1] == 1) & (lane[1:] == 1) & ~cav[None, :])
                accel = np.diff(v.astype(float), axis=0) / 0.5
                if not np.isfinite(accel[sel]).all():
                    raise ValueError("nonfinite lane1 rollout acceleration")
                n += int(sel.sum())
                hard += int((accel[sel] < -3).sum())
        if n == 0:
            raise ValueError("empty lane1 rollout panel")
        out[arm] = {"n": n, "a_lt_minus3": hard, "share": hard / n}
    return out


class Waiter:
    def __init__(self, args):
        self.repo, self.root = safe_path(args.repo).resolve(), safe_path(args.root).resolve()
        self.mode = args.mode
        self.run = self.root / "runs" / "c4_wm_v10_round13c"
        self.process = ProductionProcess(args.pid, self.run, args.start_ticks)
        self.out = self.root / "chk_r13c" / self.mode
        self.work = Path("/dev/shm") / f"chk_r13c_{self.mode}"
        self.commands = []

    def command(self, script, *args):
        command = ["nice", "-n", "15", sys.executable, "-B", str(safe_path(script)), *map(str, args)]
        self.commands.append(command)
        env = os.environ.copy()
        env.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = os.pathsep.join(map(str, (self.repo / "src", self.repo,
                                            self.repo / "tools", self.repo / "tools/r10b_checks")))
        log = self.work / f"command_{len(self.commands):02d}.log"
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.run(command, cwd=self.repo, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=4 * 3600, check=False)
        if proc.returncode:
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 4000))
                tail = stream.read().decode(errors="replace")
            raise RuntimeError(f"measurement command exit {proc.returncode}: {tail}")

    def wait_checkpoint(self):
        while True:
            alive = self.process.alive()
            if self.mode == "midrun":
                checkpoint = self.run / "wm_s_stage120.pt"
                if checkpoint.is_file():
                    return checkpoint
                if not alive:
                    raise RuntimeError("production exited/replaced without stage120 checkpoint")
            elif (self.run / "RUN_SUMMARY.json").is_file() and not alive:
                for name in ("wm_s.pt", "wm_s_stage300.pt", "wm_s_best.pt", "wm_s_stage120.pt"):
                    checkpoint = self.run / name
                    if checkpoint.is_file():
                        return checkpoint
                raise RuntimeError("completed production has no checkpoint")
            elif not alive:
                raise RuntimeError("production exited/replaced without RUN_SUMMARY.json")
            time.sleep(POLL)

    def snapshot(self, src, name):
        """Bounded stable-copy + read retries only; preserve inference physics."""
        import torch
        torch.set_num_threads(1)
        src = safe_path(src)
        directory = self.work / name
        directory.mkdir()
        dst = directory / src.name
        error = None
        for _ in range(6):
            try:
                before = src.stat()
                time.sleep(3)
                stable = src.stat()
                if (before.st_size, before.st_mtime_ns) != (stable.st_size, stable.st_mtime_ns) or not stable.st_size:
                    continue
                shutil.copyfile(src, dst)
                after = src.stat()
                if (stable.st_size, stable.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    continue
                payload = torch.load(dst, map_location="cpu", weights_only=True)
                if payload.get("format") != "wm_s/1" or not payload.get("model"):
                    raise ValueError("not a wm_s/1 checkpoint")
                cfg = payload["config"]
                physics = safe_path(src.with_name("physics.json"))
                if physics.is_file():
                    shutil.copyfile(physics, directory / "physics.json")
                elif not cfg.get("physics"):
                    raise ValueError("checkpoint has no inference physics")
                calibration = safe_path(src.with_name("lane_calib.json"))
                if calibration.is_file():
                    shutil.copyfile(calibration, directory / calibration.name)
                return dst, {"source": str(src), "sha256": digest(dst),
                             "emissions": {k: cfg.get(k, False) for k in EMISSIONS}}
            except (OSError, RuntimeError, ValueError, EOFError) as exc:
                error = exc
        raise RuntimeError(f"checkpoint never became complete: {src}: {error}")

    def measure(self, checkpoint):
        # No stale files can satisfy DONE/report checks after a failed prior run.
        self.work.mkdir(exist_ok=False)
        panel, provenance = train_panel(self.root, self.work)
        candidate, candidate_info = self.snapshot(checkpoint, "candidate")
        baseline, baseline_info = self.snapshot(self.root / "runs/c4_wm_v10_round12c/wm_s.pt", "baseline")
        r9_src = self.root / "runs/c4_wm_v10_round9/wm_s_stage300.pt"
        r9, r9_info = self.snapshot(r9_src, "r9")
        if not all(baseline_info["emissions"].values()) or any(r9_info["emissions"].values()):
            raise ValueError("unexpected S-R12C / round9 stored emission flags")
        checkpoints = {"r10b": candidate_info, "r12c": baseline_info, "r9": r9_info}
        tools = self.repo / "tools"
        variables = ["--var", f"r10b={candidate}", "--var", f"r12c={baseline}", "--var", f"r9={r9}"]
        if self.mode == "postrun":
            hz = self.work / "headzero" / "wm_s.pt"
            hz.parent.mkdir()
            receipt = self.out / "headzero_receipt.json"
            self.command(tools / "r10b_checks/make_headzero.py", candidate, hz, receipt)
            report = read_json(receipt)
            if not report.get("all_other_tensors_equal") or not report.get("zeroed_are_zero"):
                raise ValueError("headzero verification failed")
            variables += ["--var", f"headzero={hz}"]
        overrides = {"r10b": candidate_info["emissions"],
                     "r12c": {"driver_param_mode": "physical", **dict.fromkeys(EMISSIONS, True)},
                     "r9": {"driver_param_mode": "legacy", "driver_param_init": "checkpoint",
                             "residual_accel_reinit": False, **dict.fromkeys(EMISSIONS, False)}}
        if self.mode == "postrun":
            overrides["headzero"] = candidate_info["emissions"]
        arm_sets = []
        for arm, settings in overrides.items():
            for key, value in settings.items():
                arm_sets += ["--arm-set", f"{arm}=model.{key}={str(value).lower()}"]
        limits = ["--limit", "24", "--stride-eps", "10"] if self.mode == "midrun" else ["--stride-eps", "4"]
        os_dir = self.work / "os"
        self.command(tools / "r10b_checks/run_onestep.py", "--repo", self.repo,
                     "--config", "c4_wm_v10round13c.toml", *variables, *arm_sets,
                     "--replay-present", "filled", "--base-cols", "--horizon-s", "120",
                     "--roots", *(self.root / "data" / s for s in SOURCES),
                     "--workers", WORKERS, *limits, "--out", os_dir)
        done = read_json(os_dir / "ONESTEP_DONE.json")
        if done.get("replay_present") != "filled" or not done.get("base_cols") or not done.get("windows"):
            raise ValueError("one-step completion/presence/columns missing")
        if self.mode == "midrun" and done["windows"] != 24:
            raise ValueError("midrun requires exactly 24 one-step windows")
        arms = ["r10b", "r12c", "r9"] + (["headzero"] if self.mode == "postrun" else [])
        mae = paired_mae(os_dir, arms)
        self.command(tools / "r12_checks/base_calibration.py", "--os", os_dir,
                     "--out", self.out / "BASE.md", "--json", self.out / "BASE.json")
        self.command(tools / "r12_checks/midrun_check.py", "--os-dir", os_dir,
                     "--out-json", self.out / "MIDRUN.json", "--out-md", self.out / "MIDRUN.md")
        base, legacy = read_json(self.out / "BASE.json"), read_json(self.out / "MIDRUN.json")
        if (base.get("replay_present") != "filled" or "error" in legacy
                or legacy.get("decision") not in ("STOP", "CONTINUE", "UNDECIDED")
                or not all(a in base.get("classes", {}) for a in ("r10b", "r9"))
                or any(base.get("rules", {}).get(k, ["MISSING"])[0] not in ("PASS", "REJECT", "UNDECIDED")
                       for k in ("BASE-1", "BASE-2", "BASE-3", "BASE-5"))):
            raise ValueError("BASE/midrun report error or missing results")
        if self.mode == "postrun" and "headzero" not in base.get("classes", {}):
            raise ValueError("postrun BASE lacks headzero read")
        roll = self.work / "roll"
        self.command(self.root / "diag_brake/d11_run5.py", "--repo", self.repo,
                     "--states", panel, "--limit", "24", "--k", "1", "--steps", "120",
                     "--workers", WORKERS, "--var", f"r10b={candidate}",
                     "--var", f"r12c={baseline}", "--var", f"r9={r9}", "--out", roll)
        done = read_json(roll / "ROLLOUT_DONE.json")
        if (done.get("n_states"), done.get("k"), done.get("steps")) != (24, 1, 120):
            raise ValueError("invalid rollout completion receipt")
        brakes = hard_brakes(roll, ("r10b", "r12c", "r9"))
        reasons = stop_reasons(brakes["r10b"]["share"], brakes["r12c"]["share"],
                               mae["r10b"]["mae"], mae["r12c"]["mae"])
        return {"decision": "STOP" if reasons else "CONTINUE", "reasons": reasons,
                "labels": {"r10b": "R13C candidate", "r12c": "S-R12C", "r9": "as-trained round9"},
                "checkpoints": checkpoints, "provenance": provenance, "one_step": mae,
                "one_step_windows": read_json(os_dir / "ONESTEP_DONE.json")["windows"],
                "full_road_60s_hard_brakes": brakes,
                "BASE_verdict": base["verdict"], "legacy_midrun_context_only": legacy["decision"]}

    def execute(self):
        self.out.mkdir(parents=True, exist_ok=True)
        result = {"mode": self.mode, "pid": self.process.pid, "start_ticks": self.process.start_ticks,
                  "signal": "not requested", "intermediates": str(self.work)}
        # Claim per mode, but never hold the CPU lock while waiting for training:
        # postrun can be armed first without blocking the midrun stop check.
        claim = self.out / "CLAIM.json"
        with claim.open("x", encoding="utf-8") as stream:
            json.dump(result, stream)
        try:
            checkpoint = self.wait_checkpoint()
            import fcntl
            with (self.out.parent / "measurements.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result.update(self.measure(checkpoint))
        except Exception as exc:
            result.update(decision="STOP" if self.mode == "midrun" else "ERROR",
                          reasons=[f"measurement error (fail closed): {type(exc).__name__}: {exc}"])
        if self.mode == "midrun" and result["decision"] == "STOP":
            try:
                result["signal"] = self.process.terminate()
            except Exception as exc:
                result["signal"] = f"signal failed: {exc}"
        result["commands"] = self.commands
        (self.out / "RESULT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        lines = [f"# R13C {self.mode}: {result['decision']}", "",
                 "r10b = R13C; r12c = S-R12C; r9 = as-trained round9.",
                 "Stop: full-road 60s lane1 HDV a<-3 share >2x S-R12C OR one-step MAE >S-R12C+0.1.",
                 "Baseline is recomputed in the SAME invocation/panel. Legacy BASE/MIDRUN are context.",
                 f"PID {self.process.pid}, start ticks {self.process.start_ticks}; {result['signal']}.", ""]
        lines += [f"- {reason}" for reason in result["reasons"]]
        for key in ("one_step", "full_road_60s_hard_brakes", "BASE_verdict"):
            if key in result:
                lines.append(f"- {key}: {json.dumps(result[key])}")
        (self.out / "RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 0 if result["decision"] == "CONTINUE" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path("/root/autodl-tmp/c4/v5"))
    parser.add_argument("--pid", required=True, type=int, help="own production exec PID, never the smoke PID")
    parser.add_argument("--start-ticks", type=int, help="field 22 from /proc/PID/stat, captured before exec")
    parser.add_argument("--mode", required=True, choices=("midrun", "postrun"))
    args = parser.parse_args(argv)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    if hasattr(os, "nice"):
        os.nice(10)
    return Waiter(args).execute()


if __name__ == "__main__":
    raise SystemExit(main())

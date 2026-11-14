"""SUMO episode collection and crash-tolerant multiprocess workers for the C4 MAPPO baseline.

One SUMO environment per worker process (``concurrent.futures.ProcessPoolExecutor`` with
the ``spawn`` context: identical on Windows and Linux; the worker builds its env once in
the initializer).  Actor weights travel per job as state dicts.  A job that raises inside
the worker returns a failure record (the env is closed and rebuilt on the next job); a
worker process that dies (``BrokenProcessPool``) or an iteration that exceeds its deadline
kills only this pool's own worker processes, rebuilds the pool and reports the unfinished
jobs as failed.  The trainer retries each failed episode once with a new seed.

Team reward: ``wms.control.reward.team_reward`` per 0.5 s PHYSICS step, summed over the two
physics steps of each 1 s control step (exactly what the C4 imagination loop accumulates).
The intermediate (t + 0.5 s) SUMO frame comes from the env's rolling history; vehicle
identity across each physics step is checked by vehicle index, and a vehicle that leaves
the network (or whose slot is re-used) is carried for the reward at constant speed (the
reward contract requires identity-preserving transitions).  Envs without a history (the
fake) use the 1 s transition once.

Entry queue (SRCWAIT): an env whose ``step`` info carries ``entry_queue_steps`` = [q1, q2]
(time-averaged number of requested-but-not-inserted vehicles during each 0.5 s physics step,
``CavSumoEnv``) passes q_i as ``entry_queue`` to ``team_reward`` for physics transition i,
so ``source_wait`` = 0.5 * q_i is charged with the delay weight.  Single-transition fallback
(no history): ``entry_queue = q1 + q2`` so the 1 s transition's source wait equals
w1 + w2 (the sum over the two steps).  Envs without the field: old behaviour, exactly.
"""

from __future__ import annotations

import importlib
import os
import shutil
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor

from wms.control.contract import CONTROL_DT_S, PHYSICS_DT_S, RewardWeights
from wms.control.lanes import LaneRules
from wms.control.reward import team_reward

from .agent import STATE_KEYS, ActorStep, build_networks
from .buffer import STEP_KEYS, EpisodeData

FAKE_NAMES = {"fake": "wms.control.mappo.fakeenv.FakeCavEnv"}
STOP_POLL_S = 1.0


class StopRequested(Exception):
    """A stop file appeared (spot termination notice): abandon the in-flight iteration."""
COMPONENTS = ("delay", "throughput", "smooth", "hard_brake", "fairness_delay", "source_wait",
              "total_wait")


def resolve_name(name: str):
    """Dotted-name lazy import (``fake`` = the CPU fake env)."""
    name = FAKE_NAMES.get(name, name)
    module_name, _, attr = name.rpartition(".")
    if not module_name:
        raise ValueError(f"not a dotted name: {name!r}")
    return getattr(importlib.import_module(module_name), attr)


# ------------------------------------------------------------------------ physics reward
def _mid_state(env, state: Mapping[str, Tensor], next_state: Mapping[str, Tensor]):
    """(mid state, same-vehicle masks [2, K]) from the env history, or None."""
    history = getattr(env, "history", None)
    if history is None or not hasattr(history, "raw") or len(history) < 3:
        return None
    raw = history.raw(3)
    kin = torch.as_tensor(raw["kin"][1])
    present = torch.as_tensor(raw["mask"][1]).bool()
    vid = torch.as_tensor(raw["vid_index"].astype(np.int64))
    same = torch.as_tensor(raw["target_mask"]).bool()          # [2, K]: t->t+.5, t+.5->t+1
    onehot = kin[:, 1:4] > 0.5
    lane = torch.where(onehot.any(-1), onehot.float().argmax(-1), torch.full_like(vid[1], -1))
    is_cav = torch.where(same[0], state["is_cav"][0].bool(),
                         torch.where(same[1], next_state["is_cav"][0].bool(),
                                     torch.zeros_like(present)))
    dtype = state["s_m"].dtype
    mid = {"s_m": kin[:, 0].to(dtype)[None], "v_mps": kin[:, 4].to(dtype)[None],
           "a_mps2": kin[:, 5].to(dtype)[None], "lane": lane.long()[None],
           "present": present[None], "is_cav": (is_cav & present)[None],
           "length_m": state.get("length_m", kin[:, 6:7].T.to(dtype))}
    return mid, same


def _carried(before: Mapping[str, Tensor], after: Mapping[str, Tensor], same: Tensor):
    """``after`` with vehicles of ``before`` that are not the same vehicle in ``after``
    carried at constant speed (reward only); returns (patched, reused-slot mask)."""
    present = before["present"].bool()
    lost = present & ~same
    if not bool(lost.any()):
        return after, torch.zeros_like(lost)
    patched = dict(after)
    patched["s_m"] = torch.where(lost, before["s_m"] + before["v_mps"] * PHYSICS_DT_S, after["s_m"])
    patched["v_mps"] = torch.where(lost, before["v_mps"], after["v_mps"])
    patched["a_mps2"] = torch.where(lost, before["a_mps2"], after["a_mps2"])
    return patched, lost & after["present"].bool()


def _entry_queue_steps(env, info) -> list[float] | None:
    """[q1, q2] time-averaged entry queue of the control step's two physics steps, or None."""
    steps = None
    if isinstance(info, Mapping):
        steps = info.get("entry_queue_steps")
    if steps is None:
        steps = getattr(env, "last_entry_queue_steps", None)
    if steps is None or len(steps) == 0:
        return None
    return [float(q) for q in steps]


def control_step_reward(env, state, next_state, weights: RewardWeights,
                        info: Mapping | None = None) -> dict[str, float]:
    """Team reward components of one 1 s control step (sum over its physics steps).

    ``info`` (the env.step info) supplies ``entry_queue_steps`` (module docstring); without it
    (or from an env that has none) the reward is the old on-road-only one."""
    mid = _mid_state(env, state, next_state)
    queues = _entry_queue_steps(env, info)
    if mid is None:
        transitions = [(state, next_state, torch.ones_like(state["present"]).bool())]
        if queues is not None:
            queues = [sum(queues)]            # one 1 s transition charged w1 + w2
    else:
        mid_state, same = mid
        transitions = [(state, mid_state, same[0][None]), (mid_state, next_state, same[1][None])]
        if queues is not None and len(queues) == 1:
            queues = [queues[0], queues[0]]
    if queues is not None and len(queues) != len(transitions):
        queues = None
    total: dict[str, float] = {}
    for index, (before, after, same) in enumerate(transitions):
        patched, reused = _carried(before, after, same)
        accel = torch.where(before["present"].bool(), patched["a_mps2"],
                            torch.zeros_like(patched["a_mps2"]))
        queue = None if queues is None else torch.tensor(
            [queues[index]], dtype=before["s_m"].dtype, device=before["s_m"].device)
        out = team_reward(before, patched, accel, weights, entry_queue=queue)
        if bool(reused.any()):          # TTC exposure of the ACTUAL next occupants
            out["cost_ttc"] = team_reward(after, after, torch.zeros_like(accel), weights)["cost_ttc"]
        for key, value in out.items():
            total[key] = total.get(key, 0.0) + float(value.detach()[0])
    total["total_wait"] = total.get("delay", 0.0) + total.get("source_wait", 0.0)
    return total


# ------------------------------------------------------------------------ one episode
def collect_episode(env, actor: ActorStep, *, weights: RewardWeights, seed: int,
                    demand_vph: int, penetration: float, warmup_s: float, horizon_s: float,
                    noise_seed: int = 0) -> EpisodeData:
    """One stochastic team episode: ``int(horizon_s)`` decisions, each executed with
    ``env.step(command, lanes=lanes)`` exactly as the C4 deployment does."""
    started = time.monotonic()
    state = env.reset(seed=seed, demand_vph=demand_vph, penetration=penetration,
                      warmup_s=warmup_s, horizon_s=horizon_s)
    reset_wall = time.monotonic() - started
    actor.reset()
    generator = torch.Generator().manual_seed(int(noise_seed))
    n = int(round(horizon_s / CONTROL_DT_S))
    frames: dict[str, list[Tensor]] = {k: [] for k in STATE_KEYS}
    steps: dict[str, list[Tensor]] = {k: [] for k in STEP_KEYS}
    rewards, costs = [], []
    components: dict[str, list[float]] = {k: [] for k in COMPONENTS}
    lane_counts: dict[str, int] = {}
    lane_changes, decisions, terminated, info = 0, 0, False, {}
    policy_wall = 0.0
    for _ in range(n):
        tick = time.monotonic()
        out = actor(state, generator=generator)
        policy_wall += time.monotonic() - tick
        for key in STATE_KEYS:
            if key in out["state"]:
                frames[key].append(out["state"][key][0].detach().clone())
        for key in STEP_KEYS:
            steps[key].append(out[key][0].detach().clone())
        next_state, info = env.step(out["command"], lanes=out["lanes"])
        tick = time.monotonic()
        row = control_step_reward(env, state, next_state, weights, info)
        policy_wall += time.monotonic() - tick
        rewards.append(row["reward"])
        costs.append(row["cost_ttc"])
        for key in COMPONENTS:
            components[key].append(row.get(key, 0.0))
        for key, value in (info.get("lane_commands") or {}).items():
            lane_counts[key] = lane_counts.get(key, 0) + int(value)
        lane_changes += int(out["lane_changed"].sum())
        decisions += int(out["cav_mask"].sum())
        state = next_state
        terminated = bool(info.get("terminated", False))
        if terminated or info.get("done", False):
            break
    augmented = actor.memory.augment(state)
    for key in STATE_KEYS:
        if key in augmented and len(frames[key]) == len(rewards):
            frames[key].append(augmented[key][0].detach().clone())
    wall = time.monotonic() - started
    summary = info.get("summary") or {}
    meta = {"seed": seed, "demand_vph": demand_vph, "penetration": penetration,
            "warmup_s": warmup_s, "horizon_s": horizon_s, "noise_seed": noise_seed,
            "wall_s": wall, "reset_wall_s": reset_wall, "policy_wall_s": policy_wall,
            "sim_s": warmup_s + len(rewards) * CONTROL_DT_S,
            "sim_s_per_wall_s": (warmup_s + len(rewards) * CONTROL_DT_S) / wall if wall > 0 else None,
            "lane_commands": lane_counts, "policy_lane_changes": lane_changes,
            "cav_decisions": decisions,
            "collisions": summary.get("collisions"), "teleports": summary.get("teleports"),
            "max_present_vehicles": summary.get("max_present_vehicles"),
            "dropped_vehicle_steps": summary.get("dropped_vehicle_steps"), "pid": os.getpid()}
    keys = [k for k in STATE_KEYS if len(frames[k]) == len(rewards) + 1]
    return EpisodeData(
        states={k: torch.stack(frames[k]) for k in keys},
        steps={k: torch.stack(v) if v else torch.zeros(0) for k, v in steps.items()},
        rewards=torch.tensor(rewards, dtype=torch.float32),
        costs=torch.tensor(costs, dtype=torch.float32),
        terminated=terminated,
        components={k: torch.tensor(v, dtype=torch.float32) for k, v in components.items()},
        meta=meta)


# ------------------------------------------------------------------------ workers
_WORKER: dict = {}


def _scenario_dirs(root: Path) -> set[Path]:
    return set(root.iterdir()) if root.is_dir() else set()


def _remove_own_dirs(paths) -> None:
    """Delete scenario directories THIS worker's env created (plain dirs only, no links)."""
    for path in paths:
        try:
            if path.is_symlink() or not path.is_dir():
                continue
            if any(p.is_symlink() for p in path.rglob("*")):
                continue
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


def build_env(spec: Mapping):
    kwargs = dict(spec.get("env_kwargs", {}))
    root = spec.get("scenario_root")
    if root:
        root = Path(root) / f"pid{os.getpid()}"
        root.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("scenario_root", str(root))
    return resolve_name(spec["env_name"])(**kwargs)


def _worker_init(spec: dict) -> None:
    torch.set_num_threads(1)
    _WORKER.clear()
    _WORKER["spec"] = spec


def _worker_deps(spec: Mapping) -> dict:
    if "actor" not in _WORKER:
        policy, encoder, _ = build_networks()
        policy.eval()
        encoder.eval()
        _WORKER["actor"] = ActorStep(policy, encoder, LaneRules(**spec.get("lane_rules", {})))
        _WORKER["weights"] = RewardWeights(**spec.get("reward_weights", {}))
    if _WORKER.get("env") is None:
        _WORKER["env"] = build_env(spec)
    return _WORKER


def run_job(job: Mapping) -> dict:
    """One episode (worker process, or in-process for the serial path); never raises for
    episode-level errors: returns ``{"ok": False, "error": ...}`` instead."""
    spec = job.get("spec") or _WORKER.get("spec")
    if spec is None:
        raise RuntimeError("run_job needs a worker spec")
    if _WORKER.get("spec") != spec:          # serial path: a different trainer / config
        env = _WORKER.get("env")
        if env is not None and hasattr(env, "close"):
            env.close()
        _WORKER.clear()
        _WORKER["spec"] = spec
    env, before, scen_root = None, None, None
    params = {k: job[k] for k in ("seed", "demand_vph", "penetration", "warmup_s", "horizon_s")}
    try:
        deps = _worker_deps(spec)
        actor: ActorStep = deps["actor"]
        actor.policy.load_state_dict(job["policy_state"])
        actor.group_encoder.load_state_dict(job["encoder_state"])
        env = deps["env"]
        scen_root = getattr(env, "scenario_root", None)
        if scen_root and spec.get("cleanup_scenarios"):
            before = _scenario_dirs(Path(scen_root))
        episode = collect_episode(env, actor, weights=deps["weights"],
                                  noise_seed=job.get("noise_seed", 0), **params)
        return {"ok": True, "episode": episode, "job": dict(job_public(job))}
    except Exception as exc:  # noqa: BLE001 - episode-level failure is reported, not raised
        try:
            if env is not None:
                env.close()
        except Exception:  # noqa: BLE001
            pass
        _WORKER["env"] = None
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8), "job": dict(job_public(job)),
                "pid": os.getpid()}
    finally:
        if before is not None:
            try:
                if hasattr(env, "close") and not getattr(env, "done", True):
                    env.close()
            except Exception:  # noqa: BLE001
                pass
            _remove_own_dirs(_scenario_dirs(Path(scen_root)) - before)


def job_public(job: Mapping) -> dict:
    return {k: v for k, v in job.items() if k not in ("policy_state", "encoder_state", "spec")}


class WorkerPool:
    """``workers`` spawn processes, one SUMO env each; crash/timeout tolerant."""

    def __init__(self, workers: int, spec: dict):
        import multiprocessing

        self.workers, self.spec = int(workers), spec
        self.ctx = multiprocessing.get_context("spawn")
        self.executor = None
        self.restarts = 0

    def _start(self) -> None:
        if self.executor is None:
            self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=self.ctx,
                                                initializer=_worker_init, initargs=(self.spec,))

    def kill(self) -> None:
        """Stop this pool's OWN worker processes (never anything by image name)."""
        ex, self.executor = self.executor, None
        if ex is None:
            return
        procs = list(getattr(ex, "_processes", {}).values())
        for proc in procs:
            try:
                if proc.is_alive():
                    proc.terminate()
            except Exception:  # noqa: BLE001
                pass
        for proc in procs:
            try:
                proc.join(timeout=10)
                if proc.is_alive():
                    proc.kill()
            except Exception:  # noqa: BLE001
                pass
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001
            pass
        self.restarts += 1

    def run(self, jobs: list[dict], deadline_s: float | None = None,
            stop_check=None) -> list[dict]:
        """Results in job order; failed jobs are ``{"ok": False, ...}`` records.  ``stop_check()``
        is polled every ``STOP_POLL_S``: when true the workers are killed and
        :class:`StopRequested` is raised."""
        if not jobs:
            return []
        self._start()
        futures = {}
        try:
            for i, job in enumerate(jobs):
                futures[self.executor.submit(run_job, job)] = i
        except BrokenProcessPool:
            pass
        results: list[dict | None] = [None] * len(jobs)
        pending = set(futures)
        end = None if deadline_s is None else time.monotonic() + float(deadline_s)
        broken = False
        while pending:
            if stop_check is not None and stop_check():
                self.kill()
                raise StopRequested("stop requested while collecting")
            left = None if end is None else max(0.0, end - time.monotonic())
            timeout = STOP_POLL_S if left is None else min(left, STOP_POLL_S)
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_COMPLETED)
            if not done and (end is None or time.monotonic() < end):
                continue
            if not done:      # deadline reached
                broken = True
                for fut in pending:
                    results[futures[fut]] = {"ok": False, "error": "timeout",
                                             "job": job_public(jobs[futures[fut]])}
                break
            for fut in done:
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except BrokenProcessPool as exc:
                    broken = True
                    results[i] = {"ok": False, "error": f"worker died: {exc}",
                                  "job": job_public(jobs[i])}
                except Exception as exc:  # noqa: BLE001
                    results[i] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                  "job": job_public(jobs[i])}
        for i, row in enumerate(results):
            if row is None:
                broken = True
                results[i] = {"ok": False, "error": "not submitted (pool broken)",
                              "job": job_public(jobs[i])}
        if broken:
            self.kill()
        return results  # type: ignore[return-value]

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None

"""Undirected random CAV-action data for world-model training (GENRAND_NOTES.md).

Design rule: the world model learns action effects from UNDIRECTED data only -
native driving, generic random direct-output commands and the learned policy's own rollouts.
Nothing here is a designed controller: there is no spatial selection, no gap/merge targeting,
no damping/PI/wall behaviour.  Random STRUCTURE (hold durations, jerk, shared random latents
among random CAV subsets, random lane intents) is the only shaping.

:class:`GenericRandomPolicy` (``collect.py --policy generic_random``), per CAV:

* target acceleration ~ U[-3, 2.6] m/s^2 held for a log-uniform 2-60 s (whole control
  seconds), approached at a per-segment jerk ~ U[0.5, 3] m/s^3; a CAV's first command
  starts from its measured acceleration (continuity only).  Commands are NOT pre-clipped to
  feasibility: the env's speed-bound clip (``issued``) and SUMO's own safety logic
  (speedMode 27, ``realized``) override them, so requested != issued != realized is in the data.
* per episode: rho ~ U(0, 1) and 1-3 shared latent processes (each a random target U[-3, 2.6]
  held log-uniform 2-60 s).  At every latent hold boundary each present CAV independently
  (Bernoulli(rho)) starts a new segment synchronised with that latent: target =
  w * latent + (1 - w) * own U[-3, 2.6] draw, w ~ U(0, 1), hold = the latent's hold.  rho = 0
  gives independent CAVs; no position, lane or traffic state enters the selection.
* ``mode="speed"`` (DEFAULT, keeps states realistic while staying
  undirected): each hold draws a random TARGET SPEED instead of a target acceleration -
  65 % U[0.8, 1.15] x the speed limit (31.29 m/s), 25 % U[0.3, 0.8] x it, 10 % "free"
  (command +2.6; SUMO caps it at the allowed speed) - with an acceleration limit U[0.5, 2.6]
  and a deceleration limit U[1, 3] m/s^2; the command is (v_target - v) / 1 s clipped to the
  limits, approached at the segment jerk.  Targets scale with the speed LIMIT, not the local
  flow (a local-flow reference compounded: CAVs at 0.4x native speed in the first smoke).  A
  CAV below 1 m/s for more than 20 s ends its hold early (new draw).  Latents carry a speed
  factor (the non-free part of the mixture), mixed as above.  ``mode="accel"`` is the
  original target-acceleration process above (``--gr-mode accel``).
* random lane intents: per episode a Poisson rate lambda log-uniform in [1/600, 1/20] s^-1
  per CAV; an event picks left/right (+-1 WM lane) uniformly and keeps requesting it for a
  log-uniform 1-10 s or until the CAV is in that lane.  Every tick the policy returns a
  ``lanes [1,K]`` array (-1 = no request), so CAVs are under the env's policy lane control
  (laneChangeMode 513, safety-gapped ``changeLane``; impossible targets are refused there).

:func:`run_genrand_job` runs one base episode (recorded like every collector episode) and, at
the branch cadence, saves the SUMO state; afterwards every saved state is branched into
``M`` re-seeded random arms (``SEED:j``) + a native reference arm (``NATIVE``), each ``H`` s,
with common random numbers (fresh SUMO process + ``loadState``, the
:mod:`traffic_shepherd.c4.branching` primitives).  Each branch is written as a convertible
``c4_sumo_episode/1`` episode holding the base's 60 s pre-history + the branch, its world id
= the base world, a unique arm tag, the base-run entry times (so the converter's entry
counter is exact) and a pair key ``<world>@t<branch time>`` for a later paired loss.
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

ACCEL_LOW, ACCEL_HIGH = -3.0, 2.6
HOLD_S = (2.0, 60.0)
JERK = (0.5, 3.0)
LATENTS = (1, 3)
LANE_RATE = (1.0 / 600.0, 1.0 / 20.0)
LANE_HOLD_S = (1.0, 10.0)
MODES = ("speed", "accel")
SPEED_LIMIT_MPS = 31.29
SPEED_MIX = (0.65, 0.25, 0.1)            # cruise U[0.8,1.15] / slow U[0.3,0.8] / free
CRUISE_FACTOR, SLOW_FACTOR = (0.8, 1.15), (0.3, 0.8)
ACCEL_LIMIT = (0.5, 2.6)
DECEL_LIMIT = (1.0, 3.0)
STUCK_SPEED_MPS, STUCK_S = 1.0, 20
POLICY_TAG = "generic_random"
PREHISTORY_S = 60
SCHEMA_BRANCHES = "c4_genrand_branches/1"


def _loguniform(rng, lo: float, hi: float, size=None):
    return np.exp(rng.uniform(math.log(lo), math.log(hi), size))


def _hold_ticks(rng, lo: float, hi: float) -> int:
    return int(min(hi, max(lo, round(float(_loguniform(rng, lo, hi))))))


def _np(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(x)


def episode_policy_seed(seed: int, demand: int, penetration: float) -> int:
    """Episode seed of the generic random controller (one per collector case)."""
    return int(np.random.SeedSequence([int(seed), int(demand),
                                       int(round(float(penetration) * 1000)), 0x6E5A]
                                      ).generate_state(1)[0])


def arm_seed(policy_seed: int, t_s: int, j: int) -> int:
    """Seed of re-seeded branch arm ``SEED:j`` at decision time ``t_s``
    (:func:`traffic_shepherd.c4.branching.seed_for_arm`)."""
    from .branching import seed_for_arm
    return seed_for_arm(policy_seed, t_s, j)


class GenericRandomPolicy:
    """Undirected random per-CAV direct commands + random lane intents (module docstring).

    ``policy(state, with_lanes=True, vehicle_ids=None) -> (accel [1,K], lanes [1,K])``.  CAVs
    are tracked by vehicle id when ``vehicle_ids`` (the env's per-slot ids) is given, else by
    slot (a slot that newly holds a CAV starts a new track).  ``has_lane_head`` makes
    :func:`traffic_shepherd.c4.baselines.run_policy_episode` pass the lanes to the env.
    """

    has_lane_head = True

    def __init__(self, seed: int, *, low: float = ACCEL_LOW, high: float = ACCEL_HIGH,
                 hold_s=HOLD_S, jerk=JERK, n_latents=LATENTS, lane_rate=LANE_RATE,
                 lane_hold_s=LANE_HOLD_S, rho: Optional[float] = None,
                 lane_rate_per_s: Optional[float] = None, n_latent: Optional[int] = None,
                 mode: str = "speed"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.seed = int(seed)
        self.low, self.high = float(low), float(high)
        self.hold_s, self.jerk_range = tuple(hold_s), tuple(jerk)
        self.lane_hold_s = tuple(lane_hold_s)
        rng = np.random.default_rng([self.seed, 0])           # episode-level parameters
        self.rho = float(rng.uniform(0.0, 1.0)) if rho is None else float(rho)
        self.n_latent = (int(rng.integers(n_latents[0], n_latents[1] + 1)) if n_latent is None
                         else int(n_latent))
        self.lane_rate = (float(_loguniform(rng, *lane_rate)) if lane_rate_per_s is None
                          else float(lane_rate_per_s))
        self.reset()

    # ----------------------------------------------------------------- params / state
    def params(self) -> dict:
        return {"policy": POLICY_TAG, "mode": self.mode, "seed": self.seed, "rho": self.rho,
                "n_latent": self.n_latent, "lane_rate_per_s": self.lane_rate,
                "accel_range_mps2": [self.low, self.high], "hold_s": list(self.hold_s),
                "jerk_mps3": list(self.jerk_range), "lane_hold_s": list(self.lane_hold_s),
                **({"speed_mix": list(SPEED_MIX), "accel_limit_mps2": list(ACCEL_LIMIT),
                    "decel_limit_mps2": list(DECEL_LIMIT), "speed_ref": "limit",
                    "cruise_factor": list(CRUISE_FACTOR), "slow_factor": list(SLOW_FACTOR),
                    "stuck": [STUCK_SPEED_MPS, STUCK_S]} if self.mode == "speed" else {})}

    def reset(self) -> None:
        self.rng = np.random.default_rng([self.seed, 1])
        self.tracks: dict = {}          # key -> {cmd, target, jerk, left, lane_target, lane_left}
        self.latents = [self._new_latent() for _ in range(self.n_latent)]
        self.events = {"segments": 0, "latent_joins": 0, "lane_intents": 0}
        self._fresh_latent = [True] * self.n_latent

    def reseed(self, seed: int) -> "GenericRandomPolicy":
        """Branch arm ``SEED:j``: new random stream from this state.  Episode parameters (rho,
        latent count, lane rate) and every CAV's current command level are kept; every CAV and
        latent starts a new segment at the next tick and pending lane intents are dropped."""
        self.rng = np.random.default_rng([int(seed), 2])
        for track in self.tracks.values():
            track["left"] = 0
            track["lane_target"], track["lane_left"] = -1, 0
        self.latents = [self._new_latent() for _ in range(self.n_latent)]
        self._fresh_latent = [True] * self.n_latent
        self.reseeded_with = int(seed)
        return self

    def _new_latent(self) -> dict:
        value = (self._speed_factor(free_ok=False)[0] if self.mode == "speed"
                 else float(self.rng.uniform(self.low, self.high)))
        return {"value": value, "left": _hold_ticks(self.rng, *self.hold_s)}

    def _speed_factor(self, free_ok: bool = True) -> tuple:
        """(factor of the speed limit, free) from the speed-mode mixture."""
        u = float(self.rng.uniform(0.0, 1.0))
        p_scaled, p_slow, _ = SPEED_MIX
        if not free_ok:
            u *= p_scaled + p_slow
        if u < p_scaled:
            return float(self.rng.uniform(*CRUISE_FACTOR)), False
        if u < p_scaled + p_slow:
            return float(self.rng.uniform(*SLOW_FACTOR)), False
        return 1.0, True

    def _speed_segment(self, track: dict, factor: float, free: bool, hold: int) -> None:
        track["free"] = bool(free)
        track["v_target"] = SPEED_LIMIT_MPS * (1.0 if free else float(factor))
        track["alim"] = float(self.rng.uniform(*ACCEL_LIMIT))
        track["dlim"] = float(self.rng.uniform(*DECEL_LIMIT))
        track["slow"] = 0
        self._new_segment(track, 0.0, hold)

    def _new_segment(self, track: dict, target: float, hold: int) -> None:
        track["target"] = float(target)
        track["left"] = int(hold)
        track["jerk"] = float(self.rng.uniform(*self.jerk_range))
        self.events["segments"] += 1

    # ------------------------------------------------------------------------ act
    def __call__(self, state, with_lanes: bool = False, vehicle_ids=None):
        cav = (np.asarray(_np(state["is_cav"])[0], bool)
               & np.asarray(_np(state["present"])[0], bool))
        k = cav.shape[0]
        a_now = np.asarray(_np(state["a_mps2"])[0], np.float64)
        lane_now = np.asarray(_np(state["lane"])[0], np.int64)
        speed_mode = self.mode == "speed"
        if speed_mode:
            v_now = np.asarray(_np(state["v_mps"])[0], np.float64)
        slots = np.flatnonzero(cav).tolist()
        # Track key: vehicle id (exact) or slot (a slot that was empty at the previous call
        # starts a new track; same-tick slot reuse is only detectable with vehicle ids).
        keys = ({s: str(vehicle_ids[s]) for s in slots} if vehicle_ids is not None
                else {s: s for s in slots})
        live = set(keys.values())
        for key in [key for key in self.tracks if key not in live]:
            self.tracks.pop(key)
        order = sorted(slots, key=lambda s: str(keys[s]))
        rng = self.rng
        # 1. latent processes (boundary -> Bernoulli(rho) synchronised subsets)
        joins: dict = {}
        for li, lat in enumerate(self.latents):
            boundary = self._fresh_latent[li] or lat["left"] <= 0
            if not self._fresh_latent[li] and lat["left"] <= 0:
                lat.update(self._new_latent())
            self._fresh_latent[li] = False
            if boundary:
                for s in order:
                    if rng.random() < self.rho:
                        joins[keys[s]] = (lat["value"], lat["left"], float(rng.uniform(0.0, 1.0)))
            lat["left"] -= 1
        # 2. per-CAV segments
        for s in order:
            key = keys[s]
            track = self.tracks.get(key)
            if track is None:
                start = float(np.clip(a_now[s], self.low, self.high))
                track = self.tracks[key] = {"cmd": start, "target": start, "jerk": 1.0,
                                            "left": 0, "lane_target": -1, "lane_left": 0}
            if speed_mode:
                track["slow"] = track.get("slow", 0) + 1 if v_now[s] < STUCK_SPEED_MPS else 0
                if track["slow"] > STUCK_S and key not in joins:
                    track["left"] = 0                  # stuck: end the hold early
                    self.events["stuck_redraws"] = self.events.get("stuck_redraws", 0) + 1
            if key in joins:
                value, hold, w = joins[key]
                if speed_mode:
                    own, _ = self._speed_factor(free_ok=False)
                    self._speed_segment(track, w * value + (1.0 - w) * own, False,
                                        max(1, hold))
                else:
                    own = float(rng.uniform(self.low, self.high))
                    self._new_segment(track, w * value + (1.0 - w) * own, max(1, hold))
                self.events["latent_joins"] += 1
            elif track["left"] <= 0:
                if speed_mode:
                    factor, free = self._speed_factor()
                    self._speed_segment(track, factor, free, _hold_ticks(rng, *self.hold_s))
                else:
                    self._new_segment(track, float(rng.uniform(self.low, self.high)),
                                      _hold_ticks(rng, *self.hold_s))
            if speed_mode:              # desired accel toward the held target speed
                track["target"] = (self.high if track["free"] else float(np.clip(
                    track["v_target"] - v_now[s], -track["dlim"], track["alim"])))
            step = float(np.clip(track["target"] - track["cmd"], -track["jerk"], track["jerk"]))
            track["cmd"] += step
            track["left"] -= 1
        # 3. lane intents (Poisson per CAV, left/right uniformly)
        p_event = 1.0 - math.exp(-self.lane_rate * 1.0)
        lanes = np.full(k, -1, np.int64)
        for s in order:
            track = self.tracks[keys[s]]
            cur = int(lane_now[s])
            if track["lane_target"] >= 0 and (cur == track["lane_target"] or track["lane_left"] <= 0):
                track["lane_target"], track["lane_left"] = -1, 0
            if rng.random() < p_event and cur >= 0:
                track["lane_target"] = cur + (1 if rng.random() < 0.5 else -1)
                track["lane_left"] = _hold_ticks(rng, *self.lane_hold_s)
                self.events["lane_intents"] += 1
                if track["lane_target"] < 0:          # off the road edge: no request
                    track["lane_target"], track["lane_left"] = -1, 0
            if track["lane_target"] >= 0:
                lanes[s] = track["lane_target"]
                track["lane_left"] -= 1
        out = np.full(k, np.nan)
        for s in slots:
            out[s] = self.tracks[keys[s]]["cmd"]
        if with_lanes:
            return out[None], lanes[None]
        return out[None]


# ====================================================================== branch episodes
class MemoryRecorder:
    """In-memory stand-in for :class:`EpisodeRecorder` during a branch (no files)."""

    def __init__(self, k: int):
        self.k = int(k)
        self.frames: list = []
        self.controls: list = []
        self.lane_controls: list = []
        self.lane_requests: list = []

    def on_frame(self, frame, row=None) -> None:
        self.frames.append(frame)

    def on_control(self, time_s, frame, decisions, vehicle_ids, lane_commands=None,
                   lane_requests=None) -> None:
        from .recorder import control_row
        ctrl, acc, tgt, req, lane_target, lane_req = control_row(
            self.k, decisions, lane_commands, lane_requests)
        self.controls.append((float(time_s), ctrl, acc, tgt, req))
        self.lane_controls.append(lane_target)
        self.lane_requests.append(lane_req)

    def finish(self, *args, **kwargs):
        return None

    def fail(self, error: str) -> None:
        self.error = error


def first_seen(frames, vehicle_ids) -> tuple[np.ndarray, np.ndarray]:
    """``(entry_first_seen_s [V], entry_first_entrance [V])`` over ``frames``: the first
    occupied tick of every registry vehicle and its entrance (1 = ramp lane, 0 mainline;
    -1 / NaN = never seen) - the converter's entry-counter convention."""
    n = len(vehicle_ids)
    seen = np.full(n, np.nan)
    ent = np.full(n, -1, np.int8)
    for f in frames:
        idx = f.vid_index[f.present]
        lanes = f.lane_carry[f.present]
        for v, ln in zip(idx.tolist(), lanes.tolist()):
            if 0 <= v < n and math.isnan(seen[v]):
                seen[v] = float(f.time_s)
                ent[v] = 1 if ln == 2 else 0
    return seen, ent


def write_branch_episode(out_dir: Path, *, base_recorder, i_k: int, branch: MemoryRecorder,
                         env, meta: dict, summary: dict, prehistory_s: int = PREHISTORY_S) -> dict:
    """Base pre-history (``prehistory_s`` before the branch, from base frame index ``i_k``) +
    the branch frames as a ``c4_sumo_episode/1`` directory (manifest/summary/episode.npz)."""
    from .recorder import build_episode_arrays, write_json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    lo = i_k - int(round(prehistory_s / 0.5))
    if lo < 0:
        raise ValueError("branch time leaves less than the pre-history in the base episode")
    t_k = float(base_recorder.frames[i_k].time_s)
    frames = list(base_recorder.frames[lo:i_k + 1]) + list(branch.frames)
    keep = [c for c, row in enumerate(base_recorder.controls)
            if t_k - prehistory_s - 1e-6 <= row[0] < t_k - 1e-6]
    controls = [base_recorder.controls[c] for c in keep] + branch.controls
    lane_controls = [base_recorder.lane_controls[c] for c in keep] + branch.lane_controls
    lane_requests = [base_recorder.lane_requests[c] for c in keep] + branch.lane_requests
    vids = list(env.builder.vehicle_ids)
    seen, ent = first_seen(list(base_recorder.frames[:i_k + 1]) + list(branch.frames), vids)
    arrays = build_episode_arrays(frames, controls, lane_controls, lane_requests, k=branch.k,
                                  vehicle_ids=vids, cav_ids=env.cav_ids, meta=meta)
    arrays["entry_first_seen_s"] = seen
    arrays["entry_first_entrance"] = ent
    np.savez_compressed(out_dir / "episode.npz", **arrays)
    write_json(out_dir / "summary.json", summary)
    manifest = {"schema": "c4_sumo_episode/1", "status": "complete", **meta}
    write_json(out_dir / "manifest.json", manifest)
    return {"dir": str(out_dir), "n_states": len(frames), "n_control": len(controls)}


def _snapshot(env) -> dict:
    from .branching import snapshot_env
    rec = env.recorder
    env.recorder = None
    try:
        return snapshot_env(env)
    finally:
        env.recorder = rec


def _step(env, policy, state):
    accel, lanes = policy(state, with_lanes=True, vehicle_ids=env.slot_vehicle_ids())
    return env.step(accel, lanes=lanes)


def branch_times(warmup_s: int, horizon_s: int, cadence_s: int, branch_h: int,
                 first_s: Optional[int] = None) -> list[int]:
    """Decision times ``first + k * cadence`` with a full branch inside the episode and a full
    pre-history after the warmup start (default first = warmup + cadence)."""
    if cadence_s <= 0:
        return []
    first = warmup_s + cadence_s if first_s is None else int(first_s)
    end = warmup_s + horizon_s
    first = max(first, PREHISTORY_S)
    return list(range(first, end - branch_h + 1, cadence_s))


def run_genrand_job(env, *, seed: int, demand_vph: int, penetration: float, warmup_s: int,
                    horizon_s: int, episode_dir: Path, branch_root: Optional[Path],
                    cadence_s: int = 60, n_arms: int = 4, branch_h: int = 60,
                    native_arm: bool = True, repeat_check: int = 0, meta: Optional[dict] = None,
                    first_s: Optional[int] = None, mode: str = "speed") -> dict:
    """One base episode with :class:`GenericRandomPolicy` + paired branches (module docstring).

    ``env`` must be a direct-mode :class:`CavSumoEnv` launched with
    ``branching.SAVE_STATE_CLI``.  ``repeat_check`` > 0 re-runs arm ``SEED:0`` of the first
    ``repeat_check`` decision points and records whether it reproduced exactly (determinism /
    common-random-number check).  Returns the base summary plus a ``branches`` block."""
    from .branching import (load_state, plan_seed_arms, prepare_branch_mode, reseeded,
                            save_state, trajectory_diff)

    episode_dir, policy_seed = Path(episode_dir), episode_policy_seed(seed, demand_vph, penetration)
    policy = GenericRandomPolicy(policy_seed, mode=mode)
    times = branch_times(warmup_s, horizon_s, cadence_s, branch_h, first_s) if branch_root else []
    meta = dict(meta or {}, policy=POLICY_TAG, genrand=policy.params())
    t_start = time.monotonic()
    state = env.reset(seed=seed, demand_vph=demand_vph, penetration=penetration,
                      warmup_s=warmup_s, horizon_s=horizon_s, controller="direct",
                      episode_dir=episode_dir, meta=meta)
    base_rec = env.recorder
    state_dir = None if branch_root is None else Path(branch_root) / "_states"
    saved = []
    while True:
        now = int(round(env.now))
        if times and now in times:
            state_dir.mkdir(parents=True, exist_ok=True)
            path = state_dir / f"t{now:04d}.xml"
            save_state(env, path)
            saved.append({"t_s": now, "path": path, "snap": _snapshot(env),
                          "policy": copy.deepcopy(policy), "i_k": len(base_rec.frames) - 1})
        state, info = _step(env, policy, state)
        if info["done"]:
            base_summary = info["summary"]
            break
    base_wall = time.monotonic() - t_start
    world_meta = {k: meta[k] for k in meta if k not in ("genrand",)}
    branches, checks = [], []
    t_branch = time.monotonic()
    if saved and env.api is None:          # the finished base episode closed TraCI
        import traci
        env.api = traci
    for p_i, point in enumerate(saved):
        t_k = point["t_s"]
        arms = plan_seed_arms(n_arms, native=native_arm)   # SEED:j ... + B (all native)
        if p_i < repeat_check and n_arms > 0:
            arms.append(dict(arms[0], repeat=True))
        trajs = {}
        for arm in arms:
            load_state(env, point["path"], point["snap"])
            prepare_branch_mode(env)
            env.end_s = max(env.end_s, t_k + branch_h + 1)  # a branch never closes the episode
            rec = MemoryRecorder(env.k)
            env.recorder = rec
            if arm["kind"] == "SEED":
                fn = reseeded(point["policy"], arm_seed(policy_seed, t_k, arm["j"]))
            else:
                fn = None
            st = env.state()
            totals = {"collisions": 0, "teleports": 0, "lane_issued": 0, "lane_completed": 0}
            traj = []
            w0 = time.monotonic()
            for _ in range(int(branch_h)):
                if fn is None:
                    st, info = env.step(None)
                else:
                    st, info = _step(env, fn, st)
                iv = info["interval"]
                totals["collisions"] += int(iv.get("collision_involvements", 0))
                totals["teleports"] += int(iv.get("starting_teleports", 0))
                totals["lane_issued"] += int(info["lane_commands"].get("issued", 0))
                totals["lane_completed"] += int(info["lane_commands"].get("completed", 0))
                f = env.frame
                ids = env.builder.vehicle_ids
                traj.append((round(f.time_s, 3), {ids[int(f.vid_index[s])]: (float(f.s[s]), float(f.v[s]))
                                                  for s in np.flatnonzero(f.present).tolist()}))
            env.recorder = None
            if arm.get("repeat"):
                checks.append({"t_s": t_k, **trajectory_diff(trajs["SEED:0"], traj)})
                continue
            trajs[arm["name"]] = traj
            tag = "native" if arm["kind"] == "B" else f"seed{arm['j']}"
            arm_tag = f"{POLICY_TAG}_b{t_k:04d}_{tag}"
            world = _world(seed, demand_vph, penetration, warmup_s)
            pair_key = f"{world}@t{t_k}"
            bmeta = dict(world_meta, seed=seed, demand_vph=demand_vph, penetration=penetration,
                         warmup_s=float(t_k), horizon_s=float(branch_h), world_t0_s=float(warmup_s),
                         controller="direct", k_slots=env.k, dt_s=0.5, control_dt_s=1.0,
                         arm=arm_tag, policy=arm_tag,
                         branch={"pair_key": pair_key, "base_world": world,
                                 "base_episode": str(episode_dir), "branch_t_s": t_k,
                                 "arm": arm["name"], "kind": arm["kind"],
                                 "arm_seed": (None if fn is None else fn.reseeded_with),
                                 "horizon_s": branch_h, "prehistory_s": PREHISTORY_S,
                                 "policy_params": None if fn is None else fn.params()})
            bsum = {"schema": "c4_genrand_branch_summary/1", **totals,
                    "wall_s": time.monotonic() - w0, "sim_s": branch_h}
            out = write_branch_episode(Path(branch_root) / f"t{t_k:04d}_{tag}",
                                       base_recorder=base_rec, i_k=point["i_k"], branch=rec,
                                       env=env, meta=bmeta, summary=bsum)
            branches.append({"t_s": t_k, "arm": arm["name"], "pair_key": pair_key, **totals,
                             "wall_s": bsum["wall_s"], **out})
        point["path"].unlink(missing_ok=True)      # own scratch state file
    env.close()
    if state_dir is not None and state_dir.exists() and not any(state_dir.iterdir()):
        state_dir.rmdir()
    block = {"schema": SCHEMA_BRANCHES, "n_points": len(saved), "decision_times_s": times,
             "n_branches": len(branches), "arms_per_point": n_arms + int(native_arm),
             "branch_horizon_s": branch_h, "collisions": sum(b["collisions"] for b in branches),
             "teleports": sum(b["teleports"] for b in branches),
             "repeat_checks": checks, "base_wall_s": base_wall,
             "branch_wall_s": time.monotonic() - t_branch, "branches": branches,
             "policy": policy.params(), "policy_events": dict(policy.events)}
    if branch_root is not None:
        Path(branch_root).mkdir(parents=True, exist_ok=True)
        from .recorder import write_json
        write_json(Path(branch_root) / "BRANCHES_COMPLETE.json", _jsonable(block))
    return {"summary": base_summary, "branches": block}


def _world(seed, demand, penetration, t0) -> str:
    pct = float(penetration) * 100.0
    return f"w{int(seed)}m{int(round(float(demand)))}p{int(round(pct)):02d}t{int(round(t0))}"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dumps(value) -> str:
    return json.dumps(_jsonable(value), allow_nan=False)

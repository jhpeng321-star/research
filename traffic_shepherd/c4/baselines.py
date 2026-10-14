"""Policy runner, baseline controllers, random direct-output policy, wall law, loaders.

``run_policy_episode(env, policy, ...)`` drives one episode through the same harness for
every controller: a callable ``policy_fn(state_dict) -> accel [1,K]`` (learned policy,
random policy) runs the ``direct`` controller; the strings ``native`` / ``damping`` /
``pi`` / ``wall`` select the frozen baselines inside :class:`CavSumoEnv`.
"""
from __future__ import annotations

import importlib
import math
from typing import Callable, Optional

import numpy as np

ACCEL_MIN, ACCEL_MAX = -3.0, 1.5
BASELINES = ("native", "damping", "pi", "wall")


# ------------------------------------------------------------------ runner
def run_policy_episode(env, policy, *, seed: int, demand_vph: int, penetration: float,
                       warmup_s: float, horizon_s: float, episode_dir=None,
                       meta: Optional[dict] = None,
                       on_step: Optional[Callable] = None) -> dict:
    """Run one episode; returns the SUMO summary (also written by the recorder)."""
    if isinstance(policy, str):
        if policy not in BASELINES:
            raise ValueError(f"baseline must be one of {BASELINES}")
        controller, policy_fn = policy, None
    elif callable(policy):
        controller, policy_fn = "direct", policy
    else:
        raise TypeError("policy must be a baseline name or a callable state -> accel")
    if hasattr(policy_fn, "reset"):
        policy_fn.reset()
    state = env.reset(seed=seed, demand_vph=demand_vph, penetration=penetration,
                      warmup_s=warmup_s, horizon_s=horizon_s, controller=controller,
                      episode_dir=episode_dir, meta=meta)
    # Learned policies with a lane head (wms.control.policy.PolicyExecutor) also return per-CAV
    # lane targets, executed by the env as lane commands; accel-only callables are unchanged.
    with_lanes = bool(getattr(policy_fn, "has_lane_head", False))
    while True:
        if with_lanes:
            accel, lanes = policy_fn(state, with_lanes=True)
            state, info = env.step(accel, lanes=lanes)
        else:
            accel = None if policy_fn is None else policy_fn(state)
            state, info = env.step(accel)
        if on_step is not None:
            on_step(state, info)
        if info["done"]:
            return info["summary"]


# --------------------------------------------------------- random direct policy
class RandomAccelPolicy:
    """Seeded smooth random per-CAV accelerations (exploration / WM training data).

    Per slot: a target acceleration drawn U[-2.5, 1.5] m/s^2 and held U[2, 8] s, reached
    with a 1 m/s^3 jerk limit; a slot that newly holds a CAV restarts at 0.
    """

    def __init__(self, seed: int, *, k: Optional[int] = None, jerk_mps3: float = 1.0,
                 low: float = -2.5, high: float = 1.5, hold_s=(2, 8)):
        self.seed, self.k = int(seed), k
        self.jerk, self.low, self.high, self.hold = float(jerk_mps3), low, high, hold_s
        self.reset()

    def reset(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._accel = self._target = self._left = self._prev = None

    def __call__(self, state) -> np.ndarray:
        cav = np.asarray(_np(state["is_cav"])[0], bool) & np.asarray(_np(state["present"])[0], bool)
        k = cav.shape[0]
        if self._accel is None:
            self._accel = np.zeros(k)
            self._target = np.zeros(k)
            self._left = np.zeros(k, np.int64)
            self._prev = np.zeros(k, bool)
        new = cav & ~self._prev
        self._accel[new] = 0.0
        self._left[new] = 0
        draw = cav & (self._left <= 0)
        n = int(draw.sum())
        if n:
            self._target[draw] = self.rng.uniform(self.low, self.high, n)
            self._left[draw] = self.rng.integers(self.hold[0], self.hold[1] + 1, n)
        self._accel = np.where(cav, self._accel + np.clip(self._target - self._accel,
                                                          -self.jerk, self.jerk), 0.0)
        self._left -= 1
        self._prev = cav
        out = np.full(k, np.nan)
        out[cav] = np.clip(self._accel[cav], ACCEL_MIN, ACCEL_MAX)
        return out[None]


class ConstantAccelPolicy:
    """Constant direct command to every live CAV.  ``lane_mode_513=True`` adds a lane head that
    always answers -1 ("no change"), which puts the CAVs in laneChangeMode 513 exactly like a
    learned lane-head policy (MAPPO audit 2026-10-14: the trivial "max accel + LC513" arm)."""

    def __init__(self, value: float = 0.0, lane_mode_513: bool = False):
        self.value = float(value)
        self.has_lane_head = bool(lane_mode_513)

    def __call__(self, state, with_lanes=False):
        cav = _np(state["is_cav"])[0].astype(bool) & _np(state["present"])[0].astype(bool)
        out = np.full(cav.shape[0], np.nan)
        out[cav] = self.value
        if with_lanes:
            return out[None], np.full((1, cav.shape[0]), -1)
        return out[None]


def _np(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(x)


# ----------------------------------------------------- learned policy (contract)
class ContractPolicyAdapter:
    """Compose the ``wms.control.contract`` pieces into ``state -> accel [1,K]``.

    ``obs_builder(state)`` -> Observation; ``rsu_field(state)`` -> [1,N,F];
    ``group_encoder(obs, state["s_m"], rsu)`` -> [1,K,G]; ``policy(obs, group,
    deterministic=True)`` -> (accel, logp); ``safety(state, accel)`` -> accel.  Non-CAV
    slots are returned as NaN (released / ignored by the environment).
    """

    def __init__(self, policy, obs_builder, *, group_encoder=None, rsu_field=None,
                 safety=None, deterministic: bool = True):
        self.policy, self.obs_builder = policy, obs_builder
        self.group_encoder, self.rsu_field, self.safety = group_encoder, rsu_field, safety
        self.deterministic = deterministic

    def __call__(self, state):
        import torch

        from .wms_glue import contract
        with torch.no_grad():
            obs = self.obs_builder(state)
            if self.group_encoder is not None:
                rsu = self.rsu_field(state) if self.rsu_field is not None else None
                group = self.group_encoder(obs, state["s_m"], rsu)
            else:
                b, k = state["s_m"].shape
                group = torch.zeros(b, k, contract().GROUP_SIGNAL_DIM)
            accel, _ = self.policy(obs, group, deterministic=self.deterministic)
            if self.safety is not None:
                # Bound by the policy's own stored cap when both sides support it (ACCELCAP_NOTES.md).
                bound = getattr(self.policy, "accel_max", None)
                from .wms_glue import ensure_wms_importable
                ensure_wms_importable()
                accepts_accel_max = importlib.import_module("wms.control.safety").accepts_accel_max
                if bound is not None and accepts_accel_max(self.safety):
                    accel = self.safety(state, accel, accel_max=float(bound))
                else:
                    accel = self.safety(state, accel)
        out = accel.detach().cpu().numpy().astype(np.float64)
        out[~obs.cav_mask.cpu().numpy()] = np.nan
        return out


def make_policy(spec: str, *, seed: int = 0, loader: str = "wms.control.policy:load_policy_fn"):
    """``spec`` -> baseline name or ``state -> accel`` callable.

    * ``native|damping|pi|wall`` -> the name (frozen controllers inside the env);
    * ``random`` -> :class:`RandomAccelPolicy` (seeded per episode by the caller);
    * ``zero`` -> constant 0 m/s^2 direct commands;
    * ``module:attr`` -> imported zero-argument factory returning a callable;
    * a checkpoint path -> ``loader(path)`` (default ``wms.control.policy:load_policy_fn``,
      to be provided by the policy workstream; it must return ``state -> accel [1,K]``).
    """
    if spec in BASELINES:
        return spec
    if spec == "random":
        return RandomAccelPolicy(seed)
    if spec == "zero":
        return ConstantAccelPolicy(0.0)
    if spec in ("maxaccel_lc513", "maxaccel_sumolc"):
        # trivial reference arms: ask every CAV for the 2.6 m/s^2 cap (SUMO limits it to normal
        # driving), with SUMO lane changing off (LC513, as under a lane-head policy) or on.
        return ConstantAccelPolicy(2.6, lane_mode_513=spec == "maxaccel_lc513")
    from .wms_glue import ensure_wms_importable
    ensure_wms_importable()
    if ":" in spec and not _looks_like_path(spec):
        module, attr = spec.split(":", 1)
        return getattr(importlib.import_module(module), attr)()
    module, attr = loader.split(":", 1)
    return getattr(importlib.import_module(module), attr)(spec)


def _looks_like_path(spec: str) -> bool:
    return len(spec) > 2 and spec[1] == ":" and spec[2] in "/\\" or spec.endswith((".pt", ".pth"))


# ------------------------------------------------------------------ wall law
class WallLaw:
    """Wall baseline: ``wms.physics.wall_exec.NumpyWallExecutor`` (law v2.2 merge-release).

    Formation rule (this harness, documented in SUMO_NOTES.md): when no wall is active and
    the cooldown has passed, pick one CAV per mainline lane (lanes 0 and 1) with s in
    [-2000, -300] m whose longitudinal offset is <= 75 m (closest pair); wall speed =
    clip(min member speed - 3, 6, 25) m/s.  Members are owned exclusively by the wall until
    handback (merge release at s = 0, abort or 60 s assembly timeout); every other CAV runs
    CIRCLES damping.  Positions are passed to the executor as ``s + 2644.2`` so its v2.2
    release point (2644.2 m) is the merge start.
    """

    S_RANGE = (-2000.0, -300.0)
    PAIR_MAX_DS = 75.0
    COOLDOWN_S = 30.0

    def __init__(self, k: int):
        self.k = k
        self.executor = None
        self.members: dict[int, int] = {}       # slot -> vehicle registry index
        self.cooldown_until = -math.inf
        self.events: list = []
        self.n_walls = 0

    def _form(self, frame, now_s):
        from .wms_glue import wall_executor

        ok = (frame.present & frame.is_cav & ~frame.junction
              & (frame.s >= self.S_RANGE[0]) & (frame.s <= self.S_RANGE[1]))
        lane0 = [i for i in np.flatnonzero(ok & (frame.lane == 0)).tolist()]
        lane1 = [i for i in np.flatnonzero(ok & (frame.lane == 1)).tolist()]
        best = None
        for i in lane0:
            for j in lane1:
                ds = abs(float(frame.s[i] - frame.s[j]))
                if ds <= self.PAIR_MAX_DS and (best is None or ds < best[0]):
                    best = (ds, i, j)
        if best is None:
            return
        _, i, j = best
        mask = np.zeros((1, self.k), bool)
        mask[0, [i, j]] = True
        speed = float(np.clip(min(frame.v[i], frame.v[j]) - 3.0, 6.0, 25.0))
        self.executor = wall_executor(mask, [speed], prior_controller=np.ones((1, self.k)))
        self.members = {i: int(frame.vid_index[i]), j: int(frame.vid_index[j])}
        self.n_walls += 1
        self.events.append({"event": "formed", "timestamp_s": now_s, "speed_mps": speed,
                            "slots": [i, j]})

    def commands(self, frame, builder, *, now_s: float, live_cavs) -> dict:
        if self.executor is None and now_s >= self.cooldown_until:
            self._form(frame, now_s)
        if self.executor is None:
            return {}
        present = frame.present.copy()
        for slot, idx in self.members.items():
            if int(frame.vid_index[slot]) != idx:
                present[slot] = False
        lane = np.where(frame.junction, frame.lane_carry, frame.lane).astype(float)
        result = self.executor.step((frame.s + 2644.2)[None].astype(float),
                                    frame.v[None].astype(float), now_s=now_s,
                                    lane_index=lane[None], present=present[None])
        self.events.extend(dict(e, timestamp_s=now_s) for e in result["events"])
        if result["phase"][0] == 2:              # handback
            self.executor, self.members = None, {}
            self.cooldown_until = now_s + self.COOLDOWN_S
            return {}
        out = {}
        for slot, idx in self.members.items():
            vid = builder.vehicle_ids[idx]
            if vid in live_cavs and present[slot]:
                out[vid] = float(result["acceleration_mps2"][0, slot])
        return out

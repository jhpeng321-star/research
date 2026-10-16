"""SUMO saved-state branching: measure CAV cooperation in the real simulator.

    python -m traffic_shepherd.c4.branching --controller damping --out runs/active/C4-BRANCH-x \\
        [--seeds 4107 --demand 2700 --penetration 0.2 --warmup 300 --points 2 \\
         --spacing 60 --horizon 30]

Design (docs/WM_C4_DESIGN.md section 6, "cooperation as the main training signal"), identical
for every controller (our C4 policy, MAPPO, CIRCLES damping, PI, random, ...):

1. One base episode runs the controller under test on every CAV (native warmup first).  At
   each decision time ``t_k`` the full SUMO state is saved (``simulation.saveState`` with
   ``--save-state.precision 17 --save-state.rng true``, the planner/closed_loop.py convention)
   together with a deep copy of the environment's Python state (slot pool, history buffer,
   request ledger, command ownership, controller sessions) and of the controller object.
2. From every saved state the branches below run ``H`` seconds (default 30 s) with the SAME
   SUMO RNG state (common random numbers):

   * ``B``         all CAVs native (reference), run twice (``B``, ``B_repeat``) -> determinism;
   * ``ALL``       every CAV follows the controller;
   * ``ONLY:i``    only CAV ``i`` follows it, every other CAV native (all members of G, plus
                   ``--extra-solo`` sampled CAVs);
   * ``GROUP``     only the close group ``G`` follows it (CAVs chained within 150 m along s,
                   size 2..6);
   * ``FLIP:i``    ``G \\ {i}`` follows it (``--n-flip`` sampled members).

   CAVs that enter during a branch are native unless the branch is ``ALL``.
3. Every branch is scored with the same team metrics as ``wms.control.reward.team_reward``
   (default ``RewardWeights``; called per 0.5 s step on the SUMO slot state with exits
   carried and reused slots split, so vehicle identity is exact) plus the env's
   ``RequestLedger`` source waiting, true-gap TTC exposure, braking, arrivals, collisions and
   an exact CVaR(90 %) of per-vehicle branch delay (on-road delay + source waiting).

Per decision point (sign convention: ``D_X[m] = improvement of branch X over B`` for metric
``m``, i.e. ``cost_B - cost_X`` for cost metrics and ``value_X - value_B`` for benefit
metrics; POSITIVE = BETTER):

    D_i, D_G, D_{G\\i}, D_all;  C_G = D_G - sum_{i in G} D_i  (cooperation gain);
    M_i = D_G - D_{G\\i}  (marginal-in-team);  I_i = M_i - D_i  (interaction).

Outputs (``--out``, a new directory): ``branches.jsonl`` (one record per branch: members,
metrics, wall-clock), ``points.jsonl`` (one per decision point: group, D/C/M/I for every
metric, determinism and save/load fidelity), ``BRANCHING_SUMMARY.json`` (bootstrap 95 % CIs
across decision points and seeds, wall-clock per decision point).

Every load goes into a FRESH SUMO process (same launch command, then ``loadState``):
loading into the running process leaks pre-load simulator state into the first steps
(observed: B vs B_repeat divergence and spurious junction collisions).  speedMode is not in
the state file and is re-asserted (27) for every CAV the ownership had configured.

Determinism checks: ``B`` vs ``B_repeat`` (two loads of the same state, must be identical:
max |ds| over (time, vehicle)); and save/load fidelity = the ``ALL`` branch (after a load) vs
the base episode's own live continuation of the same ``H`` seconds (no load), reported when
``spacing >= H``.
"""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import re
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

SAVE_STATE_CLI = ("--save-state.precision", "17", "--save-state.rng", "true")
NAMED = ("native", "damping", "pi")
SPEED_LIMIT_MPS = 31.29
MERGE_S1_M = 75.44
DT_S = 0.5
GROUP_LINK_M = 150.0
GROUP_MIN, GROUP_MAX = 2, 6
CVAR_ALPHA = 0.9
_RESTRICTED = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1", re.I)

#: metric -> +1 benefit (higher is better) / -1 cost (lower is better)
METRIC_SIGN = {
    "team_delay_vs": -1, "road_delay_vs": -1, "source_wait_vs": -1,
    "system_vs": -1, "fairness_cvar90_s": -1, "fairness_step_mean_s": -1,
    "throughput_soft": +1, "merge_crossings": +1, "arrivals": +1, "reward": +1,
    "smooth": -1, "hard_brake": -1, "braking_vehicle_seconds": -1,
    "acceleration_energy": -1, "cost_ttc": -1, "ttc_below_3_vs": -1,
    "collisions": -1, "teleports": -1,
}
PRIMARY_METRIC = "team_delay_vs"


def checked(path) -> Path:
    value = Path(path)
    if _RESTRICTED.search(str(value)) or _RESTRICTED.search(str(value.resolve())):
        raise ValueError(f"restricted path refused: {value}")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = sorted(value) if isinstance(value, (set, frozenset)) else value
        return [_jsonable(v) for v in seq]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


# ===================================================================== groups / arms
def chain_groups(s_by_vid: dict, *, link_m: float = GROUP_LINK_M) -> list[list[str]]:
    """Single-linkage chains of CAVs along s (any lane): consecutive gap <= ``link_m``."""
    order = sorted(s_by_vid, key=lambda v: (s_by_vid[v], v))
    groups, cur = [], []
    for vid in order:
        if cur and s_by_vid[vid] - s_by_vid[cur[-1]] > link_m:
            groups.append(cur)
            cur = []
        cur.append(vid)
    if cur:
        groups.append(cur)
    return groups


def select_group(s_by_vid: dict, *, link_m: float = GROUP_LINK_M, min_size: int = GROUP_MIN,
                 max_size: int = GROUP_MAX) -> Optional[list[str]]:
    """The close group G: the largest chain (ties -> nearest the merge end); a chain longer
    than ``max_size`` is cut to its tightest window of ``max_size`` consecutive CAVs."""
    chains = [c for c in chain_groups(s_by_vid, link_m=link_m) if len(c) >= min_size]
    if not chains:
        return None

    def key(c):
        return (min(len(c), max_size), -min(abs(s_by_vid[v] - MERGE_S1_M) for v in c))
    best = max(chains, key=key)
    if len(best) > max_size:
        spans = [(s_by_vid[best[j + max_size - 1]] - s_by_vid[best[j]], j)
                 for j in range(len(best) - max_size + 1)]
        _, j = min(spans)
        best = best[j:j + max_size]
    return list(best)


def plan_seed_arms(n_arms: int, *, native: bool = True) -> list[dict]:
    """Re-seeded random arms for undirected data (traffic_shepherd.c4.genrand): ``SEED:j``
    (every CAV runs the controller re-seeded with ``j``; the controller must offer
    ``reseed(seed)``) for j < ``n_arms``, plus the all-native reference ``B`` (``native``)."""
    arms = [{"name": f"SEED:{j}", "kind": "SEED", "members": None, "i": None, "j": j}
            for j in range(int(n_arms))]
    if native:
        arms.append({"name": "B", "kind": "B", "members": frozenset(), "i": None, "j": None})
    return arms


def reseeded(fn_snap, seed: int):
    """Deep copy of a snapshotted stateful controller, re-seeded (``SEED:j`` arms)."""
    if fn_snap is None or not hasattr(fn_snap, "reseed"):
        raise TypeError("SEED arms need a controller with reseed(seed)")
    return copy.deepcopy(fn_snap).reseed(int(seed))


def seed_for_arm(base_seed: int, t_s: int, j: int) -> int:
    """Deterministic seed of arm ``SEED:j`` at decision time ``t_s`` (same for every run)."""
    return int(np.random.SeedSequence([int(base_seed), int(t_s), int(j), 0xB4A2]
                                      ).generate_state(1)[0])


def plan_arms(cavs: list, group: Optional[list], *, rng, extra_solo: int = 0,
              n_flip: int = 2, repeat_reference: bool = True) -> list[dict]:
    """Branch arms for one decision point: ``{"name", "kind", "members" (None = all), "i"}``."""
    arms = [{"name": "B", "kind": "B", "members": frozenset(), "i": None}]
    if repeat_reference:
        arms.append({"name": "B_repeat", "kind": "B_repeat", "members": frozenset(), "i": None})
    arms.append({"name": "ALL", "kind": "ALL", "members": None, "i": None})
    solo = list(group or [])
    others = [v for v in cavs if v not in solo]
    if extra_solo > 0 and others:
        pick = rng.choice(len(others), size=min(extra_solo, len(others)), replace=False)
        solo += [others[int(j)] for j in sorted(pick.tolist())]
    for vid in solo:
        arms.append({"name": f"ONLY:{vid}", "kind": "ONLY", "members": frozenset([vid]), "i": vid})
    if group:
        arms.append({"name": "GROUP", "kind": "GROUP", "members": frozenset(group), "i": None})
        if n_flip > 0:
            pick = rng.choice(len(group), size=min(n_flip, len(group)), replace=False)
            for j in sorted(pick.tolist()):
                vid = group[int(j)]
                arms.append({"name": f"FLIP:{vid}", "kind": "FLIP",
                             "members": frozenset(group) - {vid}, "i": vid})
    return arms


# ===================================================================== cooperation algebra
def improvement(metrics_x: dict, metrics_b: dict) -> dict:
    """``D[m]`` = signed improvement of X over the all-native reference (positive = better)."""
    return {m: sign * (float(metrics_x[m]) - float(metrics_b[m]))
            for m, sign in METRIC_SIGN.items() if m in metrics_x and m in metrics_b}


def cooperation(d: dict, group: Optional[list]) -> dict:
    """Cooperation terms per metric from branch improvements ``d[name] -> {metric: D}``."""
    metrics = sorted(d["ALL"]) if "ALL" in d else []
    out = {"D_all": d.get("ALL"), "D_i": {}, "D_G": None, "D_G_minus_i": {},
           "sum_D_i": None, "C_G": None, "M_i": {}, "I_i": {}}
    for name, value in d.items():
        if name.startswith("ONLY:"):
            out["D_i"][name[5:]] = value
        elif name.startswith("FLIP:"):
            out["D_G_minus_i"][name[5:]] = value
    if group and "GROUP" in d and all(v in out["D_i"] for v in group):
        dg = d["GROUP"]
        out["D_G"] = dg
        out["sum_D_i"] = {m: sum(out["D_i"][v][m] for v in group) for m in metrics}
        out["C_G"] = {m: dg[m] - out["sum_D_i"][m] for m in metrics}
        for vid, flip in out["D_G_minus_i"].items():
            out["M_i"][vid] = {m: dg[m] - flip[m] for m in metrics}
            out["I_i"][vid] = {m: out["M_i"][vid][m] - out["D_i"][vid][m] for m in metrics}
    return out


def bootstrap_ci(values, *, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> dict:
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], np.float64)
    if x.size == 0:
        return {"n": 0, "mean": None, "lo": None, "hi": None}
    if x.size == 1:
        return {"n": 1, "mean": float(x[0]), "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(1)
    a = (1.0 - level) / 2.0
    return {"n": int(x.size), "mean": float(x.mean()), "lo": float(np.quantile(means, a)),
            "hi": float(np.quantile(means, 1.0 - a)), "frac_positive": float(np.mean(x > 0))}


def cvar(values, alpha: float = CVAR_ALPHA) -> float:
    """Empirical upper-tail CVaR: mean of the worst ceil((1 - alpha) n) values."""
    x = np.sort(np.asarray(list(values), np.float64))[::-1]
    if x.size == 0:
        return 0.0
    k = max(1, int(math.ceil((1.0 - alpha) * x.size - 1e-9)))
    return float(x[:k].mean())


# ===================================================================== per-step metrics
def lane_of(kin: np.ndarray) -> np.ndarray:
    onehot = kin[:, 1:4]
    return np.where(onehot.max(-1) > 0.5, onehot.argmax(-1), -1).astype(np.int64)


def step_pair(kin0, m0, id0, kin1, m1, id1) -> tuple[dict, dict, np.ndarray, dict]:
    """(state, next_state, accel, info) for ``team_reward`` with exact vehicle identity.

    Slots of vehicles that left carry their last values (absent from the next state);
    a slot reused by a new vehicle in the same step is split: the new vehicle moves to an
    appended slot that is absent in ``state`` and present in ``next_state``.
    """
    m0, m1 = np.asarray(m0, bool), np.asarray(m1, bool)
    cont = m0 & m1 & (np.asarray(id0) == np.asarray(id1))
    reused = m0 & m1 & ~cont
    extra = np.flatnonzero(reused)
    n = m0.shape[0] + extra.size
    lane0, lane1 = lane_of(kin0), lane_of(kin1)

    def pad(x, fill=0):
        return np.concatenate([x, np.full(extra.size, fill, dtype=x.dtype)])
    state = {"s_m": pad(np.where(m0, kin0[:, 0], 0.0)), "v_mps": pad(np.where(m0, kin0[:, 4], 0.0)),
             "lane": pad(np.where(m0, lane0, -1), -1), "present": pad(m0, False),
             "length_m": np.full(n, 5.0, np.float32)}
    new_free = m1 & ~m0
    take1 = cont | new_free
    s1 = np.where(take1, kin1[:, 0], kin0[:, 0])
    v1 = np.where(take1, kin1[:, 4], kin0[:, 4])
    l1 = np.where(take1, lane1, lane0)
    nxt = {"s_m": np.concatenate([s1, kin1[extra, 0]]),
           "v_mps": np.concatenate([v1, kin1[extra, 4]]),
           "lane": np.concatenate([l1, lane1[extra]]).astype(np.int64),
           "present": np.concatenate([take1, np.ones(extra.size, bool)]),
           "length_m": np.full(n, 5.0, np.float32)}
    accel = pad(np.where(cont, kin1[:, 5], 0.0))
    crossings = int(np.sum(cont & (kin0[:, 0] < MERGE_S1_M) & (kin1[:, 0] >= MERGE_S1_M)
                           & (lane0 >= 0) & (lane1 >= 0)))
    v_next_cohort = np.where(cont, kin1[:, 4], kin0[:, 4])
    delay = np.where(m0, DT_S * np.maximum(0.0, 1.0 - v_next_cohort / SPEED_LIMIT_MPS), 0.0)
    return state, nxt, accel, {"merge_crossings": crossings, "delay_by_slot": delay}


class BranchTracker:
    """Accumulates one branch's team metrics after every ``env.step`` (two physics steps)."""

    def __init__(self, env, *, record_trajectory: bool = True, reward_fn: Optional[Callable] = None):
        self.env = env
        self.t0 = env.now
        self.counters = Counter()
        self.reward_sums = Counter()
        self.fair_steps = []
        self.delay_by_vid = defaultdict(float)
        self.first_seen: dict[str, float] = {}
        ledger = env.ledger
        self.waiting = {vid: r.depart_s for vid, r in ledger.requests.items()
                        if vid not in ledger.inserted}
        self.record = record_trajectory
        self.trajectory: list = []
        self.reward_fn = reward_fn
        if self.reward_fn is None:
            from .wms_glue import team_reward_terms
            self.reward_fn = team_reward_terms

    def update(self, info: dict) -> None:
        env = self.env
        self.counters.update(info["interval"])
        raw = env.history.raw(3)
        ids = env.builder.vehicle_ids
        now = env.now
        for j in (0, 1):
            kin0, m0, id0 = raw["kin"][j], raw["mask"][j], raw["vid_index"][j]
            kin1, m1, id1 = raw["kin"][j + 1], raw["mask"][j + 1], raw["vid_index"][j + 1]
            t1 = now - DT_S * (1 - j)
            state, nxt, accel, extra = step_pair(kin0, m0, id0, kin1, m1, id1)
            terms = self.reward_fn(state, nxt, accel)
            for key, value in terms.items():
                if key != "fairness_delay":
                    self.reward_sums[key] += value
            self.fair_steps.append(terms.get("fairness_delay", 0.0))
            self.counters["merge_crossings"] += extra["merge_crossings"]
            for slot in np.flatnonzero(m0).tolist():
                self.delay_by_vid[ids[int(id0[slot])]] += float(extra["delay_by_slot"][slot])
            present = np.flatnonzero(m1).tolist()
            for slot in present:
                vid = ids[int(id1[slot])]
                self.first_seen.setdefault(vid, t1)
            if self.record:
                self.trajectory.append((round(t1, 3), {ids[int(id1[s])]: (float(kin1[s, 0]),
                                                                          float(kin1[s, 4]))
                                                       for s in present}))

    def finish(self) -> dict:
        t_end = self.env.now
        per_vehicle = dict(self.delay_by_vid)
        source_by_vid = {}
        for vid, depart in self.waiting.items():
            start = max(depart, self.t0)
            if start >= t_end:
                continue
            stop = min(self.first_seen.get(vid, t_end), t_end)
            if stop > start:
                source_by_vid[vid] = stop - start
                per_vehicle[vid] = per_vehicle.get(vid, 0.0) + (stop - start)
        c = self.counters
        road = float(self.reward_sums.get("delay", 0.0))
        source = float(c.get("source_wait_vehicle_seconds", 0.0))
        return {
            "t0_s": self.t0, "t_end_s": t_end,
            "team_delay_vs": road + source, "road_delay_vs": road, "source_wait_vs": source,
            "source_wait_per_vehicle_sum_vs": float(sum(source_by_vid.values())),
            "system_vs": float(c.get("requested_not_arrived_vehicle_seconds", 0.0)),
            "fairness_cvar90_s": cvar(per_vehicle.values()),
            "fairness_step_mean_s": float(np.mean(self.fair_steps)) if self.fair_steps else 0.0,
            "throughput_soft": float(self.reward_sums.get("throughput", 0.0)),
            "merge_crossings": int(c.get("merge_crossings", 0)),
            "arrivals": int(c.get("arrivals", 0)),
            "reward": float(self.reward_sums.get("reward", 0.0)),
            "smooth": float(self.reward_sums.get("smooth", 0.0)),
            "hard_brake": float(self.reward_sums.get("hard_brake", 0.0)),
            "cost_ttc": float(self.reward_sums.get("cost_ttc", 0.0)),
            "braking_vehicle_seconds": float(c.get("braking_vehicle_seconds", 0.0)),
            "acceleration_energy": float(c.get("acceleration_energy", 0.0)),
            "ttc_below_3_vs": float(c.get("ttc_below_3_vehicle_seconds", 0.0)),
            "collisions": int(c.get("collision_involvements", 0)),
            "teleports": int(c.get("starting_teleports", 0)),
            "dropped_vehicle_steps": int(c.get("dropped_vehicle_steps", 0)),
            "n_vehicles": len(per_vehicle),
        }


def trajectory_diff(a: list, b: list) -> dict:
    """max |ds|, max |dv| over common (time, vehicle) and the count of mismatched entries."""
    ds = dv = 0.0
    mismatched = 0
    bt = dict(b)
    for t, rows in a:
        other = bt.get(t)
        if other is None:
            mismatched += len(rows)
            continue
        mismatched += len(set(rows) ^ set(other))
        for vid in set(rows) & set(other):
            ds = max(ds, abs(rows[vid][0] - other[vid][0]))
            dv = max(dv, abs(rows[vid][1] - other[vid][1]))
    mismatched += sum(len(r) for t, r in b if t not in dict(a))
    return {"max_abs_ds_m": ds, "max_abs_dv_mps": dv, "mismatched_entries": mismatched,
            "n_steps": len(a), "identical": ds == 0.0 and dv == 0.0 and mismatched == 0}


# ===================================================================== controller plug-in
def split_output(out):
    """Policy output -> (accel [1,K] float array, lane [1,K] or None).

    Accepts an accel array/tensor, ``(accel, lane)`` or ``{"accel": ..., "lane": ...}``."""
    lane = None
    if isinstance(out, dict):
        accel, lane = out["accel"], out.get("lane")
    elif isinstance(out, tuple) and len(out) == 2:
        accel, lane = out
    else:
        accel = out
    return _np(accel).astype(np.float64).reshape(1, -1), (None if lane is None
                                                          else _np(lane).reshape(1, -1))


def _np(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(x)


class BranchController:
    """Controller under test: a ``state -> accel [1,K]`` callable (optionally with lane
    commands, see :func:`split_output`) or one of the frozen per-CAV names native / damping /
    pi.  Strings ``random`` / ``zero`` / ``module:attr`` / checkpoint paths resolve through
    :func:`traffic_shepherd.c4.baselines.make_policy`."""

    def __init__(self, spec, *, seed: int = 0, name: Optional[str] = None):
        if isinstance(spec, str) and spec in NAMED:
            self.named, self.fn = spec, None
        elif callable(spec):
            self.named, self.fn = None, spec
        elif isinstance(spec, str):
            from .baselines import make_policy
            resolved = make_policy(spec, seed=seed)
            if isinstance(resolved, str):
                if resolved not in NAMED:
                    raise ValueError(f"{resolved!r} is a group action, not a per-CAV controller")
                self.named, self.fn = resolved, None
            else:
                self.named, self.fn = None, resolved
        else:
            raise TypeError("controller must be a callable or a name")
        self.name = name or (self.named if self.named else
                             (spec if isinstance(spec, str) else type(spec).__name__))
        self.lane_dropped = 0

    @property
    def base_mode(self) -> str:
        """Env controller of the base episode: always ``direct``.  Named controllers run
        through the env's per-tick ``fallback`` path, the same code path as every branch
        arm, so the base episode's live continuation and the ``ALL`` arm differ only by the
        SUMO save/load round trip."""
        return "direct"

    def snapshot(self):
        return None if self.fn is None else copy.deepcopy(self.fn)

    def command(self, env, state, members, fn) -> tuple:
        """``(accel or None, step kwargs)`` for one control tick; ``members`` None = all CAVs."""
        if self.named == "native":
            return None, {}
        if self.named is not None:
            return None, {"fallback": self.named,
                          "fallback_ids": None if members is None else frozenset(members)}
        accel, lane = split_output(fn(state))
        mask = member_mask(env, members)
        out = np.full(accel.shape, np.nan)
        out[0, mask] = accel[0, mask]
        kwargs = {}
        if lane is not None:
            # env lane semantics: [1,K] WM target lane, -1 = no change (non-members stay -1)
            if "lanes" in inspect.signature(env.step).parameters:
                lane_out = np.full(lane.shape, -1, dtype=np.int64)
                lane_out[0, mask] = np.asarray(lane[0, mask], dtype=np.int64)
                kwargs["lanes"] = lane_out
            elif np.any(lane[0, mask] >= 0):
                self.lane_dropped += int(np.sum(lane[0, mask] >= 0))
        return out, kwargs


def member_mask(env, members) -> np.ndarray:
    f = env.frame
    mask = f.present & f.is_cav
    if members is None:
        return mask
    ids = env.builder.vehicle_ids
    keep = np.array([vi >= 0 and ids[vi] in members for vi in f.vid_index.tolist()], bool)
    return mask & keep


# ===================================================================== save / restore
def _shared(env) -> dict:
    """Objects shared (not copied) by snapshots: the TraCI module and the ONNX controller."""
    memo = {}
    for obj in (env.api, getattr(env, "_circles", None)):
        if obj is not None:
            memo[id(obj)] = obj
    return memo


def snapshot_env(env) -> dict:
    if env.recorder is not None:
        raise ValueError("branching runs without an episode recorder")
    return copy.deepcopy(dict(env.__dict__), _shared(env))


def restore_env(env, snap: dict) -> None:
    env.__dict__.update(copy.deepcopy(snap, _shared(env)))


#: id(env) -> (TraCI connection, SUMO time, state file) of the last ``loadState`` into the
#: env's current SUMO process (the connection object identifies the process)
_LOADED: dict = {}
_LOADER_TIME = re.compile(rb"<delay [^>]* loaderTime=")


def _connection(env):
    try:
        return env.api.getConnection()
    except Exception:  # noqa: BLE001 - no live connection / non-traci api
        return None


def save_state(env, path: Path) -> None:
    """``saveState`` to ``path``; the written file must carry the route loader's ``loaderTime``.

    SUMO 1.27.1 only writes ``<delay loaderTime=...>`` once the route loader rebuilt by
    ``loadState`` has loaded routes again, i.e. after the first ``simulationStep``.  A state
    saved right after a ``loadState`` (no step in between) therefore lacks it, and loading that
    file rewinds the route file without skipping already-handled requests: every vehicle that
    had already arrived is inserted a second time (seen as a duplicate departure in the
    RequestLedger, e.g. ``mainline.0`` re-departing at t = 902.5 s).  This happened whenever the
    live continuation ended exactly on the next decision point (``spacing == horizon``).  The
    process state is then by definition the loaded file, so that file is copied byte for byte."""
    marker = _LOADED.get(id(env))
    conn = _connection(env)
    if (marker is not None and conn is not None and marker[0] is conn
            and abs(env.now - marker[1]) < 1e-9):
        if Path(marker[2]).resolve() != Path(path).resolve():
            shutil.copyfile(marker[2], path)
    else:
        env.api.simulation.saveState(str(path))
    check_state_file(path)


def check_state_file(path: Path) -> None:
    """Refuse a state file without ``loaderTime`` (its load would re-insert arrived vehicles)."""
    with open(path, "rb") as fh:
        if not _LOADER_TIME.search(fh.read()):
            raise RuntimeError(f"state file {path} has no <delay loaderTime>: loading it would "
                               "re-insert already-arrived vehicles (SUMO route-loader rewind)")


def relaunch(env) -> None:
    """Close this process's SUMO and start a fresh one with the episode's launch command.

    ``loadState`` into the SAME running SUMO is not clean: state of the pre-load simulation
    leaks into the first steps after the load (observed: B vs B_repeat diverging, spurious
    junction collisions), so every load goes into a fresh SUMO process (the campaign
    convention, verified bit-exact there)."""
    from traffic_shepherd.simulation import start_sumo

    api = env.api
    if api.isLoaded():
        api.close()
    start_sumo(env.binary, env.scene, seed=env.seed, config=env.config, api=api,
               extra_cli=env.extra_sumo_cli or None)


def load_state(env, path: Path, snap: dict, *, fresh: bool = True) -> dict:
    """SUMO ``loadState`` (into a fresh SUMO process by default) + Python state restore;
    re-subscribe every vehicle and re-assert speedMode 27 (not stored in the state file)."""
    check_state_file(path)
    if fresh:
        relaunch(env)
    env.api.simulation.loadState(str(path))
    restore_env(env, snap)
    conn = _connection(env)
    if conn is not None:
        _LOADED[id(env)] = (conn, float(env.now), Path(path))
    env.reader._subscribed = set()
    if abs(env.now - snap["frame"].time_s) > 1e-6:
        raise RuntimeError(f"loadState landed at {env.now}, expected {snap['frame'].time_s}")
    restored = 0
    live = set(env.api.vehicle.getIDList())
    for vid in sorted(env.ownership._mode_configured & live):
        if env.api.vehicle.getSpeedMode(vid) != env.ownership.cav_speed_mode:
            env.api.vehicle.setSpeedMode(vid, env.ownership.cav_speed_mode)
            restored += 1
    return {"speed_mode_restored": restored}


def prepare_branch_mode(env) -> None:
    """Guard: base episode and branches all step the env in ``direct`` mode."""
    if env.mode != "direct" or env.fleet is not None or env.wall_law is not None:
        raise RuntimeError("branching requires a direct-mode environment")


def run_branch(env, controller: BranchController, members, horizon_s: int, fn,
               *, record: bool = True) -> tuple[dict, list]:
    tracker = BranchTracker(env, record_trajectory=record)
    for _ in range(int(horizon_s)):
        accel, kwargs = controller.command(env, env.state(), members, fn)
        _, info = env.step(accel, **kwargs)
        if info["done"]:
            raise RuntimeError("episode ended inside a branch (base horizon too short)")
        tracker.update(info)
    return tracker.finish(), tracker.trajectory


def base_step(env, controller: BranchController, fn) -> dict:
    accel, kwargs = controller.command(env, env.state(), None, fn)
    _, info = env.step(accel, **kwargs)
    return info


def cav_positions(env) -> dict:
    f = env.frame
    ids = env.builder.vehicle_ids
    return {ids[int(f.vid_index[s])]: float(f.s[s])
            for s in np.flatnonzero(f.present & f.is_cav).tolist()}


# ===================================================================== episode driver
def measure_episode(env, controller: BranchController, *, seed: int, demand_vph: int,
                    penetration: float, warmup_s: int, decision_times: list, horizon_s: int,
                    workdir: Path, rng, extra_solo: int = 0, n_flip: int = 2,
                    link_m: float = GROUP_LINK_M, branch_sink: Optional[Callable] = None,
                    imagine: Optional[Callable] = None,
                    base_controller: Optional[BranchController] = None,
                    seed_arms: int = 0, seed_base: int = 0) -> list[dict]:
    """Run one base episode and branch at every decision time; returns per-point records.

    ``base_controller`` (optional) drives the base episode instead of ``controller`` so two
    controllers (e.g. MAPPO vs ours) can be branched from IDENTICAL start states; the arms'
    callable then starts each point from its fresh (reset) state and the live-continuation
    fidelity check is skipped."""
    times = sorted(int(t) for t in decision_times)
    if not times or times[0] <= warmup_s:
        raise ValueError("decision times must lie after the warmup")
    spacing = min((b - a for a, b in zip(times, times[1:])), default=None)
    base = base_controller if base_controller is not None else controller
    live_check = (spacing is None or spacing >= horizon_s) and base is controller
    end = times[-1] + horizon_s + 1
    fn = controller.fn
    if fn is not None and hasattr(fn, "reset"):
        fn.reset()
    base_fn = base.fn
    if base is not controller and base_fn is not None and hasattr(base_fn, "reset"):
        base_fn.reset()
    env.reset(seed=seed, demand_vph=demand_vph, penetration=penetration, warmup_s=warmup_s,
              horizon_s=end - warmup_s, controller=base.base_mode)
    workdir.mkdir(parents=True, exist_ok=True)
    points = []
    for k, t_k in enumerate(times):
        while env.now < t_k - 1e-9:
            base_step(env, base, base_fn)
        wall0 = time.monotonic()
        path = workdir / f"s{seed}_t{t_k}.xml"
        save_state(env, path)
        snap = snapshot_env(env)
        fn_snap = controller.snapshot()
        cavs = cav_positions(env)
        group = select_group(cavs, link_m=link_m)
        if seed_arms > 0:            # undirected re-seeded arms (no group / cooperation algebra)
            arms = plan_seed_arms(seed_arms) + [{"name": "B_repeat", "kind": "B_repeat",
                                                 "members": frozenset(), "i": None, "j": None}]
        else:
            arms = plan_arms(sorted(cavs), group, rng=rng, extra_solo=extra_solo, n_flip=n_flip)
        imagined = None
        if imagine is not None:
            imagined = imagine(env, arms, controller, fn_snap, horizon_s)
            imagined["coop"] = cooperation(imagined["D"], group)
        # live continuation (no load) of the base controller = ALL without a reload
        live = None
        if live_check:
            live_metrics, live_traj = run_branch_live(env, controller, fn, horizon_s)
            mid_path = workdir / f"s{seed}_t{t_k}_mid.xml"
            if k + 1 < len(times):
                save_state(env, mid_path)
                mid_snap = snapshot_env(env)
            live = {"metrics": live_metrics, "trajectory": live_traj}
        metrics, trajs, notes = {}, {}, {}
        for arm in arms:
            t_arm = time.monotonic()
            note = load_state(env, path, snap)
            prepare_branch_mode(env)
            if arm["kind"] == "SEED":
                arm_fn = reseeded(fn_snap, seed_for_arm(seed_base, t_k, arm["j"]))
            else:
                arm_fn = None if fn_snap is None else copy.deepcopy(fn_snap)
            m, traj = run_branch(env, controller, arm["members"], horizon_s, arm_fn)
            metrics[arm["name"]], trajs[arm["name"]] = m, traj
            notes[arm["name"]] = dict(note, wall_s=time.monotonic() - t_arm)
            if branch_sink is not None:
                branch_sink({"seed": seed, "t_s": t_k, "arm": arm["name"], "kind": arm["kind"],
                             "members": None if arm["members"] is None else sorted(arm["members"]),
                             "metrics": m, **notes[arm["name"]]})
        # resume the base episode
        if k + 1 < len(times):
            if live_check:
                load_state(env, mid_path, mid_snap)
            else:
                load_state(env, path, snap)
        d = {name: improvement(m, metrics["B"]) for name, m in metrics.items()}
        record = {
            "seed": seed, "demand_vph": demand_vph, "penetration": penetration,
            "t_s": t_k, "horizon_s": horizon_s, "controller": controller.name,
            "base_controller": base.name,
            "n_cavs_live": len(cavs), "group": group,
            "group_s_m": None if group is None else [cavs[v] for v in group],
            "arms": [a["name"] for a in arms],
            "metrics": metrics, "D": d, "coop": cooperation(d, group),
            "determinism_B_vs_B_repeat": (trajectory_diff(trajs["B"], trajs["B_repeat"])
                                          if "B_repeat" in trajs else None),
            "metrics_B_equal_B_repeat": (metrics["B"] == metrics.get("B_repeat")),
            "fidelity_ALL_vs_live": (None if live is None or "ALL" not in trajs
                                     else trajectory_diff(live["trajectory"], trajs["ALL"])),
            "metrics_live": None if live is None else live["metrics"],
            "speed_mode_restored": sum(n["speed_mode_restored"] for n in notes.values()),
            "lane_commands_dropped": controller.lane_dropped,
            "wall_s": time.monotonic() - wall0, "n_branches": len(arms),
            "imagined": imagined,
        }
        points.append(record)
    env.close()
    return points


def run_branch_live(env, controller, fn, horizon_s) -> tuple[dict, list]:
    tracker = BranchTracker(env)
    for _ in range(int(horizon_s)):
        info = base_step(env, controller, fn)
        if info["done"]:
            raise RuntimeError("episode ended during the live continuation")
        tracker.update(info)
    return tracker.finish(), tracker.trajectory


# ===================================================================== summary
SUMMARY_METRICS = ("team_delay_vs", "road_delay_vs", "source_wait_vs", "reward",
                   "fairness_cvar90_s", "merge_crossings", "hard_brake", "ttc_below_3_vs")


def summarize(points: list, *, n_boot: int = 2000) -> dict:
    out = {"n_points": len(points), "convention": (
        "D_X[m] = improvement of branch X over all-native B (cost_B - cost_X for costs, "
        "value_X - value_B for benefits); positive = better. C_G = D_G - sum_i D_i; "
        "M_i = D_G - D_{G minus i}; I_i = M_i - D_i."), "metrics": {}}
    for m in SUMMARY_METRICS:
        rows = defaultdict(list)
        for p in points:
            c = p["coop"]
            if c["D_all"]:
                rows["D_all"].append(c["D_all"].get(m))
            if c["D_G"]:
                rows["D_G"].append(c["D_G"].get(m))
                rows["sum_D_i"].append(c["sum_D_i"].get(m))
                rows["C_G"].append(c["C_G"].get(m))
            for vid, v in c["D_i"].items():
                rows["D_i (per CAV)"].append(v.get(m))
            for vid, v in c["M_i"].items():
                rows["M_i (per flip)"].append(v.get(m))
                rows["I_i (per flip)"].append(c["I_i"][vid].get(m))
        out["metrics"][m] = {key: bootstrap_ci(vals, n_boot=n_boot) for key, vals in rows.items()}
    det = [p["determinism_B_vs_B_repeat"] for p in points if p["determinism_B_vs_B_repeat"]]
    fid = [p["fidelity_ALL_vs_live"] for p in points if p["fidelity_ALL_vs_live"]]
    out["determinism"] = {
        "max_abs_ds_m": max((d["max_abs_ds_m"] for d in det), default=None),
        "max_abs_dv_mps": max((d["max_abs_dv_mps"] for d in det), default=None),
        "mismatched_entries": sum(d["mismatched_entries"] for d in det),
        "all_identical": all(d["identical"] for d in det) if det else None,
        "metrics_identical": all(p["metrics_B_equal_B_repeat"] for p in points)}
    out["save_load_fidelity_ALL_vs_live"] = {
        "max_abs_ds_m": max((d["max_abs_ds_m"] for d in fid), default=None),
        "max_abs_dv_mps": max((d["max_abs_dv_mps"] for d in fid), default=None),
        "mismatched_entries": sum(d["mismatched_entries"] for d in fid),
        "all_identical": all(d["identical"] for d in fid) if fid else None}
    walls = [p["wall_s"] for p in points]
    out["wall_s_per_decision_point"] = {
        "mean": float(np.mean(walls)) if walls else None,
        "max": float(np.max(walls)) if walls else None,
        "mean_branches": float(np.mean([p["n_branches"] for p in points])) if points else None}
    imagined = [p["imagined"] for p in points if p.get("imagined")]
    if imagined:
        out["imagined_vs_sumo"] = {m: imagined_agreement(points, m)
                                   for m in ("reward", "road_delay_vs", "throughput_soft")}
        out["imagined_wall_s_mean"] = float(np.mean([i.get("wall_s", 0.0) for i in imagined]))
    return out


def imagined_agreement(points: list, metric: str = "reward") -> dict:
    """Correlation and sign agreement of imagined vs SUMO branch improvements D."""
    sim, img = [], []
    for p in points:
        im = p.get("imagined") or {}
        for arm, d in (im.get("D") or {}).items():
            if arm in p["D"] and metric in d and metric in p["D"][arm] and arm not in ("B", "B_repeat"):
                sim.append(p["D"][arm][metric])
                img.append(d[metric])
    if len(sim) < 2:
        return {"metric": metric, "n": len(sim)}
    a, b = np.asarray(sim), np.asarray(img)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None
    return {"metric": metric, "n": len(sim), "pearson": corr,
            "sign_agreement": float(np.mean(np.sign(a) == np.sign(b)))}


# ===================================================================== CLI
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m traffic_shepherd.c4.branching",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--controller", required=True,
                    help="native|damping|pi|random|zero|module:attr|<policy checkpoint>")
    ap.add_argument("--base-controller", default=None,
                    help="optional controller of the base episode (start states); default = --controller")
    ap.add_argument("--out", required=True, help="new output directory")
    ap.add_argument("--seeds", default="4107", help="e.g. 4107-4110,4120 (exposed C4 seeds)")
    ap.add_argument("--demand", type=int, default=2700)
    ap.add_argument("--penetration", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--points", type=int, default=2, help="decision points per seed")
    ap.add_argument("--spacing", type=int, default=60, help="s between decision points")
    ap.add_argument("--first", type=int, default=None, help="first decision time (default warmup+spacing)")
    ap.add_argument("--horizon", type=int, default=30, help="branch horizon H in s")
    ap.add_argument("--extra-solo", type=int, default=0, help="extra ONLY_i CAVs outside G")
    ap.add_argument("--n-flip", type=int, default=2, help="FLIP (G minus i) arms per point")
    ap.add_argument("--link-m", type=float, default=GROUP_LINK_M)
    ap.add_argument("--k-slots", type=int, default=192)
    ap.add_argument("--rng-seed", type=int, default=0, help="arm sampling / random policy seed")
    ap.add_argument("--wm-ckpt", default=None, help="optional WM-S checkpoint: imagined D's too")
    ap.add_argument("--wm-samples", type=int, default=1)
    ap.add_argument("--population", default="legacy3", choices=("legacy3", "textbook"),
                    help="SUMO driver population (default legacy3 = unchanged)")
    return ap


def main(argv=None) -> int:
    from .env import CavSumoEnv
    from .seeds import parse_seeds

    args = parser().parse_args(argv)
    out = checked(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"--out {out} exists and is not empty (never overwrite a run)")
    out.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    first = args.first if args.first is not None else args.warmup + args.spacing
    times = [first + k * args.spacing for k in range(args.points)]
    imagine = None
    if args.wm_ckpt:
        from .wms_glue import imagined_cooperation
        ckpt = checked(args.wm_ckpt)

        def imagine(env, arms, controller, fn_snap, horizon_s):
            return imagined_cooperation(ckpt, env, arms, controller, fn_snap, horizon_s,
                                        n_samples=args.wm_samples)
    manifest = {"schema": "c4_branching/1", "created": _now(), "args": vars(args),
                "decision_times_s": times, "save_state_cli": list(SAVE_STATE_CLI)}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    env_kw = {} if args.population == "legacy3" else {"population": args.population}
    env = CavSumoEnv(k_slots=args.k_slots, controller="direct", extra_sumo_cli=SAVE_STATE_CLI,
                     scenario_root=out / "scenarios", physics_log=False, **env_kw)
    rng = np.random.default_rng(args.rng_seed)
    all_points = []
    t_start = time.monotonic()
    with open(out / "branches.jsonl", "w", encoding="utf-8") as fb, \
            open(out / "points.jsonl", "w", encoding="utf-8") as fp:
        def sink(rec):
            fb.write(json.dumps(_jsonable(rec), allow_nan=False) + "\n")
            fb.flush()
        for seed in seeds:
            controller = BranchController(args.controller, seed=args.rng_seed + seed)
            base = (None if args.base_controller is None else
                    BranchController(args.base_controller, seed=args.rng_seed + seed))
            try:
                points = measure_episode(
                    env, controller, seed=seed, demand_vph=args.demand,
                    penetration=args.penetration, warmup_s=args.warmup, decision_times=times,
                    horizon_s=args.horizon, workdir=out / "states", rng=rng,
                    extra_solo=args.extra_solo, n_flip=args.n_flip, link_m=args.link_m,
                    branch_sink=sink, imagine=imagine, base_controller=base)
            finally:
                env.close()
            for p in points:
                fp.write(json.dumps(_jsonable(p), allow_nan=False) + "\n")
                fp.flush()
            all_points.extend(points)
    summary = summarize(all_points)
    summary.update(controller=args.controller, seeds=seeds, demand_vph=args.demand,
                   penetration=args.penetration, horizon_s=args.horizon,
                   decision_times_s=times, total_wall_s=time.monotonic() - t_start)
    (out / "BRANCHING_SUMMARY.json").write_text(json.dumps(_jsonable(summary), indent=2,
                                                           allow_nan=False), encoding="utf-8")
    print(json.dumps(_jsonable({k: summary[k] for k in (
        "n_points", "determinism", "save_load_fidelity_ALL_vs_live",
        "wall_s_per_decision_point")}), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

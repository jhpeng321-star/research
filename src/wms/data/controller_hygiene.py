"""DATAHYG: no designed controllers in the WM-S training windows (``data.controller_hygiene``).

Design rule: the world model learns action effects only from native driving, undirected random
direct-output commands and the learned policy's own rollouts.  Two TRAIN sources violated it
(audit runs/scratch/actiondata, DATAHYG_NOTES.md):

1. the switching pack's ``direct-mix`` arm switches every CAV among {native, CIRCLES damping,
   direct} (damping = 19.4 % of CAV time);
2. the dspeed trajectory arm's scripted segment types ``merge_aimed`` and ``gap_opening`` are
   designed behaviours (random_target / slowdown_ramp / speedup_ramp are undirected), and the
   loader labelled EVERY CAV of a dspeed window ``direct`` with the realized acceleration as
   its "command".

``"strict"`` (default ``"off"`` = bit-identical):

* direct-mix: an item is dropped when its window (``horizon_steps`` ticks from t0, which is
  also the CAV-teacher span) contains a ``damping`` control step of ANY CAV on the road - the
  t0 cohort from the recorded per-item timeline, every CAV (cohort and post-t0 entrants) from
  the generator's deterministic per-vehicle schedule (union of both).  A CAV on the road whose
  identity cannot be resolved counts as damping (fail closed).
* dspeed: windows whose span overlaps a ``merge_aimed`` / ``gap_opening`` segment of a
  directed CAV are refused - an item whose every start (unshifted and start-shifted) is
  refused is dropped; otherwise a refused draw is moved to an allowed start.  Timelines are
  relabelled: directed CAVs during the 120 s directed window = ``direct``, every other
  CAV/step = ``native``; issued commands are NOT recoverable (pack ``cav_plan_ch`` = realized
  v/a, overlay = outcomes only, shards gone) so they are 0 with ``command_known = False``.

Both generators are re-implemented bit-identically (golden values in tests/unit/test_datahyg.py):
``traffic_shepherd.switching_data.sample_switch_schedule`` (branch wms/swpack) and
``output/campaign_stage1_v3/trajectory_arm.py`` (branch impl/dspeed-arm).
"""
from __future__ import annotations

import functools
import hashlib
import math
import random
import re
from typing import Sequence

import numpy as np

HYGIENE_MODES = ("off", "strict")

# ---------------------------------------------------------------- direct-mix schedule ----
SWITCH_CHOICES_DIRECT_MIX = ("native", "damping", "direct")
SWITCH_HOLD_S = (1, 60)
SWITCH_DURATION_S = 1200
_WORLD_RE = re.compile(r"^w(\d+)m")


def world_seed(world: str) -> int:
    match = _WORLD_RE.match(str(world))
    if not match:
        raise ValueError(f"cannot parse a seed from world id {world!r}")
    return int(match.group(1))


def switch_schedule(seed: int, vehicle_id: str, *, duration_s: int = SWITCH_DURATION_S,
                    controllers: Sequence[str] = SWITCH_CHOICES_DIRECT_MIX,
                    min_hold_s: int = SWITCH_HOLD_S[0], max_hold_s: int = SWITCH_HOLD_S[1]
                    ) -> list[tuple[int, int, str]]:
    """One vehicle's ``(start_s, end_s, controller)`` segments (bit-identical generator copy)."""
    rng = random.Random(f"switch_schedule_v1:{seed}:{vehicle_id}")
    segments, tick, previous = [], 0, None
    while tick < duration_s:
        hold = max(min_hold_s, min(max_hold_s, int(round(math.exp(
            rng.uniform(math.log(min_hold_s), math.log(max_hold_s)))))))
        mode = rng.choice([m for m in controllers if m != previous])
        end = min(int(duration_s), tick + hold)
        segments.append((tick, end, mode))
        previous, tick = mode, end
    return segments


@functools.lru_cache(maxsize=65536)
def scheduled_damping_seconds(seed: int, vehicle_id: str, *,
                              duration_s: int = SWITCH_DURATION_S) -> np.ndarray:
    """``bool[duration_s]``: the vehicle is scheduled ``damping`` during that control second.

    Cached (read-only result; callers must not modify it)."""
    out = np.zeros(int(duration_s), bool)
    for start, end, mode in switch_schedule(seed, vehicle_id, duration_s=duration_s):
        if mode == "damping":
            out[start:end] = True
    out.setflags(write=False)
    return out


# ---------------------------------------------------------------- dspeed trajectory arm --
DSPEED_WINDOW_S = 120.0
DSPEED_SEGMENT_TYPES = ("random_target", "slowdown_ramp", "speedup_ramp", "merge_aimed",
                        "gap_opening")
DESIGNED_SEGMENTS = ("merge_aimed", "gap_opening")
#: uniform draws per segment after its length (target | decel+drop | accel+rise | ...)
_DSPEED_DRAWS = {"random_target": 1, "slowdown_ramp": 2, "speedup_ramp": 2,
                 "merge_aimed": 1, "gap_opening": 2}
#: ticks of margin around a designed segment (control-tick / pack-tick alignment)
DESIGNED_MARGIN_TICKS = 2


def dspeed_seed(world: str, arm_index: int) -> int:
    raw = hashlib.sha256(f"{world}|traj|{int(arm_index)}".encode("utf-8"))
    return int.from_bytes(raw.digest()[:8], "big")


def dspeed_selected(cav_ids: Sequence[str], seed: int, arm_index: int) -> list[str]:
    """The seeded directed subset of the t0 CAV roster (generator ``select_cavs``)."""
    cavs = sorted(str(c) for c in cav_ids)
    if not cavs:
        return []
    fraction = 0.3 if int(arm_index) % 2 == 0 else 0.5
    shuffled = list(cavs)
    random.Random(int(seed)).shuffle(shuffled)
    count = max(1, min(len(cavs), int(round(fraction * len(cavs)))))
    return sorted(shuffled[:count])


def dspeed_segments(seed: int, cav: str, *, horizon_s: float = DSPEED_WINDOW_S
                    ) -> list[tuple[float, float, str]]:
    """``(t_start_s, t_end_s, type)`` of one directed CAV (generator ``sample_segments``).

    Each segment type draws a fixed number of uniforms after its length, so the type/length
    sequence does not depend on the lane limit (only the random target value does).
    """
    digest = hashlib.sha256(f"{int(seed)}|{cav}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    out, t = [], 0.0
    while t < horizon_s - 1e-9:
        kind = DSPEED_SEGMENT_TYPES[rng.randrange(len(DSPEED_SEGMENT_TYPES))]
        length = 10.0 if kind == "gap_opening" else rng.uniform(10.0, 30.0)
        length = min(length, horizon_s - t)
        if length <= 1e-9:
            break
        for _ in range(_DSPEED_DRAWS[kind]):           # the segment's numeric drivers
            rng.random()
        end = round(t + length, 6)
        out.append((round(t, 6), end, kind))
        t = end
    return out


def arm_index_of(arm: str) -> int:
    match = re.search(r"(\d+)$", str(arm))
    if not match:
        raise ValueError(f"dspeed arm without an index: {arm!r}")
    return int(match.group(1))


def same_vehicle_chain(target_mask: np.ndarray, veh_mask: np.ndarray, start: np.ndarray
                       ) -> np.ndarray:
    """``bool[T, K]``: slot still holds the vehicle it held at row 0 (``start`` slots).

    ``target_mask[t, k]`` = the vehicle of slot ``k`` at row ``t`` is in the slot at ``t+1``.
    """
    rows = target_mask.shape[0]
    chain = np.zeros(veh_mask.shape, bool)
    chain[0] = start & veh_mask[0]
    for t in range(1, rows):
        chain[t] = chain[t - 1] & target_mask[t - 1] & veh_mask[t]
    return chain


class DspeedRun:
    """Directed CAVs and designed-segment ticks of one dspeed run (pack rows from tick0)."""

    def __init__(self, world: str, arm: str, slot_vid: dict[int, str],
                 veh_mask: np.ndarray, target_mask: np.ndarray, is_cav0: np.ndarray):
        arm_index = arm_index_of(arm)
        seed = dspeed_seed(world, arm_index)
        roster = {k: v for k, v in slot_vid.items() if bool(is_cav0[k])}
        chosen = set(dspeed_selected(list(roster.values()), seed, arm_index))
        self.selected_slots = sorted(k for k, v in roster.items() if v in chosen)
        rows, k_slots = veh_mask.shape
        start = np.zeros(k_slots, bool)
        start[self.selected_slots] = True
        chain = same_vehicle_chain(target_mask, veh_mask, start)
        window = int(round(DSPEED_WINDOW_S * 2))       # 0.5 s pack ticks
        self.directed = chain.copy()
        self.directed[window:] = False
        self.designed = np.zeros(rows, bool)
        self.n_designed_segments = 0
        for k in self.selected_slots:
            for t0, t1, kind in dspeed_segments(seed, roster[k]):
                if kind not in DESIGNED_SEGMENTS:
                    continue
                a = max(0, int(math.floor(t0 * 2)) - DESIGNED_MARGIN_TICKS)
                b = min(rows, int(math.ceil(t1 * 2)) + DESIGNED_MARGIN_TICKS)
                alive = chain[a:b, k]
                if alive.any():
                    self.designed[a:b] |= alive
                    self.n_designed_segments += 1

    def allowed_starts(self, horizon_steps: int) -> np.ndarray:
        """``bool[rows]``: a window ``[t, t+horizon]`` starting at row ``t`` has no designed tick."""
        rows = self.designed.shape[0]
        csum = np.concatenate(([0], np.cumsum(self.designed.astype(np.int64))))
        t = np.arange(rows)
        hi = np.minimum(rows, t + int(horizon_steps) + 1)
        return (csum[hi] - csum[t]) == 0


# ---------------------------------------------------------------- direct-mix windows ----
def directmix_damping_ticks(world: str, tick0: int, horizon_steps: int, *,
                            veh_mask: np.ndarray, target_mask: np.ndarray, is_cav: np.ndarray,
                            cohort_vid: dict[int, str], recorded_ctrl: np.ndarray,
                            snapshots: Sequence[tuple[int, dict[int, str]]],
                            run_start_tick: int = 0) -> dict:
    """Damping census of one direct-mix window ``[tick0, tick0 + horizon_steps]``.

    ``veh_mask``/``target_mask``/``is_cav`` are pack rows from ``tick0`` that extend PAST the
    horizon (by one snapshot spacing, bounded by the run end) so a post-t0 entrant can be
    followed along its slot's same-vehicle chain to the run's next snapshot item, whose
    identity row names it; ``snapshots`` lists ``(tick, slot->vid)`` of the run's items.
    ``cohort_vid`` maps the t0 slots to vehicle ids; ``recorded_ctrl [control, K]`` is the
    item's recorded timeline (realized owner).  A cohort CAV is damping where the recorded
    label OR the schedule says so; any other CAV where its schedule says so; a CAV whose
    identity is unresolved is ``unknown``.  Returns ``damping``/``unknown`` ``[horizon+1]``
    and bookkeeping counts.
    """
    seed = world_seed(world)
    rows, k_slots = veh_mask.shape
    keep = min(rows, int(horizon_steps) + 1)
    cav = veh_mask & is_cav
    damping = np.zeros(rows, bool)
    unknown = np.zeros(rows, bool)
    sched_cache: dict[str, np.ndarray] = {}

    def sched(vid: str) -> np.ndarray:
        if vid not in sched_cache:
            sched_cache[vid] = scheduled_damping_seconds(seed, vid)
        return sched_cache[vid]

    def seconds(t_rows: np.ndarray) -> np.ndarray:
        return np.clip(((tick0 - run_start_tick + t_rows) // 2), 0, SWITCH_DURATION_S - 1)

    stats = {"cohort_cav_rows": 0, "other_cav_rows": 0, "unknown_cav_rows": 0,
             "recorded_damping_steps": 0, "recorded_damping_not_scheduled": 0,
             "scheduled_damping_recorded_other": 0}
    start = np.zeros(k_slots, bool)
    start[list(cohort_vid)] = True
    chain = same_vehicle_chain(target_mask, veh_mask, start)
    t_all = np.arange(rows)
    in_h = t_all < keep
    for k, vid in cohort_vid.items():
        alive = chain[:, k] & cav[:, k]
        if not alive.any():
            continue
        stats["cohort_cav_rows"] += int((alive & in_h).sum())
        sd = sched(vid)[seconds(t_all)]
        step = np.minimum(t_all // 2, recorded_ctrl.shape[0] - 1)
        rec = (recorded_ctrl[step, k] == 1) & alive & (t_all // 2 < recorded_ctrl.shape[0])
        damping |= (sd & alive) | rec
        on_control = alive & (t_all % 2 == 0) & in_h
        stats["recorded_damping_steps"] += int((rec & on_control).sum())
        stats["recorded_damping_not_scheduled"] += int((rec & ~sd & on_control).sum())
        stats["scheduled_damping_recorded_other"] += int((sd & ~rec & on_control).sum())
    other = cav & ~chain
    snaps = sorted((int(t), ids) for t, ids in snapshots)
    for k in np.flatnonzero(other[:keep].any(axis=0)):
        col = other[:, k]
        t = 0
        while t < keep:
            if not col[t]:
                t += 1
                continue
            t_a = t                                    # one same-vehicle run of the slot
            while t + 1 < rows and col[t + 1] and target_mask[t, k]:
                t += 1
            t_b = t + 1
            vid = next((ids[k] for tick, ids in snaps
                        if tick0 + t_a <= tick <= tick0 + t_b - 1 and k in ids), None)
            seg = np.arange(t_a, t_b)
            n_in = int((seg < keep).sum())
            if vid is None:
                unknown[seg] = True
                stats["unknown_cav_rows"] += n_in
            else:
                stats["other_cav_rows"] += n_in
                damping[seg] |= sched(vid)[seconds(seg)]
            t = t_b
    return {"damping": damping[:keep], "unknown": unknown[:keep], "stats": stats}

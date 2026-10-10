"""Packed per-CAV action schedules applied to the real P3 rollout (contract 2).

This module binds the *packed planned schedule* to the real
``rollout2.step.step_once`` rollout.  It is the scorer-side analogue of the
per-tick action derivation the featurizer/real API already use; it never
invents a new controller.

How training derives the per-step per-CAV action (the code reproduced here):

* The featurizer writes the per-tick per-CAV action token as a one-hot over
  ``features.constants.ACTIONS`` = ``(native, damping, pi, wall, unknown)``
  from the planned timeline, ``features/featurizer.py:308-312``
  (``veh[k, s, C.ACT_NATIVE + a] = 1.0`` with
  ``a = timeline.action_slot(vid, t)``); the per-30 s planned schedule is the
  plan tensor at ``features/featurizer.py:346-362``
  (``plan[di, s, seg, C.PLAN_ACTION_SLOTS + idx] = 1.0``).
* The Idea A training rollout teacher-forces the CAV slots with the recorded
  trajectory built from the packed ``accel_next``
  (``train/idea_a.py:1086-1096`` and ``train/idea_a.py:1900-1977``, the CAV
  action at ``train/idea_a.py:1958``); the correction summary de-scales
  ``cav_plan_ch[..., 0:2]`` (``train/idea_a.py:510-528``).  The packed plan
  channels are therefore the controller's planned schedule/trajectory, never
  an arm label.
* The real rollout turns the planned schedule into per-tick
  ``StrategyActions`` in ``wm_api/engine.py:131-143`` (``_actions_for_tick``):
  each slot defaults to ``native``, is overwritten by its planned action code,
  and non-CAV slots are set to ``none``.  ``actions_for_step`` reproduces that
  decode for the per-tick packed schedule.

Packed schedule layout (``PackedSchedule.plan``, float64 ``[n_steps, K, 8]``),
the same columns as ``features.constants.PLAN_FEATURE_NAMES``
(``features/constants.py:86-97``): ``0:5`` the action one-hot
(native/damping/pi/wall/unknown), ``5`` wall id index, ``6`` wall speed / 40,
``7`` wall hold / 60 s.  ``cav_mask`` marks the t0 CAV slots.  The trajectory
channels of ``features.idea_a.cav_plan_channels`` are NOT an action schedule
and are refused (fail closed).

No native default: a CAV slot with no/unknown/invalid planned action raises
:class:`MissingActionSchedule`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from features.constants import ACTIONS, PLAN_ACTION_SLOTS, PLAN_WALL_SPEED
from rollout2 import baseline
from rollout2.engine import _t0_kin, _world_f64
from rollout2.state import (Rollout2Config, Rollout2State, Rollouts,
                            StrategyActions, WorldState)
from rollout2.step import step_once

DT_S = 0.5

#: Canonical per-CAV action codes (mirror of ``ctrl.config.ACTION_CODES``).
ACTION_CODES = {"none": -1, "native": 0, "damping": 1, "pi": 2, "wall": 3}

#: The action token alphabet carried by the packed schedule (no "none").
_ACTIONS = tuple(str(a) for a in ACTIONS)
_N_ACTION_SLOTS = len(_ACTIONS)
_PLAN_F = 8
_WALL_SPEED_SCALE = 40.0


class MissingActionSchedule(RuntimeError):
    """A packed per-CAV schedule is absent, malformed, or unsafe to apply."""


@dataclass(frozen=True)
class PackedSchedule:
    """A per-tick packed per-CAV action schedule for one run/world.

    ``plan`` is the float64 ``[n_steps, K, 8]`` slice of the packed plan
    channel starting at ``local_t0_tick``; ``cav_mask`` is the boolean ``[K]``
    t0 CAV roster the schedule is validated against.
    """

    run_key: str
    local_t0_tick: int
    n_steps: int
    plan: np.ndarray
    cav_mask: np.ndarray


def _as_action_plan(cav_plan_ch) -> np.ndarray:
    if cav_plan_ch is None:
        raise MissingActionSchedule(
            "packed per-CAV schedule is missing (plan is None)")
    try:
        arr = np.asarray(cav_plan_ch, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MissingActionSchedule(
            f"packed per-CAV schedule is not a numeric array: {exc}") from exc
    if arr.ndim != 3 or arr.shape[2] != _PLAN_F:
        raise MissingActionSchedule(
            "packed per-CAV schedule must be [n_ticks, K, 8] "
            f"(action one-hot 0:5 + wall fields); got {tuple(arr.shape)}")
    return arr


def _validate_cav_rows(plan: np.ndarray, cav_mask: np.ndarray) -> None:
    """Fail closed on a non-one-hot, non-finite, or unknown CAV action."""
    if not cav_mask.any():
        raise MissingActionSchedule(
            "packed per-CAV schedule has no CAV slots to apply")
    rows = plan[:, cav_mask, :]
    if not np.isfinite(rows).all():
        raise MissingActionSchedule(
            "packed per-CAV schedule has a NaN/inf value in a CAV slot")
    action = rows[:, :, PLAN_ACTION_SLOTS:PLAN_ACTION_SLOTS + _N_ACTION_SLOTS]
    onehot = action > 0.5
    counts = onehot.sum(axis=-1)
    if not bool((counts == 1).all()):
        raise MissingActionSchedule(
            "packed per-CAV schedule action columns are not a one-hot over "
            f"{_ACTIONS}; pass the packed plan-schedule channel (action "
            "one-hot), not the cav_plan_ch trajectory channels")


def schedule_from_packed(cav_plan_ch, *, run_key, local_t0_tick, n_steps,
                         is_cav_t0) -> PackedSchedule:
    """Slice and validate a packed per-CAV action schedule (fail closed)."""
    if not isinstance(run_key, str) or not run_key.strip():
        raise MissingActionSchedule("run_key must be a non-empty string")
    try:
        local_t0_tick = int(local_t0_tick)
        n_steps = int(n_steps)
    except (TypeError, ValueError) as exc:
        raise MissingActionSchedule(
            f"local_t0_tick/n_steps must be integers: {exc}") from exc
    if local_t0_tick < 0:
        raise MissingActionSchedule("local_t0_tick must be >= 0")
    if n_steps < 1:
        raise MissingActionSchedule("n_steps must be >= 1")
    arr = _as_action_plan(cav_plan_ch)
    if is_cav_t0 is None:
        raise MissingActionSchedule("is_cav_t0 CAV roster is missing")
    cav_mask = np.asarray(is_cav_t0).astype(bool)
    if cav_mask.ndim != 1 or cav_mask.shape[0] != arr.shape[1]:
        raise MissingActionSchedule(
            f"is_cav_t0 must be [K]={arr.shape[1]}; got {tuple(cav_mask.shape)}")
    end = local_t0_tick + n_steps
    if end > arr.shape[0]:
        raise MissingActionSchedule(
            f"packed schedule is shorter than n_steps: need ticks "
            f"[{local_t0_tick}, {end}) but the channel has {arr.shape[0]}")
    plan = np.ascontiguousarray(
        arr[local_t0_tick:end, :, :], dtype=np.float64)
    _validate_cav_rows(plan, cav_mask)
    return PackedSchedule(run_key=str(run_key), local_t0_tick=local_t0_tick,
                          n_steps=n_steps, plan=plan, cav_mask=cav_mask)


def _codes_for_row(row: np.ndarray, cav_mask: np.ndarray) -> tuple:
    """``(action_code [K], wall_speed [K])`` for one packed plan row."""
    k_n = int(row.shape[0])
    codes = np.full(k_n, ACTION_CODES["none"], dtype=np.int64)
    wall_speed = np.full(k_n, np.nan, dtype=np.float64)
    for slot in np.flatnonzero(cav_mask):
        action = row[slot, PLAN_ACTION_SLOTS:
                     PLAN_ACTION_SLOTS + _N_ACTION_SLOTS]
        token = _ACTIONS[int(np.argmax(action))]
        if token not in ACTION_CODES or token == "unknown":
            raise MissingActionSchedule(
                f"CAV slot {int(slot)} has an unknown planned action {token!r}"
                " (never default to native)")
        code = ACTION_CODES[token]
        codes[slot] = code
        if code == ACTION_CODES["wall"]:
            speed = float(row[slot, PLAN_WALL_SPEED]) * _WALL_SPEED_SCALE
            if not np.isfinite(speed) or speed <= 0.0:
                raise MissingActionSchedule(
                    f"wall CAV slot {int(slot)} has a non-positive/non-finite "
                    f"planned wall speed {speed!r}")
            wall_speed[slot] = speed
    return codes, wall_speed


def actions_for_step(schedule, step, world) -> StrategyActions:
    """Per-tick ``StrategyActions`` decoded from the packed schedule.

    Non-CAV slots always receive ``none``; HDV slots are never given an
    action.  A schedule/plan/mask mismatch raises :class:`MissingActionSchedule`
    instead of silently defaulting to native.
    """
    if schedule is None:
        raise MissingActionSchedule(
            "no packed per-CAV schedule was supplied (refusing the native "
            "default)")
    if not isinstance(schedule, PackedSchedule):
        raise MissingActionSchedule(
            f"schedule must be a PackedSchedule; got {type(schedule)!r}")
    try:
        step = int(step)
    except (TypeError, ValueError) as exc:
        raise MissingActionSchedule(f"step must be an integer: {exc}") from exc
    if step < 0 or step >= schedule.n_steps:
        raise MissingActionSchedule(
            f"step {step} is outside the schedule [0, {schedule.n_steps})")
    if world is None or not hasattr(world, "is_cav"):
        raise MissingActionSchedule("world must be a rollout2 WorldState")
    if int(world.batch) != 1:
        raise MissingActionSchedule(
            f"a packed schedule is single-world; got world batch {world.batch}")
    cav = world.is_cav.detach().to(torch.bool).cpu().numpy().reshape(-1)
    if cav.shape != schedule.cav_mask.shape or \
            not np.array_equal(cav, schedule.cav_mask):
        raise MissingActionSchedule(
            "CAV mask mismatch: the packed schedule roster differs from the "
            "world's is_cav (refusing a shifted action assignment)")
    codes, wall_speed = _codes_for_row(schedule.plan[step], schedule.cav_mask)
    return StrategyActions(
        action_code=torch.from_numpy(codes).reshape(1, -1),
        wall_speed_mps=torch.from_numpy(wall_speed).reshape(1, -1))


def rollout_with_schedule(bundle, world, schedule, *, n_steps, integrator,
                          residual_enabled, n_samples=1, seed=0,
                          t0_s=0.0):
    """Step the real P3 rollout for ``n_steps`` applying the packed schedule.

    ``integrator`` is ``"d1"`` for model A (with the HDV residual) or
    ``"v2g"`` for the Part A baseline (``residual_enabled=False``).  Returns a
    :class:`rollout2.state.Rollouts` carrying ``positions``/``speeds``/
    ``masks``/``accel``/``slot_ids`` plus the frozen ``t0_kin``/``t0_mask``/
    ``t0_is_cav``/``t0_slot_id``/``t0_counter`` channels.
    """
    if schedule is None:
        raise MissingActionSchedule(
            "no packed per-CAV schedule was supplied (refusing the native "
            "default)")
    if not isinstance(schedule, PackedSchedule):
        raise MissingActionSchedule(
            f"schedule must be a PackedSchedule; got {type(schedule)!r}")
    n_steps = int(n_steps)
    if n_steps < 1:
        raise MissingActionSchedule("n_steps must be >= 1")
    if n_steps > schedule.n_steps:
        raise MissingActionSchedule(
            f"schedule is shorter than the requested rollout: {schedule.n_steps}"
            f" < {n_steps}")
    if integrator not in ("d1", "v2g"):
        raise MissingActionSchedule(
            f"integrator must be 'd1' or 'v2g'; got {integrator!r}")
    n_samples = int(n_samples)
    if n_samples < 1:
        raise MissingActionSchedule("n_samples must be >= 1")
    if world is None or not isinstance(world, WorldState):
        raise MissingActionSchedule("world must be a rollout2 WorldState")
    if int(world.batch) != 1:
        raise MissingActionSchedule(
            f"a packed schedule is single-world; got world batch {world.batch}")
    cav = world.is_cav.detach().to(torch.bool).cpu().numpy().reshape(-1)
    if cav.shape != schedule.cav_mask.shape or \
            not np.array_equal(cav, schedule.cav_mask):
        raise MissingActionSchedule(
            "CAV mask mismatch: the packed schedule roster differs from the "
            "world's is_cav")

    b_n, k_n = world.batch, world.slots
    out_dtype = world.s_m.dtype
    cfg0 = Rollout2Config(
        params=None, integrator=str(integrator),
        residual_enabled=bool(residual_enabled), arrivals_enabled=False,
        t0_s=float(t0_s))
    params = baseline.Params(cfg0.params)
    dt = float(params.dt)
    rng = np.random.default_rng(int(seed))
    t0_kin = _t0_kin(world).detach().clone()
    t0_mask = world.present.detach().to(torch.bool).clone()
    t0_is_cav = world.is_cav.detach().to(torch.bool).clone()
    t0_slot_id = world.slot_id.detach().clone()

    samples = []
    for _sample in range(n_samples):
        st = Rollout2State(b_n, k_n)
        st.next_slot_id = world.slot_id.detach().cpu().amax(dim=1) + 1
        cfg = replace(cfg0)
        cfg.state = st
        w = _world_f64(world)
        pos = np.zeros((n_steps + 1, b_n, k_n), dtype=np.float64)
        spd = np.zeros((n_steps + 1, b_n, k_n), dtype=np.float64)
        lns = np.zeros((n_steps + 1, b_n, k_n), dtype=np.int64)
        msk = np.zeros((n_steps + 1, b_n, k_n), dtype=bool)
        ids = np.zeros((n_steps + 1, b_n, k_n), dtype=np.int64)
        acc = np.zeros((n_steps, b_n, k_n), dtype=np.float64)
        cmd = np.full((n_steps, b_n, k_n), np.nan, dtype=np.float64)
        arr = np.zeros((n_steps, b_n, 2), dtype=np.int64)
        ins = np.zeros((n_steps, b_n, 2), dtype=np.int64)
        totals = None

        def _snap(j: int) -> None:
            pos[j] = w.s_m.numpy()
            spd[j] = w.v_mps.numpy()
            lns[j] = w.lane.numpy()
            msk[j] = w.present.numpy()
            ids[j] = w.slot_id.numpy()

        _snap(0)
        for i in range(n_steps):
            acts = actions_for_step(schedule, i, w)
            w, aux = step_once(w, acts, bundle=bundle, config=cfg,
                               t_s=float(t0_s) + i * dt, rng=rng)
            acc[i] = w.a_mps2.numpy()
            cmd[i] = aux.cav_target_speed_mps.numpy()
            arr[i] = getattr(st, "last_arrival_counts",
                             np.zeros((b_n, 2), dtype=np.int64))
            ins[i] = getattr(st, "last_inserted",
                             np.zeros((b_n, 2), dtype=np.int64))
            totals = aux.counts if totals is None else _add_counts(totals,
                                                                   aux.counts)
            _snap(i + 1)
        samples.append(dict(
            pos=pos, spd=spd, lns=lns, msk=msk, ids=ids, acc=acc, cmd=cmd,
            arr=arr, ins=ins, totals=totals,
            collisions=dict(st.collisions_by_step),
            anomalies=dict(st.anomalies)))

    def _stack(key):
        return torch.from_numpy(np.stack([s[key] for s in samples]))

    return Rollouts(
        positions=_stack("pos").to(out_dtype),
        speeds=_stack("spd").to(out_dtype),
        lanes=_stack("lns"),
        masks=_stack("msk"),
        accel=_stack("acc").to(out_dtype),
        arrivals=_stack("arr"),
        insertions=_stack("ins"),
        validity=[s["totals"] for s in samples],
        cav_commands=_stack("cmd").to(out_dtype),
        collisions_by_step=[s["collisions"] for s in samples],
        anomalies=[s["anomalies"] for s in samples],
        t0_kin=t0_kin.to(out_dtype),
        t0_mask=t0_mask,
        t0_counter=None,
        t0_is_cav=t0_is_cav,
        t0_slot_id=t0_slot_id,
        slot_ids=_stack("ids"))


def _add_counts(a, b):
    from model.rollout import ValidityCounts

    out = ValidityCounts(**a.as_dict())
    out.add(b)
    return out

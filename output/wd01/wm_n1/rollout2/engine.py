"""P3 Idea A rollout engine: controller-in-the-loop, strategy-agnostic.

``rollout2(bundle, batch, n_steps=...)`` runs the composite step
(P1 ctrl -> baseline IDM/MOBIL merge + P2 residual -> P4 arrivals -> validity
with the D1 integration) for ``n_steps`` 0.5 s ticks, batched over worlds and
sampleable.  This REPLACES ``model.rollout.rollout`` for Idea A: the v1
rollout moves every vehicle with the learned step decoder and never integrates
position (model/rollout.py:84, 234-265).

K-sample rollouts are deterministic under ``seed`` (one
``numpy.random.default_rng`` seeded once, reused sequentially across
samples, matching the v1 convention).  All internal math is float64;
returned tensors are cast back to the input world's dtype.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from . import baseline
from .state import (Rollouts, Rollout2Config, Rollout2State,
                    StrategyActions, WorldState)
from .step import step_once
from model.rollout import ValidityCounts


@dataclass
class IdeaABundle:
    """The composed Idea A modules (all P2-P5 pieces)."""

    ctrl: object            # ctrl.CavMover (None = no control stage)
    hdv: object             # hdv.HdvResidualNet (None = zero residual)
    arrival: object         # arrival.IdeaAArrivalHead (None = arrivals off)
    corr: object            # corr.CorrectionHead
    cfg: object = None      # train.IdeaATrainConfig or a light namespace
    # A6 (amendment 6): the state-only global-token encoder.  Present when the
    # checkpoint trained with ``state_global_token`` ON; the evaluation
    # outcome paths build the token from the t0 state with these weights.
    state_enc: object = None


def _world_f64(world: WorldState) -> WorldState:
    return WorldState(
        s_m=world.s_m.detach().to(torch.float64).clone(),
        v_mps=world.v_mps.detach().to(torch.float64).clone(),
        a_mps2=world.a_mps2.detach().to(torch.float64).clone(),
        lane=world.lane.detach().to(torch.int64).clone(),
        length_m=world.length_m.detach().to(torch.float64).clone(),
        present=world.present.detach().to(torch.bool).clone(),
        is_cav=world.is_cav.detach().to(torch.bool).clone(),
        wall_group=world.wall_group.detach().to(torch.int64).clone(),
        slot_id=world.slot_id.detach().to(torch.int64).clone(),
        obstacle=(None if world.obstacle is None
                  else world.obstacle.detach().to(torch.bool).clone()),
    )


def _t0_kin(world: WorldState) -> Tensor:
    """The ORIGINAL t0 kinematics ``[B,K,7]`` of the input world.

    Column order mirrors ``hdv.config.KIN_FIELDS`` /
    ``features.idea_a.kinematics_view``: ``(s_down, 3x lane one-hot, v, a,
    length)``.  Read from the P3 input world BEFORE any controller step or
    teacher mutation, so the evaluation state token sees the same t0
    acceleration/length the trainer saw (contrastfix A6 parity).
    """
    one_hot = (world.lane.unsqueeze(-1) == torch.arange(
        3, device=world.lane.device, dtype=world.lane.dtype)).to(
            world.s_m.dtype)
    return torch.stack(
        [world.s_m, one_hot[..., 0], one_hot[..., 1], one_hot[..., 2],
         world.v_mps, world.a_mps2, world.length_m], dim=-1)


class Rollout2Engine:
    """Batched controller-in-the-loop rollout."""

    def __init__(self, bundle: IdeaABundle, config=None, *,
                 device=None, seed: Optional[int] = None) -> None:
        self.bundle = bundle
        self.config = config
        self.device = device
        self.seed = seed

    def run(self, world: WorldState, actions: StrategyActions, *,
            n_steps: int, n_samples: int = 1,
            teacher: Optional[Tensor] = None,
            counter: Optional[Tensor] = None) -> Rollouts:
        """Run ``n_steps`` x 0.5 s.

        ``teacher`` is an optional ``[n_steps+1, B, K, 4]`` recorded CAV
        trajectory ``(s, lane, v, a)``: before step ``i`` the CAV slots
        are replaced by ``teacher[i]`` (teacher forcing, training only).
        """
        cfg0 = self.config if self.config is not None else Rollout2Config()
        params = baseline.Params(cfg0.params)
        dt = params.dt
        b_n, k_n = world.batch, world.slots
        out_dtype = world.s_m.dtype
        if actions is None:
            actions = StrategyActions(
                action_code=torch.where(
                    world.is_cav,
                    torch.zeros_like(world.wall_group),
                    torch.full_like(world.wall_group, -1)))
        rng = np.random.default_rng(0 if self.seed is None else int(self.seed))
        t0 = float(cfg0.t0_s)
        # A6 (contrastfix follow-up): freeze the ORIGINAL t0 state channels
        # from the P3 input world before any step/teacher mutation.  The
        # trajectory tensors cannot recover them (``accel[0]`` is the first
        # FUTURE applied acceleration and the snapshots carry no length), and
        # the evaluation state token must equal the training token built from
        # ``kin``/``kin_mask``/``counter`` of the same t0.
        t0_kin = _t0_kin(world).detach().clone()
        t0_mask = world.present.detach().to(torch.bool).clone()
        t0_is_cav = world.is_cav.detach().to(torch.bool).clone()
        t0_slot_id = world.slot_id.detach().clone()
        t0_counter = None if counter is None else counter.detach().clone()

        samples = []
        for _sample in range(int(n_samples)):
            st = Rollout2State(b_n, k_n)
            # Arrival identities must not collide with the original roster.
            st.next_slot_id = world.slot_id.detach().cpu().amax(dim=1) + 1
            cfg = replace(cfg0)
            cfg.state = st
            cfg.counter = counter
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
            totals = ValidityCounts()

            def _snap(j: int) -> None:
                pos[j] = w.s_m.numpy()
                spd[j] = w.v_mps.numpy()
                lns[j] = w.lane.numpy()
                msk[j] = w.present.numpy()
                ids[j] = w.slot_id.numpy()

            _snap(0)
            for i in range(int(n_steps)):
                teacher_state = None
                if teacher is not None:
                    row = teacher[i]
                    teacher_state = WorldState(
                        s_m=row[..., 0].to(torch.float64),
                        v_mps=row[..., 2].to(torch.float64),
                        a_mps2=row[..., 3].to(torch.float64),
                        lane=row[..., 1].to(torch.int64),
                        length_m=w.length_m.clone(),
                        present=w.present.clone(),
                        is_cav=w.is_cav.clone(),
                        wall_group=w.wall_group.clone(),
                        slot_id=w.slot_id.clone(),
                    )
                w, aux = step_once(
                    w, actions, bundle=self.bundle, config=cfg,
                    t_s=t0 + i * dt, rng=rng, teacher_state=teacher_state)
                totals.add(aux.counts)
                acc[i] = w.a_mps2.numpy()
                cmd[i] = aux.cav_target_speed_mps.numpy()
                arr[i] = getattr(st, "last_arrival_counts",
                                 np.zeros((b_n, 2), dtype=np.int64))
                ins[i] = getattr(st, "last_inserted",
                                 np.zeros((b_n, 2), dtype=np.int64))
                _snap(i + 1)
            samples.append(dict(
                pos=pos, spd=spd, lns=lns, msk=msk, ids=ids, acc=acc, cmd=cmd,
                arr=arr, ins=ins, totals=totals,
                collisions=dict(st.collisions_by_step),
                anomalies=dict(st.anomalies)))

        def _stack(key, dtype=None):
            t = torch.from_numpy(np.stack([s[key] for s in samples]))
            return t if dtype is None else t.to(dtype)

        rollouts = Rollouts(
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
            # A6: the explicit ORIGINAL t0 channels (see _t0_kin above) plus
            # the batch's t0 entrance counter (``[B,2,30]``, or ``None``).
            # Every engine rollout carries all three attributes, so evaluation
            # can tell "channels carried" from a synthetic test double.
            t0_kin=t0_kin.to(out_dtype),
            t0_mask=t0_mask,
            t0_counter=t0_counter,
            t0_is_cav=t0_is_cav,
            t0_slot_id=t0_slot_id,
            slot_ids=_stack("ids"),
        )
        return rollouts


def world_from_batch(batch: dict) -> WorldState:
    """Build a :class:`WorldState` from a packed featurizer batch.

    Accepts ``veh_kin`` ``[B, K, 7]`` (the P9 kinematics-only view) or a
    full ``veh_step`` ``[B, K, 16]`` (the kinematics view is taken via
    :func:`features.idea_a.kinematics_view`), plus optional ``veh_mask``,
    ``is_cav``, ``wall_group`` and ``slot_id``.
    """
    import torch as _torch

    if "veh_kin" in batch:
        kin = batch["veh_kin"].detach().to(torch.float32)
    elif "veh_step" in batch:
        from features.idea_a import kinematics_view

        kin = kinematics_view(batch["veh_step"].detach().to(torch.float32))
    else:
        raise ValueError("batch needs 'veh_kin' or 'veh_step'")
    lane_onehot = kin[..., 1:4]
    has_lane = lane_onehot.sum(-1) > 0
    lane = _torch.where(has_lane, lane_onehot.argmax(-1),
                        _torch.full_like(lane_onehot[..., 0], -1)).long()
    if "veh_mask" in batch:
        present = batch["veh_mask"] > 0
    else:
        present = _torch.ones(kin.shape[:-1], dtype=_torch.bool,
                              device=kin.device)
    if "is_cav" in batch:
        is_cav = batch["is_cav"] > 0
    elif "veh_step" in batch:
        is_cav = batch["veh_step"][..., 7] > 0
    else:
        is_cav = _torch.zeros_like(present)
    b_n, k_n = kin.shape[0], kin.shape[1]
    return WorldState(
        s_m=kin[..., 0].float(),
        v_mps=kin[..., 4].float(),
        a_mps2=kin[..., 5].float(),
        lane=lane,
        length_m=kin[..., 6].float(),
        present=present.bool(),
        is_cav=is_cav.bool(),
        wall_group=batch.get("wall_group", _torch.full(
            (b_n, k_n), -1, dtype=_torch.int64, device=kin.device)),
        slot_id=batch.get("slot_id", _torch.arange(
            k_n, device=kin.device).expand(b_n, k_n).contiguous()),
        obstacle=(None if batch.get("obstacle") is None
                  else (batch["obstacle"] > 0).bool()),
    )


def rollout2(bundle: IdeaABundle, batch: dict, *, n_steps: int,
             n_samples: int = 1, seed: Optional[int] = None,
             teacher: Optional[Tensor] = None,
             config: Optional[Rollout2Config] = None) -> Rollouts:
    """Convenience wrapper: build a :class:`WorldState` from a packed
    batch and run :class:`Rollout2Engine`.

    Batch-level overrides: ``integrator`` ("d1"|"v2g"),
    ``residual_enabled``, ``arrivals_enabled``, ``t0_s``, ``counter``.
    """
    if config is None:
        config = Rollout2Config(
            params=batch.get("params"),
            integrator=batch.get("integrator", "d1"),
            residual_enabled=bool(batch.get(
                "residual_enabled", getattr(bundle, "hdv", None) is not None)),
            arrivals_enabled=bool(batch.get("arrivals_enabled", False)),
            t0_s=float(batch.get("t0_s", 0.0)),
            event_tokens=bool(batch.get("event_tokens", False)),
        )
    world = world_from_batch(batch)
    engine = Rollout2Engine(bundle, config, seed=seed)
    return engine.run(world, None, n_steps=n_steps, n_samples=n_samples,
                      teacher=teacher, counter=batch.get("counter"))

"""V8WM3 ``physics.merge_impatience``: the per-vehicle lane-end waiting time ``merge_wait_s``.

The counter itself (:func:`wms.physics.mobil.update_merge_wait`) is physics; this module gives
its initial value at a rollout start from the observed pre-t0 long driver history and the
causal teacher-forced sequence on recorded states (the same rule the rollout applies to its own
states, so the teacher-forced lane-v5 rows and the rollout see the same feasibility).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from wms.physics.config import PhysicsConfig
from wms.physics.mobil import update_merge_wait


def initial_merge_wait(present: Tensor, kin: Tensor, physics: PhysicsConfig,
                       driver_history: Optional[Tensor], driver_history_mask: Optional[Tensor],
                       stride_steps: int) -> Tensor:
    """Waiting time ``[B, K]`` (s) INCLUDING the t0 state ``kin``.

    Pre-t0 part from the long-history samples strictly before t0 (channels v/30, lane one-hot
    2:5, displacement/250 relative to t0 at index 6; the same conventions as
    :func:`wms.model.lane_v5.initial_acc`): ``+stride_steps * dt`` per waiting sample, held while
    the car stays in the accel lane, reset off it / unseen.  Without a long history every car
    starts at 0.  Then the t0 state is applied (:func:`update_merge_wait`).
    """
    p = physics
    wait = torch.zeros(present.shape, dtype=torch.float32, device=present.device)
    if driver_history is not None and driver_history_mask is not None \
            and driver_history.shape[-1] >= 9 and driver_history.shape[-2] > 1:
        period = float(p.dt_s) * float(stride_steps)
        with torch.no_grad():
            hist = driver_history[..., :-1, :].float()
            seen_h = driver_history_mask[..., :-1].bool()
            s0 = kin[..., 0].detach().float()
            for h in range(hist.shape[-2]):
                seen = seen_h[..., h]
                v = hist[..., h, 0] * 30.0
                lane = hist[..., h, 2:5].argmax(-1)
                s = s0 + hist[..., h, 6] * 250.0
                ramp = seen & (lane == p.merge_lane)
                waiting = (ramp & (s >= p.merge_zone_s1_m - p.merge_impatience_distance_m)
                           & (s <= p.merge_zone_s1_m) & (v <= p.merge_impatience_speed_mps))
                wait = torch.where(ramp, wait + waiting.float() * period, torch.zeros_like(wait))
    return update_merge_wait(wait, kin, present, p)


def recorded_merge_wait(states: Tensor, present: Tensor, physics: PhysicsConfig,
                        wait0: Tensor) -> Tensor:
    """Teacher-forced waiting time ``[B, T, K]`` on recorded states ``[B, T, K, 7]``:
    index 0 is ``wait0`` (:func:`initial_merge_wait`, includes state 0), then causal updates."""
    with torch.no_grad():
        out = [wait0.detach().float()]
        for i in range(1, states.shape[1]):
            out.append(update_merge_wait(out[-1], states[:, i].float(), present[:, i].bool(),
                                         physics))
        return torch.stack(out, 1)

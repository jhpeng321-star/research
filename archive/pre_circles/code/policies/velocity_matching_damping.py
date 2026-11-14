"""Fixed relative-speed feedback with exactly the learned actor's interface.

An engineering comparator inspired by velocity-difference car following;
not an implementation or calibrated reproduction of the full FVD model.
"""

import torch
from torch import nn

from env.native_damping import DampingContract, FRAME_FIELDS


class VelocityMatchingDamping(nn.Module):
    def __init__(self, contract=DampingContract(), response_seconds=5.0):
        super().__init__()
        self.contract = contract
        self.response_seconds = response_seconds
        if response_seconds <= 0:
            raise ValueError("positive velocity response time is required")

    def forward(self, observations):
        if (
            observations.shape[-1] != self.contract.observation_dim
            or not torch.isfinite(observations).all()
        ):
            raise ValueError("invalid local observation")
        frame = observations[..., -len(FRAME_FIELDS) :]
        visible = (frame[..., 9] > 0.5) & (frame[..., 16] > 0.5)
        current = (
            frame[..., 15] * self.contract.history_frames * self.contract.control_dt_s
            <= self.contract.max_age_s
        )
        correction = (-30.0 * frame[..., 8] / self.response_seconds).clamp(
            self.contract.residual_min, self.contract.residual_max
        )
        return torch.where(visible & current, correction, 0.0).unsqueeze(-1)

    def log_prob(self, observations, actions):
        if actions.shape != self(observations).shape:
            raise ValueError("invalid comparator action shape")
        # Collector storage only. This deterministic comparator is never PPO data.
        return torch.zeros(actions.shape[:-1], dtype=actions.dtype, device=actions.device)

    def manifest(self):
        return dict(
            schema="bounded_velocity_matching/1",
            response_seconds=self.response_seconds,
            equation=f"clip((visible_leader_speed-own_speed)/{self.response_seconds:g}, residual_min, residual_max)",
            missing_leader="native_passthrough",
            trained=False,
        )

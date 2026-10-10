"""P6 synthetic event fixtures (pure tensor, no data reads).

These are the test fixtures for the event channels: one stopped obstacle, one
blocked segment and one speed-limit change per world, deterministically placed.
They are implemented (not stubbed) because they define the frozen token layout
that P2/P3/P9 consume, and they must be available without any incident data.
"""
from __future__ import annotations

import torch

from .tokens import (EVENT_F, EVENT_KINDS, EVENT_LOC_SCALE_M, EVENT_MAX,
                     EVENT_SPEED_SCALE_MPS, EVENT_TIME_SCALE_S, EventTokens)


def synthetic_event_batch(*, batch: int = 2, device=None) -> EventTokens:
    """Deterministic ``[B, EVENT_MAX, EVENT_F]`` fixture with 3 live tokens."""
    b, e, f = int(batch), EVENT_MAX, EVENT_F
    tokens = torch.zeros(b, e, f, dtype=torch.float32, device=device)
    mask = torch.zeros(b, e, dtype=torch.bool, device=device)
    for world in range(b):
        # token 0: stopped obstacle on the ramp at 620 m
        tokens[world, 0, EVENT_KINDS.index("stopped_obstacle")] = 1.0
        tokens[world, 0, 3] = 620.0 / EVENT_LOC_SCALE_M
        tokens[world, 0, 4] = 2.0 / 2.0
        tokens[world, 0, 5] = 0.0
        tokens[world, 0, 6] = 100.0 / EVENT_TIME_SCALE_S
        # token 1: blocked middle lane 700..760 m
        tokens[world, 1, EVENT_KINDS.index("blocked_lane_segment")] = 1.0
        tokens[world, 1, 3] = 700.0 / EVENT_LOC_SCALE_M
        tokens[world, 1, 4] = 1.0 / 2.0
        tokens[world, 1, 5] = 60.0 / EVENT_LOC_SCALE_M
        tokens[world, 1, 6] = 120.0 / EVENT_TIME_SCALE_S
        # token 2: speed-limit change to 25 m/s at 300 m on the main-left lane
        tokens[world, 2, EVENT_KINDS.index("speed_limit_change")] = 1.0
        tokens[world, 2, 3] = 300.0 / EVENT_LOC_SCALE_M
        tokens[world, 2, 4] = 0.0
        tokens[world, 2, 5] = 25.0 / EVENT_SPEED_SCALE_MPS
        tokens[world, 2, 6] = 0.0
        mask[world, :3] = True
    return EventTokens(tokens=tokens, mask=mask)

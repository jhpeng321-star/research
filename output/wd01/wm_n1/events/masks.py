"""P6 event masks: token validity and lane-blocking masks.

These are the P3 rollout effect hooks: :func:`blocked_lane_mask` marks lanes a
stopped obstacle / blocked segment removes from service and
:func:`speed_limit_mask` gives the active per-lane speed cap.  With no live
events (or an all-zero mask) both are neutral no-ops: no lane is blocked and
every speed limit is ``NaN``, so :func:`apply_speed_limit` returns its input
unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence, Union

import torch
from torch import Tensor

from .tokens import (EVENT_KINDS, EVENT_LOC_SCALE_M, EVENT_SPEED_SCALE_MPS,
                     EVENT_TIME_SCALE_S, EventTokens, _flatten_events)

EventLike = Union[EventTokens, Sequence[Mapping]]


@dataclass
class EventEffects:
    """Per-world P3 hooks.

    ``blocked``: ``bool [B, n_lanes]``; ``speed_limit``: ``float [B, n_lanes]``
    with ``NaN`` where no limit is active.
    """

    blocked: Tensor
    speed_limit: Tensor


def event_token_mask(tokens: EventTokens) -> Tensor:
    """Valid-token mask ``bool [B, E]``; a zero token is padding."""
    nonempty = tokens.tokens.abs().sum(dim=-1) > 0
    return tokens.mask.to(torch.bool) & nonempty


def _as_event_dicts(events: EventLike) -> List[dict]:
    """Decode :class:`EventTokens` (or normalise raw dicts) to event dicts."""
    if not isinstance(events, EventTokens):
        return _flatten_events(events)
    out: List[dict] = []
    tokens = events.tokens
    mask = event_token_mask(events)
    for world in range(tokens.shape[0]):
        for slot in range(tokens.shape[1]):
            if not bool(mask[world, slot]):
                continue
            tok = tokens[world, slot]
            kind = EVENT_KINDS[int(torch.argmax(tok[:3]).item())]
            rec = {
                "world": world,
                "kind": kind,
                "s_m": float(tok[3]) * EVENT_LOC_SCALE_M,
                "lane": int(round(float(tok[4]) * 2.0)),
                "t_start_s": float(tok[6]) * EVENT_TIME_SCALE_S,
            }
            if kind == "speed_limit_change":
                rec["speed_limit"] = float(tok[5]) * EVENT_SPEED_SCALE_MPS
            else:
                rec["segment_len"] = float(tok[5]) * EVENT_LOC_SCALE_M
            out.append(rec)
    return out


def _lane(rec: Mapping, n_lanes: int) -> int:
    lane = int(rec.get("lane", -1))
    return lane if 0 <= lane < n_lanes else -1


def blocked_lane_mask(events: EventLike, *, batch: int,
                      n_lanes: int = 3) -> Tensor:
    """``bool [B, n_lanes]`` lanes blocked by a stopped obstacle / segment."""
    out = torch.zeros(int(batch), int(n_lanes), dtype=torch.bool)
    for rec in _as_event_dicts(events):
        if rec.get("kind") not in ("stopped_obstacle",
                                   "blocked_lane_segment"):
            continue
        world = int(rec.get("world", 0))
        if not 0 <= world < int(batch):
            continue
        lane = _lane(rec, int(n_lanes))
        if lane >= 0:
            out[world, lane] = True
    return out


def speed_limit_mask(events: EventLike, *, batch: int,
                     n_lanes: int = 3) -> Tensor:
    """``float [B, n_lanes]`` active speed limit (m/s, NaN = none)."""
    out = torch.full((int(batch), int(n_lanes)), float("nan"))
    for rec in _as_event_dicts(events):
        if rec.get("kind") != "speed_limit_change":
            continue
        world = int(rec.get("world", 0))
        if not 0 <= world < int(batch):
            continue
        lane = _lane(rec, int(n_lanes))
        if lane < 0:
            continue
        value = rec.get("speed_limit")
        if value is None:
            value = rec.get("speed_delta", rec.get("magnitude"))
        if value is not None:
            out[world, lane] = float(value)
    return out


def event_effects(events: EventLike, *, batch: int,
                  n_lanes: int = 3) -> EventEffects:
    """Both P3 event hooks at once (blocked lanes + speed limits)."""
    return EventEffects(
        blocked=blocked_lane_mask(events, batch=batch, n_lanes=n_lanes),
        speed_limit=speed_limit_mask(events, batch=batch, n_lanes=n_lanes),
    )


def apply_speed_limit(speeds: Tensor, speed_limit: Tensor) -> Tensor:
    """Clamp ``speeds`` to a per-lane ``speed_limit``; ``NaN`` is a no-op."""
    limit = torch.as_tensor(speed_limit, dtype=speeds.dtype,
                            device=speeds.device)
    finite = torch.isfinite(limit)
    capped = torch.minimum(speeds, limit)
    return torch.where(finite, capped, speeds)

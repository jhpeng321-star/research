"""P6 event tokens: stopped obstacles, blocked lane segments, speed-limit
changes (Idea A "event-ready inputs").

The current data does not use them; they are supported and tested with
synthetic fixtures.  An incident campaign is a later, separately registered
follow-up.

Token layout (``EVENT_F = 8``):

=====  =====================================================
index  field
=====  =====================================================
0      stopped_obstacle one-hot
1      blocked_lane_segment one-hot
2      speed_limit_change one-hot
3      s_down_m / 1000
4      lane index / 2  (-1 maps to 0 and stays masked)
5      magnitude: speed delta / 40, or segment length / 1000
6      t_start_s / 600
7      reserved (0)
=====  =====================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

EVENT_KINDS = ("stopped_obstacle", "blocked_lane_segment",
               "speed_limit_change")
EVENT_F = 8
EVENT_MAX = 8
EVENT_LOC_SCALE_M = 1000.0
EVENT_SPEED_SCALE_MPS = 40.0
EVENT_TIME_SCALE_S = 600.0


@dataclass
class EventTokens:
    """Batched event tokens + mask.

    ``tokens``: ``float [B, E, EVENT_F]``; ``mask``: ``bool [B, E]``.
    """

    tokens: Tensor
    mask: Tensor

    @property
    def batch(self) -> int:
        return int(self.tokens.shape[0])


def _lookup(ev: Mapping, *names, default=0.0) -> float:
    for name in names:
        if ev.get(name) is not None:
            return float(ev[name])
    return float(default)


def _flatten_events(events: Sequence) -> list:
    """Normalise either a flat list of dicts or a list-of-lists by world.

    A nested ``events[w]`` sequence is indexed by world; a flat sequence uses
    each dict's optional ``world`` / ``batch_index`` key (default 0).
    """
    if len(events) > 0 and not isinstance(events[0], Mapping):
        flat = []
        for world, group in enumerate(events):
            for ev in group:
                rec = dict(ev)
                rec.setdefault("world", world)
                flat.append(rec)
        return flat
    return [dict(ev) for ev in events]


def encode_events(events: Sequence[Mapping], *, batch: int = 1,
                  device=None) -> EventTokens:
    """``events`` (list of ``{kind, t_start_s, lane, s_m, magnitude}`` dicts
    grouped by world) -> :class:`EventTokens`.

    Each mapping carries ``kind`` in :data:`EVENT_KINDS`; ``s_m`` (m),
    ``lane`` (-1 maps to 0 and stays masked), ``t_start_s`` (s) and a kind
    magnitude: for ``speed_limit_change`` an absolute new limit in m/s
    (keys ``speed_limit`` / ``speed_delta`` / ``magnitude``); otherwise a
    segment length in m (keys ``segment_len`` / ``length_m`` / ``magnitude``).
    A flat list assigns events to their ``world`` key (default 0); a nested
    list-of-lists is indexed by world.  More than :data:`EVENT_MAX` live
    events for one world raises ``ValueError``.
    """
    batch = int(batch)
    tokens = torch.zeros(batch, EVENT_MAX, EVENT_F, dtype=torch.float32,
                         device=device)
    mask = torch.zeros(batch, EVENT_MAX, dtype=torch.bool, device=device)
    cursor = [0] * batch
    for ev in _flatten_events(events):
        kind = ev.get("kind")
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r}; expected one of "
                             f"{EVENT_KINDS}")
        world = int(ev.get("world", ev.get("batch_index", 0)))
        if not 0 <= world < batch:
            raise ValueError(f"event world {world} out of range for batch "
                             f"{batch}")
        slot = cursor[world]
        if slot >= EVENT_MAX:
            raise ValueError(
                f"world {world} has more than EVENT_MAX={EVENT_MAX} events")
        tokens[world, slot, EVENT_KINDS.index(kind)] = 1.0
        tokens[world, slot, 3] = _lookup(ev, "s_m") / EVENT_LOC_SCALE_M
        lane = int(ev.get("lane", 0))
        if lane < 0:
            lane = 0
        tokens[world, slot, 4] = lane / 2.0
        if kind == "speed_limit_change":
            tokens[world, slot, 5] = _lookup(
                ev, "speed_limit", "speed_delta", "magnitude") \
                / EVENT_SPEED_SCALE_MPS
        else:
            tokens[world, slot, 5] = _lookup(
                ev, "segment_len", "length_m", "magnitude") \
                / EVENT_LOC_SCALE_M
        tokens[world, slot, 6] = _lookup(ev, "t_start_s") \
            / EVENT_TIME_SCALE_S
        mask[world, slot] = True
        cursor[world] += 1
    return EventTokens(tokens=tokens, mask=mask)

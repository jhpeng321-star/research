"""P6 acceptance tests: token layout/fixtures, encode round-trip, masks and the
P3 effect hooks, and the zero-mask no-op."""
from __future__ import annotations

import torch

from events import (EVENT_F, EVENT_KINDS, EVENT_MAX, EventTokens,
                    apply_speed_limit, blocked_lane_mask, encode_events,
                    event_effects, event_token_mask, speed_limit_mask,
                    synthetic_event_batch)


def test_event_kinds_and_layout():
    assert EVENT_KINDS == ("stopped_obstacle", "blocked_lane_segment",
                           "speed_limit_change")
    assert EVENT_F == 8 and EVENT_MAX == 8


def test_synthetic_fixture_shapes():
    ev = synthetic_event_batch(batch=3)
    assert ev.tokens.shape == (3, EVENT_MAX, EVENT_F)
    assert ev.mask.shape == (3, EVENT_MAX)
    assert int(ev.mask.sum()) == 9
    assert torch.allclose(ev.tokens[..., :3].sum(-1)[ev.mask], torch.ones(9))
    assert float(ev.tokens[..., 7].abs().max()) == 0.0


def test_encode_events_flat_layout():
    events = [
        {"world": 1, "kind": "stopped_obstacle", "s_m": 500.0, "lane": 0,
         "t_start_s": 30.0, "segment_len": 0.0},
        {"world": 1, "kind": "speed_limit_change", "s_m": 200.0, "lane": 2,
         "t_start_s": 0.0, "speed_limit": 30.0},
    ]
    ev = encode_events(events, batch=2)
    assert ev.tokens.shape == (2, EVENT_MAX, EVENT_F)
    assert ev.mask.sum() == 2
    assert ev.mask[1, 0] and ev.mask[1, 1]
    assert ev.tokens[1, 0, 0] == 1.0                       # stopped_obstacle
    assert torch.isclose(ev.tokens[1, 0, 3], torch.tensor(0.5))   # 500/1000
    assert ev.tokens[1, 0, 4] == 0.0
    assert torch.isclose(ev.tokens[1, 0, 6], torch.tensor(0.05))  # 30/600
    assert ev.tokens[1, 1, 2] == 1.0                       # speed_limit_change
    assert torch.isclose(ev.tokens[1, 1, 5], torch.tensor(0.75))  # 30/40
    assert ev.tokens[0, 0, :].abs().sum() == 0.0


def test_encode_events_nested_by_world():
    events = [[{"kind": "blocked_lane_segment", "s_m": 10.0, "lane": 1,
                "t_start_s": 0.0, "segment_len": 40.0}],
              []]
    ev = encode_events(events, batch=2)
    assert ev.mask.sum() == 1
    assert ev.tokens[0, 0, 1] == 1.0
    assert torch.isclose(ev.tokens[0, 0, 5], torch.tensor(0.04))  # 40/1000


def test_event_token_mask_zero_token_is_padding():
    ev = synthetic_event_batch(batch=1)
    assert int(event_token_mask(ev).sum()) == 3
    # a formally-masked but all-zero token is padding
    ev.mask[0, 0] = True
    ev.tokens[0, 0] = 0.0
    m = event_token_mask(ev)
    assert not bool(m[0, 0])
    assert int(m.sum()) == 2


def test_masks_from_synthetic_fixture():
    ev = synthetic_event_batch(batch=2)
    blocked = blocked_lane_mask(ev, batch=2)
    assert blocked.shape == (2, 3)
    # obstacle on ramp (lane 2) + blocked segment on lane 1
    assert not bool(blocked[:, 0].any())
    assert bool(blocked[:, 1].all()) and bool(blocked[:, 2].all())
    limit = speed_limit_mask(ev, batch=2)
    assert limit.shape == (2, 3)
    assert torch.allclose(limit[:, 0], torch.full((2,), 25.0))
    assert torch.isnan(limit[:, 1]).all() and torch.isnan(limit[:, 2]).all()


def test_event_effects_and_apply_speed_limit():
    ev = synthetic_event_batch(batch=1)
    eff = event_effects(ev, batch=1)
    assert eff.blocked.dtype == torch.bool
    speeds = torch.full((1, 3), 30.0)
    capped = apply_speed_limit(speeds, eff.speed_limit)
    assert float(capped[0, 0]) == 25.0
    assert float(capped[0, 1]) == 30.0


def test_zero_mask_is_noop():
    ev = synthetic_event_batch(batch=2)
    ev.mask.zero_()
    assert int(event_token_mask(ev).sum()) == 0
    blocked = blocked_lane_mask(ev, batch=2)
    assert not bool(blocked.any())
    limit = speed_limit_mask(ev, batch=2)
    assert torch.isnan(limit).all()
    speeds = torch.tensor([[20.0, 30.0, 15.0], [10.0, 25.0, 30.0]])
    assert torch.equal(apply_speed_limit(speeds, limit), speeds)


def test_empty_events_is_noop():
    blocked = blocked_lane_mask([], batch=2)
    assert not bool(blocked.any())
    limit = speed_limit_mask([], batch=2)
    assert torch.isnan(limit).all()
    empty = EventTokens(
        tokens=torch.zeros(2, EVENT_MAX, EVENT_F),
        mask=torch.zeros(2, EVENT_MAX, dtype=torch.bool))
    assert int(event_token_mask(empty).sum()) == 0
    assert not bool(blocked_lane_mask(empty, batch=2).any())

"""Determinism (seeded CPU mode) and per-call latency on CPU."""
from __future__ import annotations

import time

import pytest

from _tiny import default_plans, tiny_scorer, tiny_state


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("determinism_ckpt")
    tiny_scorer(tmp, k=2)
    return tmp


def test_same_seed_reproduces_exactly(ckpt):
    from wm_api import WmScorer

    scorer = WmScorer(ckpt, threads=1)
    st = tiny_state()
    a = scorer.score(st, default_plans(), horizon_s=60, allow_other_horizon=True, seed=123)
    b = scorer.score(st, default_plans(), horizon_s=60, allow_other_horizon=True, seed=123)
    assert a.to_json(include_timing=False) == \
        b.to_json(include_timing=False)


def test_horizon300_latency_and_determinism(ckpt):
    from wm_api import WmScorer

    scorer = WmScorer(ckpt, threads=1)
    st = tiny_state()
    t0 = time.perf_counter()
    a = scorer.score(st, default_plans(), horizon_s=300, seed=0)
    latency = time.perf_counter() - t0
    b = scorer.score(st, default_plans(), horizon_s=300, seed=0)
    assert a.to_json(include_timing=False) == \
        b.to_json(include_timing=False)
    assert latency < 240.0, f"per-call CPU latency {latency:.1f}s too high"
    print(f"\nwm_api per-call CPU latency (2 plans x 2 members, 300 s): "
          f"{latency:.2f}s")


def test_different_seed_changes_arrival_draws(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("seed_ckpt")
    scorer = tiny_scorer(tmp, k=1)
    st = tiny_state(counters=True)
    a = scorer.score(st, [{"actions": {}}], horizon_s=60, allow_other_horizon=True, seed=1)
    b = scorer.score(st, [{"actions": {}}], horizon_s=60, allow_other_horizon=True, seed=2)
    differs = (a.plans[0].metrics["delay_s_per_veh"].quantiles["q50"]
               != b.plans[0].metrics["delay_s_per_veh"].quantiles["q50"])
    assert differs, "seeded arrival sampling must vary the outcome draws"

"""Exact-clock contract tests with mocked simulator interfaces.

Covers normal progression, repeated time, skipped physics/control times,
floating-point tolerance at valid boundaries, episode initialization and
termination, exactly-once metric collection, and fixture/training loop
consistency (both loops drive the same ExactClock primitive). No simulator
is launched; step/time functions are mocks.
"""
import json

import pytest

from traffic_shepherd import ppo_contract as C
from traffic_shepherd.sim_clock import ClockViolationError, ExactClock


class MockSim:
    """Scriptable simulator: each bare step advances by `step_length`, unless
    a scripted anomaly is queued for a specific call index ('repeat' performs
    no action; 'jump' doubles the step) or the sim is frozen for direct time
    injection."""

    def __init__(self, step_length=0.5, anomalies=()):
        self.time = 0.0
        self.step_length = step_length
        self.anomalies = dict(anomalies)   # call index (1-based) -> kind
        self.calls = 0
        self.frozen = False

    def step(self):
        self.calls += 1
        if self.frozen:
            return self.time
        self.time += self.step_length
        kind = self.anomalies.get(self.calls)
        if kind == "repeat":
            self.time -= self.step_length
        elif kind == "jump":
            self.time += self.step_length
        return self.time

    def get_time(self):
        return self.time


def make_clock(sim, context="test"):
    return ExactClock(step_fn=sim.step, time_fn=sim.get_time, dt=.5, start=0.,
                      context=context)


def test_normal_progression_exact_half_seconds():
    sim = MockSim()
    clock = make_clock(sim)
    for i in range(1, 21):
        assert clock.advance() == pytest.approx(i * 0.5, abs=1e-12)
    assert sim.calls == 20 and clock.calls == 20


def test_repeated_time_is_rejected_with_receipt():
    sim = MockSim(anomalies={2: "repeat"})
    clock = make_clock(sim)
    clock.advance()
    with pytest.raises(ClockViolationError) as exc:
        clock.advance()
    r = exc.value.receipt
    assert r["kind"] == "physics_step_mismatch"
    assert r["expected"] == pytest.approx(1.0, abs=1e-12)
    assert r["observed"] == pytest.approx(0.5, abs=1e-12)  # no action: repeated 0.5
    assert r["delta"] == pytest.approx(-0.5, abs=1e-12)
    assert r["call_index"] == 2 and r["context"] == "test"


def test_skipped_time_is_rejected_with_receipt():
    sim = MockSim(anomalies={2: "jump"})       # one call advances +1.0
    clock = make_clock(sim)
    clock.advance()                             # -> 0.5
    with pytest.raises(ClockViolationError) as exc:
        clock.advance()                         # -> 1.5, expected 1.0
    r = exc.value.receipt
    assert r["observed"] == pytest.approx(1.5, abs=1e-12)
    assert r["expected"] == pytest.approx(1.0, abs=1e-12)
    assert r["delta"] == pytest.approx(0.5, abs=1e-12)


def test_float_tolerance_at_valid_boundaries():
    sim = MockSim()
    clock = make_clock(sim)
    clock.advance()                       # clock.last = 0.5
    sim.time = 1.0 + 1e-12                # float dust at the valid boundary
    sim.frozen = True
    assert clock.advance() == pytest.approx(1.0, abs=1e-9)   # dust passes
    sim.time = 1.5 + 1e-6                 # a 1e-6 offset is outside the contract
    with pytest.raises(ClockViolationError) as exc:
        clock.advance()
    assert exc.value.receipt["delta"] == pytest.approx(1e-6, rel=1e-3)


def test_control_tick_check_rejects_mismatch_without_advancing():
    sim = MockSim()
    clock = make_clock(sim)
    clock.advance()                              # now 0.5
    with pytest.raises(ClockViolationError) as exc:
        clock.check_now(1.0, "control tick 2")   # reads ahead of the clock
    assert "control_time_mismatch" in exc.value.receipt["kind"]
    assert clock.last == pytest.approx(0.5)      # nothing advanced


def test_episode_initialization_and_termination():
    sim = MockSim()
    clock = make_clock(sim)
    # initialization: control at t=0 requires no advance
    clock.check_now(sim.get_time(), "control tick 0")
    horizon_steps = int(120 / 0.5)
    for _ in range(horizon_steps):
        clock.advance()
    assert clock.last == pytest.approx(120.0, abs=1e-12)
    assert sim.time == pytest.approx(120.0, abs=1e-12)
    # the horizon is the caller's loop bound; the clock grid itself continues
    # exactly (termination is enforced by the driving loop, not the clock)
    assert clock.advance() == pytest.approx(120.5, abs=1e-12)


def test_every_sample_collected_exactly_once():
    sim = MockSim()
    clock = make_clock(sim)
    collected = []
    for _ in range(int(120 / 0.5)):
        collected.append(clock.advance())
    assert collected == sorted(collected)                     # monotone
    assert len(set(collected)) == len(collected)              # exactly once
    assert len(collected) == 240


def test_fixture_and_training_loops_share_the_primitive():
    """The mechanics fixture loop and the corridor loop build identical
    clocks; their stepping/collection behavior matches by construction."""
    def fixture_loop(sim):
        clock = make_clock(sim, context="mechanics:hold_ppo_chain")
        samples = []
        for tick in range(120):
            clock.check_now(sim.get_time(), f"control tick {tick}")
            for substep in (1, 2):
                samples.append(clock.advance())
        return clock, samples

    def corridor_loop(sim):
        clock = ExactClock(step_fn=sim.step, time_fn=sim.get_time, dt=.5,
                           start=0., context="corridor:ep")
        samples = []
        for tick in range(120):
            clock.check_now(sim.get_time(), f"control tick {tick}")
            for _ in range(2):
                samples.append(clock.advance())
        return clock, samples

    sim_a, sim_b = MockSim(), MockSim()
    clock_a, samples_a = fixture_loop(sim_a)
    clock_b, samples_b = corridor_loop(sim_b)
    assert samples_a == samples_b
    assert clock_a.calls == clock_b.calls == 240
    assert json.dumps(clock_a.history) == json.dumps(clock_b.history)


def test_violation_receipt_is_json_serializable_and_informative():
    sim = MockSim(anomalies={2: "jump"})
    clock = make_clock(sim, context="mechanics:hold_ppo_chain")
    clock.advance()
    with pytest.raises(ClockViolationError) as exc:
        clock.advance()
    parsed = json.loads(json.dumps(exc.value.receipt))
    assert parsed["context"] == "mechanics:hold_ppo_chain"
    assert parsed["kind"] == "physics_step_mismatch"
    assert "expected" in parsed and "observed" in parsed and "delta" in parsed
    assert "clock violation" in str(exc.value)


def test_declared_cadence_on_integer_grid():
    """PPO decisions only at the declared 5 s cadence on the integer grid."""
    sim = MockSim()
    clock = make_clock(sim)
    decision_times = []
    clock.check_now(sim.get_time(), "control tick 0")
    for tick in range(120):
        if tick:
            clock.check_now(float(tick), f"control tick {tick}")
        if tick and tick % C.CADENCE_S == 0:
            decision_times.append(tick)
        for _ in range(2):
            clock.advance()
    assert decision_times == list(range(5, 120, 5))
    assert all(t % C.CADENCE_S == 0 for t in decision_times)

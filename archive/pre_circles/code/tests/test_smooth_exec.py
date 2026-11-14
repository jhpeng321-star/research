"""Tests for eval-time smooth execution + looking-behind reference
(literature probe, 2026-08-06; Stern-style smooth profiles / LCC
"looking-behind" topology).

The mode DECISION stays discrete; execution is overridden at the ``_cmd_speed``
choke point so every champion mode shares the same profile.
"""

from tests.test_shepherd_actions import _make_engine


def _engine(**overrides):
    mc = {
        "smooth_exec": True,
        "smooth_a_max_dec": 0.4,
        "smooth_a_max_acc": 0.4,
        "smooth_deadband": 0.3,
        "smooth_ema_alpha": 0.3,
        "look_behind_exec": False,
        "behind_near": 150.0,
        "behind_far": 200.0,
        "behind_ema_alpha": 0.3,
    }
    mc.update(overrides)
    return _make_engine(mode_config=mc)


def test_off_passthrough():
    # smooth_exec off -> _exec_cmd not used; _exec_prev stays empty.
    engine = _engine(smooth_exec=False)
    engine._exec_prev["cav0"] = 99.0
    assert engine._exec_prev["cav0"] == 99.0  # no auto-clearing in _exec_cmd
    # the mode path clears it via _cmd_speed; here we just check default off
    assert engine.smooth_exec is False
    assert engine.look_behind_exec is False


def test_smooth_ramp_slower_than_mode_limit():
    # mode limit is 1.0 m/s2; smooth limit is 0.4 m/s2. From 20 toward 10:
    # smooth ramp should drop by <= 0.4 per step (dt=1), not 1.0.
    engine = _engine()
    v = engine._exec_cmd("cav0", 20.0, 10.0)
    # 0.3*10 + 0.7*20 = 16 (EMA ref), rate-limit 20 -> 16 at 0.4 => 19.6
    assert abs(v - 19.6) < 1e-9


def test_smooth_ramp_converges_toward_target():
    engine = _engine()
    prev = 20.0
    for _ in range(60):
        prev = engine._exec_cmd("cav0", prev, 10.0)
    assert prev < 10.5  # has effectively reached the target
    assert prev > 9.5


def test_look_behind_uses_upstream_speed():
    # With look_behind on, the reference is the behind-band speed, not the
    # mode target. Simulate a slow upstream band (8.0) vs a high mode target.
    engine = _engine(look_behind_exec=True, smooth_a_max_acc=0.4)
    engine._behind_band_mean_speed = lambda cav: 8.0
    v = engine._exec_cmd("cav0", 20.0, 30.0)
    # EMA ref = 0.3*8 + 0.7*20 = 16.4; rate-limit 20 -> 16.4 at 0.4 => 20-0.4=19.6
    assert abs(v - 19.6) < 1e-9


def test_look_behind_no_info_falls_back_to_mode_target():
    engine = _engine(look_behind_exec=True)
    engine._behind_band_mean_speed = lambda cav: None
    v = engine._exec_cmd("cav0", 20.0, 30.0)
    # EMA ref = 0.3*30 + 0.7*20 = 23; rate-limit 20->23 at 0.4 => 20.4
    assert abs(v - 20.4) < 1e-9


def test_exec_prev_state_persists_across_steps():
    engine = _engine()
    v1 = engine._exec_cmd("cav0", 20.0, 10.0)
    assert engine._exec_prev["cav0"] is not None
    v2 = engine._exec_cmd("cav0", v1, 10.0)
    # EMA continues from stored prev, so the second ramp is gentler than the
    # mode's step limit of 1.0.
    assert abs(v2 - v1) <= 0.4 + 1e-9

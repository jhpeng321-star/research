"""IDM unit tests vs hand-computed Treiber values."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

WD01 = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WD01))

from partA_v2g.engine import (GAP_MAX, idm_accel, load_params,
                              segment_limits)

P = load_params()
IDM = P["idm"]


def idm_hand(v, gap, dv, v0):
    """Independent hand computation of the Treiber IDM."""
    a, b = 1.0, 1.5
    T, s0, delta = 1.5, 2.0, 4
    s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a * b)))
    return a * (1.0 - (v / v0) ** delta - (s_star / gap) ** 2)


def test_hand_computed_following():
    v, gap, dv, v0 = 25.0, 40.0, 2.0, 31.29
    got = float(idm_accel(np.array([v]), np.array([gap]), np.array([dv]),
                          np.array([v0]), IDM)[0])
    assert got == pytest.approx(idm_hand(v, gap, dv, v0), rel=1e-12)


def test_hand_computed_free_road():
    """Free road in the IDM sense: a large but finite gap keeps a tiny
    interaction term -- the hand formula includes it exactly."""
    v, v0 = 20.0, 31.29
    s_star = 2.0 + v * 1.5
    got = float(idm_accel(np.array([v]), np.array([GAP_MAX]),
                          np.array([0.0]), np.array([v0]), IDM)[0])
    assert got == pytest.approx(
        1.0 * (1.0 - (v / v0) ** 4 - (s_star / GAP_MAX) ** 2), rel=1e-12)
    assert got > 0.0


def test_hand_computed_stationary_equilibrium():
    """Equilibrium behind a same-speed leader: a = 0 at
    g = s*/sqrt(1 - (v/v0)^delta)."""
    v = 20.0
    s_star = 2.0 + v * 1.5
    g_eq = s_star / math.sqrt(1.0 - (v / 31.29) ** 4)
    got = float(idm_accel(np.array([v]), np.array([g_eq]),
                          np.array([0.0]), np.array([31.29]), IDM)[0])
    assert got == pytest.approx(0.0, abs=1e-12)


def test_closing_harder_than_opening():
    v, gap, v0 = 25.0, 40.0, 31.29
    close = float(idm_accel(np.array([v]), np.array([gap]),
                            np.array([4.0]), np.array([v0]), IDM)[0])
    open_ = float(idm_accel(np.array([v]), np.array([gap]),
                            np.array([-4.0]), np.array([v0]), IDM)[0])
    assert close < open_


def test_monotone_in_gap_and_speed():
    for dv in (-2.0, 0.0, 3.0):
        prev = None
        for gap in (5.0, 10.0, 25.0, 60.0, 150.0):
            a = float(idm_accel(np.array([25.0]), np.array([gap]),
                                np.array([dv]), np.array([31.29]), IDM)[0])
            if prev is not None:
                assert a > prev
            prev = a
    prev = None
    for v in (5.0, 15.0, 25.0, 31.0):
        a = float(idm_accel(np.array([v]), np.array([60.0]),
                            np.array([0.0]), np.array([31.29]), IDM)[0])
        if prev is not None:
            assert a < prev
        prev = a


def test_v0_is_the_segment_limit():
    # lane 2: ramp 22.22 below s=378.66, merge segment 31.29 above
    lane = np.array([2, 2, 0, 1])
    s = np.array([100.0, 400.0, 2000.0, 2000.0])
    lim = segment_limits(lane, s, P)
    assert lim[0] == pytest.approx(22.22)
    assert lim[1] == pytest.approx(31.29)
    assert lim[2] == pytest.approx(31.29)
    assert lim[3] == pytest.approx(31.29)


def test_gap_floor_never_nan():
    a = idm_accel(np.array([10.0]), np.array([0.0]), np.array([5.0]),
                  np.array([31.29]), IDM)
    assert np.isfinite(a).all()
    assert a[0] < -5.0      # strong emergency braking, clipped by the
    # engine's a_clip at -9, but the pure term must be finite

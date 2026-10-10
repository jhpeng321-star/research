import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "output" / "wd01" / "wm_n1"
sys.path.insert(0, str(PKG))

from wms_eval import hetdata_shift as H  # noqa: E402


def _counts(n_ticks, busy_from, busy_until=None, value=5):
    counts = np.zeros(n_ticks, dtype=np.int64)
    busy_until = n_ticks if busy_until is None else busy_until
    counts[busy_from:busy_until] = value
    return counts


def test_selects_first_busy_tick_at_or_after_120s():
    counts = _counts(320, busy_from=250)
    tick, reason = H.select_shift_tick(
        counts, 0, 320, shift_to_t0_s=120.0, min_t0_s=120.0, margin_steps=10,
        history_steps=10, horizon_steps=0)
    assert tick == 250
    assert reason == "ok"
    assert (tick - 0) * H.DT_S == 125.0          # >= 120 s into the window


def test_window_too_short_fails_closed():
    counts = _counts(120, busy_from=0)
    tick, reason = H.select_shift_tick(
        counts, 0, 120, shift_to_t0_s=120.0, min_t0_s=120.0, margin_steps=10,
        history_steps=10, horizon_steps=0)
    assert tick is None
    assert "too short" in reason


def test_no_busy_tick_fails_closed():
    counts = _counts(320, busy_from=0, value=0)
    tick, reason = H.select_shift_tick(
        counts, 0, 320, shift_to_t0_s=120.0, min_t0_s=120.0, margin_steps=10,
        history_steps=10, horizon_steps=0)
    assert tick is None
    assert "no busy tick" in reason


def test_horizon_reserves_room_inside_the_window():
    counts = _counts(400, busy_from=0)
    tick, reason = H.select_shift_tick(
        counts, 0, 300, shift_to_t0_s=120.0, min_t0_s=120.0, margin_steps=10,
        history_steps=10, horizon_steps=240)
    # upper = 300 - 1 - 240 = 59 < start 250 -> refused
    assert tick is None
    tick, reason = H.select_shift_tick(
        counts, 0, 600, shift_to_t0_s=120.0, min_t0_s=120.0, margin_steps=10,
        history_steps=10, horizon_steps=240)
    assert tick == 250
    assert tick + 240 <= 600 - 1


def test_history_never_crosses_the_run_start():
    counts = _counts(320, busy_from=0)
    tick, _ = H.select_shift_tick(
        counts, 0, 320, shift_to_t0_s=0.0, min_t0_s=0.0, margin_steps=0,
        history_steps=10, horizon_steps=0)
    assert tick == 9                              # start = max(0+0, 0+9)
    assert max(0, tick - 10 + 1) == 0

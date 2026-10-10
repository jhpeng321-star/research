from __future__ import annotations

import pytest

from output.wd01.wm_n1.wms.check400 import DRIVER_PARAM_BOUNDS, _check_driver_params


def _row(**overrides):
    stats = {name: {"min": lo, "median": lo, "max": lo}
             for name, (lo, hi) in DRIVER_PARAM_BOUNDS.items()}
    row = {"step": 1, "driver_param_stats": stats, "valid_hdv_slots_t0": 4}
    row.update(overrides)
    return row


def test_driver_param_gate_accepts_in_bounds():
    _check_driver_params([_row()])


def test_driver_param_gate_rejects_out_of_bounds():
    row = _row()
    row["driver_param_stats"]["accel"]["max"] = 99.0
    with pytest.raises(SystemExit):
        _check_driver_params([row])


def test_driver_param_gate_rejects_missing_stats():
    with pytest.raises(SystemExit):
        _check_driver_params([{"step": 1, "valid_hdv_slots_t0": 4}])


def test_driver_param_gate_rejects_no_hdv_slots():
    with pytest.raises(SystemExit):
        _check_driver_params([_row(valid_hdv_slots_t0=0)])

"""The PLANNED-SCHEDULE action descriptor (registration s.12).

Core owner requirement: the action input is the planned schedule, not
the arm name -- "damp 60 s then native" != "damp 570 s" (MUST be
distinguishable inputs)."""
from __future__ import annotations

import numpy as np
import pytest

from partB_v04b import reg
from partB_v04b import schedule_desc as SD


def _damp_then_native(t_damp: float, t0: float = 0.0) -> dict:
    return {"cav0": [(t0, t0 + t_damp, "damping"),
                     (t0 + t_damp, t0 + 570.0, "native")]}


def test_damp60_then_native_distinct_from_damp570():
    a = SD.describe(_damp_then_native(60.0))
    b = SD.describe(_damp_then_native(570.0))
    assert not np.allclose(a, b)          # MUST be distinguishable
    assert SD.desc_hash(a) != SD.desc_hash(b)
    seg_a = a[:reg.N_SEG * 4].reshape(reg.N_SEG, -1)
    seg_b = b[:reg.N_SEG * 4].reshape(reg.N_SEG, -1)
    assert np.allclose(seg_a[0, 1], 1.0)   # segment 0 fully damping
    assert np.allclose(seg_a[2, 0], 1.0)   # segment 2 fully native
    assert np.allclose(seg_b[:, 1], 1.0)   # all 19 segments damping
    # and identical plans are identical descriptors
    assert SD.desc_hash(a) == SD.desc_hash(SD.describe(
        _damp_then_native(60.0)))


def test_seg_shares_are_time_weighted_and_normalised():
    sched = _damp_then_native(45.0)       # 1.5 segments damping
    seg = SD.seg_shares(sched)
    assert seg.shape == (reg.N_SEG, 4)
    assert np.allclose(seg.sum(axis=1), 1.0)
    assert seg[0, 1] == pytest.approx(1.0)
    assert seg[1, 1] == pytest.approx(0.5)   # 15 s of 30 s damping
    assert seg[1, 0] == pytest.approx(0.5)
    assert seg[2, 0] == pytest.approx(1.0)


def test_wall_member_schedule_holds_then_releases():
    sched = SD.wall_member_schedule(("m1", "m2"), 90.0, t0=0.0,
                                    request_s=30.0, others="native")
    assert set(sched) == {"m1", "m2", "__others__"}
    m1 = sched["m1"]
    assert m1[0] == (30.0, 120.0, "wall")
    assert all(a == "native" for _lo, _hi, a in sched["__others__"])
    desc = SD.describe(sched, wall=SD.wall_block(90.0, 5.0, 0.6,
                                                 "early", 0.0, None))
    seg = desc[:reg.N_SEG * 4].reshape(reg.N_SEG, -1)
    assert seg[0, 0] == pytest.approx(1.0)      # request only at 30 s
    assert seg[1, 3] == pytest.approx(2 / 3)    # 2 wall members of 3 rows
    assert seg[5, 0] == pytest.approx(1.0)      # released well before


def test_wall_block_carries_hold_speed_cap():
    w = SD.wall_block(240.0, 5.6, 0.85, "late", 60.0, None)
    assert w[0] == pytest.approx(240.0 / 600.0)
    assert w[1] == pytest.approx(5.6 / 40.0)
    assert w[2] == pytest.approx(0.85)
    assert w[3] == 1.0 and w[4] == pytest.approx(0.1)
    assert w[5] == 0.0 and w[6] == 0.0          # v2.1 data: no hold cap
    w22 = SD.wall_block(240.0, 5.6, 0.85, "late", 60.0, 180.0)
    assert w22[5] == 1.0 and w22[6] == pytest.approx(180.0 / 600.0)
    with pytest.raises(ValueError):
        SD.wall_block(60.0, 5.0, 0.6, "sideways", 0.0, None)


def test_duration_strata_of_pairs():
    native = SD.describe(SD.fleet_schedule("native"))
    damp60 = SD.describe(_damp_then_native(60.0))
    damp570 = SD.describe(_damp_then_native(570.0))
    wall240 = SD.describe(SD.wall_member_schedule(
        ("m1",), 240.0, t0=0.0, request_s=60.0))
    assert SD.duration_stratum(damp60, native) == "le60"
    assert SD.duration_stratum(wall240, native) == "mid"
    assert SD.duration_stratum(damp570, native) == "whole"
    assert SD.duration_stratum(native, native) == "whole"


def test_first60_only_and_onset_bins():
    native = SD.describe(SD.fleet_schedule("native"))
    damp60 = SD.describe(_damp_then_native(60.0))
    damp45_at120 = SD.describe({"cav0": [(120.0, 165.0, "damping"),
                                         (165.0, 570.0, "native")]})
    assert SD.first60_only(damp60, native)
    assert not SD.first60_only(damp45_at120, native)
    assert SD.onset_bin(damp60, native) == "on0_30"
    assert SD.onset_bin(damp45_at120, native) == "on60_240"


def test_count_by_stratum():
    native = SD.describe(SD.fleet_schedule("native"))
    damp60 = SD.describe(_damp_then_native(60.0))
    damp570 = SD.describe(_damp_then_native(570.0))
    counts = SD.count_by_stratum([(damp60, native), (damp60, native),
                                  (damp570, native)])
    assert counts == {"le60": 2, "mid": 0, "whole": 1, "n": 3,
                      "n_first60_only": 2}


def test_unknown_planned_action_is_loud():
    with pytest.raises(ValueError):
        SD.seg_shares({"cav0": [(0.0, 30.0, "turbo")]})

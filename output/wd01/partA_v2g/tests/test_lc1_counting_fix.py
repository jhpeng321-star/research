"""WD-01-PA2G-LC1 class-(ii) counting-fix regression tests.

The counting audit (results/lc1/audit_report.json) showed the eval's
tracked cohort included vehicles that are junction-INTERNAL in their raw
row at a window start (retrospective_junction places them at their
destination segment start).  The prediction side -- whose snapshots
never contain junction states -- cannot mirror that, so truth counted
66 extra LC events over the 32-world panel (47 merge, 8 disc_left,
11 disc_right).  Fix: tracked_state() excludes raw-internal vehicles.

These tests pin the fix at unit level and end-to-end through
eval_a2.window_rows with a synthetic log.
"""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
WD01 = HERE.parents[1]
for p in (str(WD01), str(HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_a2 as EA2  # noqa: E402
from partA_v2g.engine import load_params  # noqa: E402

PARAMS = load_params()
ZONE = PARAMS["rsu_zone_default"]["per_lane_s"]


def _truth_row(entries):
    return {vid: (lane, s, v, a) for vid, (lane, s, v, a) in
            entries.items()}


class TestTrackedStateExcludesMidJunction:
    def test_excluded_when_inj_given(self):
        # veh A on main_right inside the zone; veh B mid-junction whose
        # retro-fixed row lands in the zone (destination placement)
        truth = _truth_row({
            "A": (1, 2000.0, 20.0, 0.0),
            "B": (2, 378.66, 5.0, 0.0),     # retro-fixed junction row
        })
        vehicles, flags = EA.tracked_state(truth, PARAMS,
                                           inj_at=frozenset({"B"}))
        assert set(vehicles) == {"A"}
        assert flags == []

    def test_included_without_inj_keeps_old_signature(self):
        truth = _truth_row({
            "A": (1, 2000.0, 20.0, 0.0),
            "B": (2, 378.66, 5.0, 0.0),
        })
        vehicles, _ = EA.tracked_state(truth, PARAMS)
        assert set(vehicles) == {"A", "B"}

    def test_out_of_zone_junction_row_excluded_either_way(self):
        truth = _truth_row({
            "C": (1, 100.0, 20.0, 0.0),     # outside the zone
        })
        vehicles, _ = EA.tracked_state(truth, PARAMS,
                                       inj_at=frozenset({"C"}))
        assert vehicles == {}


class _FakeLog:
    """Minimal log surface for eval_a2.window_rows: a tracked vehicle on
    the accel lane that merges 1 s after the window start, plus a
    mid-junction vehicle that merges later."""

    def __init__(self, with_inj=True):
        self.t0 = 0.0
        self.t_end = 30.0
        self.cav_ids = set()
        self.types = {}
        n = 240
        self.rows = {j: {} for j in range(n + 1)}
        # phys: on accel (2) at zone s=380 from j0, merges (2->1) at j0+2
        self.rows[0] = {"phys": (2, 380.0, 20.0, 0.0)}
        self.rows[1] = {"phys": (2, 390.0, 20.0, 0.0)}
        self.rows[2] = {"phys": (1, 2652.11, 20.0, 0.0)}
        for j in range(3, n + 1):
            self.rows[j] = {"phys": (1, 2652.11 + 20.0 * 0.5 * (j - 2),
                                     20.0, 0.0)}
        # inj veh: junction-internal in the RAW rows at j0; retro-fixed
        # rows place it at the destination s0 on lane 2, then it merges
        raw_inj_fixed = {0: (2, 378.66, 5.0, 0.0),
                         1: (2, 385.0, 10.0, 0.0),
                         2: (1, 2650.0, 10.0, 0.0)}
        for j, st in raw_inj_fixed.items():
            self.rows[j]["injveh"] = st
        for j in range(3, n + 1):
            self.rows[j]["injveh"] = (1, 2650.0 + 10.0 * 0.5 * (j - 2),
                                      10.0, 0.0)
        self.inj = {0: frozenset({"injveh"})} if with_inj else {}

    def truth_at(self, w):
        return self.rows[int(round((w - self.t0) / 0.5))]


class TestWindowRowsCountingIdentity:
    def _run(self, tmp_path, with_inj):
        log = _FakeLog(with_inj=with_inj)
        entry = dict(world="synthetic", chunk=4, penetration=0.1,
                     demand=2400)
        EA2.MAPPER = None  # window_rows does not need the mapper
        rows, _t = EA2.window_rows(
            PARAMS, log, entry, 0.0, replica=None,
            hook_of=lambda w, vehs: None, emit_hs=(10,), emit_const=False)
        return [r for r in rows if r["model"] == "v2g" and r["H"] == 10]

    def test_mid_junction_vehicle_not_counted(self):
        row = self._run(tmp_path=None, with_inj=True)[0]
        assert row["n_tracked"] == 1
        # only 'phys' is tracked; its own merge at step 2 counts once
        assert row["lc_true"]["merge"] == 1
        assert row["lc_true"]["disc_left"] == 0
        assert row["lc_true"]["disc_right"] == 0

    def test_without_the_fix_the_vehicle_would_be_counted(self):
        row = self._run(tmp_path=None, with_inj=False)[0]
        assert row["n_tracked"] == 2
        assert row["lc_true"]["merge"] == 2       # the old behaviour

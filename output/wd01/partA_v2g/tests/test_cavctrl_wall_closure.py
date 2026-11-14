"""partA_v2g wall-v2.1 closure: WallBook/CavControllerLoop mirror of
wall_group (assembly span test, timeout, closure commands)."""
import sys
from pathlib import Path

import pytest

WD01 = Path(__file__).resolve().parents[2]
for _p in (str(WD01), str(WD01 / "partA_ctrl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")
import refload  # noqa: E402

refload.load_reference()

from partA_v2g import cavctrl  # noqa: E402
from partA_v2g.engine import PartAV2GEngine, load_params  # noqa: E402
from traffic_shepherd import wall_group as wg  # noqa: E402
from types import SimpleNamespace  # noqa: E402

P = load_params()


def _spec(hold=None):
    from predict.contract import WallSpec
    return WallSpec(wall_id="W", members=("f", "r"), speed_mps=20.0,
                    phase="assemble", release_at_s=None, hold_s=hold)


def test_wide_pair_first_tick_matches_reference_closure():
    v = {"f": dict(lane=0, s=2050.0, v=20.0, is_cav=True),
         "r": dict(lane=1, s=2000.0, v=20.0, is_cav=True)}
    loop = cavctrl.CavControllerLoop(t0=480.0, params=P, per_cav_actions={},
                                     walls=[_spec()], cav_state_hint={})
    e = PartAV2GEngine(P, t0=480.0, vehicles=v, control_hook=loop)
    cmd = loop(480.0, e)
    # reference law -> target speed through the same direct_target bridge is
    # exercised by the engine; compare the commanded accelerations instead
    # via the emulator law on the reference session.
    s = wg.WallGroupSession("W", members=frozenset({"f", "r"}), wall_speed_mps=20.0)
    fr = lambda vid, sp, pos, lane: SimpleNamespace(  # noqa: E731
        vehicle_id=vid, timestamp_s=0.0, speed_mps=sp,
        distance_to_merge_m=-pos, lane_index=lane)
    ref = s.tick({"f": fr("f", 20.0, 2050.0, 0), "r": fr("r", 20.0, 2000.0, 1)}, 0.0)
    assert ref["f"] == 0.0 and ref["r"] == 1.5      # closure: r accelerates
    # front commanded target is at/below the rear's (rear is pushed faster)
    assert cmd["r"] > cmd["f"]


def test_assembly_needs_span_and_times_out():
    loop = cavctrl.CavControllerLoop(t0=480.0, params=P, per_cav_actions={},
                                     walls=[_spec()], cav_state_hint={})
    wb = loop.walls
    wb.observe(481.0, {"f": 20.0, "r": 20.0}, {"f": 2050.0, "r": 2000.0})
    assert wb.assembled_at["W"] is None            # speeds ok, span 50 m
    assert wb.active_speed("r", 539.0) == 20.0
    # timeout: exactly at t0 + 60 s the wall aborts (same tick as wall_group)
    assert wb.active_speed("r", 540.0) is None
    assert wb.timed_out["W"] is True
    wb.observe(541.0, {"f": 20.0, "r": 20.0}, {"f": 2000.0, "r": 2000.0})
    assert wb.assembled_at["W"] is None            # timed-out walls never assemble


def test_timeout_tick_matches_wall_group_session():
    s = wg.WallGroupSession("W", members=frozenset({"f", "r"}), wall_speed_mps=20.0)
    wb = cavctrl.WallBook([_spec()], 480.0, P)
    fr = lambda vid, sp, pos, lane, now: SimpleNamespace(  # noqa: E731
        vehicle_id=vid, timestamp_s=now, speed_mps=sp,
        distance_to_merge_m=-pos, lane_index=lane)
    for k in range(0, 66):
        cmds = s.tick({"f": fr("f", 23.0, 2000.0 + 50 + k, 0, float(k)),
                       "r": fr("r", 20.0, 2000.0, 1, float(k))}, float(k))
        assert bool(cmds) == (wb.active_speed("r", 480.0 + k) is not None), k
    assert s.abort_reason == "assembly_timeout"

"""Branch p4-delay cadence regression (zero simulation).

The branch delay must be evaluated on EVERY 0.5 s physics row -- the same
cadence as the master's physics rows and the recorded-run evaluation -- not on
the 1 s control-cadence telemetry rows (which under-count presence time by
about a quarter)."""

import pytest

from tests.planner import test_perfect_sim_fresh_process as _fresh
from tests.planner.test_perfect_sim_fresh_process import HORIZON, NATIVE_PLAN, T0, Rig
from tests.planner.test_ta_closed_loop_fixes import (
    WINDOW,
    _direct_p4,
    _fixture_rows,
    _record_from_rows,
)
from traffic_shepherd.planner.perfect_sim import (
    BranchRecord,
    MetricsExtractor,
    P4DelayEvaluator,
)
from traffic_shepherd.planner.recorded import Layout, _p4_net_path, load_p4_delay


@pytest.fixture(scope="module")
def p4_lib():
    core, geometry = load_p4_delay()
    geom = geometry.wm30_geometry(net_path=_p4_net_path())
    return core, geom, getattr(geometry, "WM30_CHAINS", None)


def test_runner_records_every_physics_step_and_1s_telemetry(tmp_path):
    rig = Rig(tmp_path)
    runner = _fresh.TestMissingControlWriteFailClosed._bare_runner(rig)
    manifest = rig.snapshotter.save(t0_s=T0, controller_state={}, kept_wall_ids=[])
    connection = rig.factory.start_process(label="cadence")
    record = runner.run(
        NATIVE_PLAN,
        connection=connection,
        **_fresh.TestMissingControlWriteFailClosed._run_kw(
            manifest, commit_s=None, continuation=None
        ),
    )
    phys = [round(s.time_s, 6) for s in record.physics_steps]
    assert phys == [round(T0 + 0.5 * k, 6) for k in range(1, int(HORIZON / 0.5) + 1)]
    assert all(s.dt_s == 0.5 for s in record.physics_steps)
    one_s = [round(s.time_s, 6) for s in record.steps]
    assert one_s == [round(T0 + k, 6) for k in range(1, int(HORIZON) + 1)]
    # the 1 s telemetry rows ARE the physics rows at whole seconds (no second copy)
    by_time = {round(s.time_s, 6): s for s in record.physics_steps}
    assert all(by_time[round(s.time_s, 6)] is s for s in record.steps)


def _subsampled_record(rows, t0_present):
    """A record whose ``steps`` are the 1 s telemetry rows and whose
    ``physics_steps`` are all 0.5 s rows (what the fixed runner records)."""
    full = _record_from_rows(rows, t0_present)
    one_s = tuple(s for s in full.steps if abs(s.time_s - round(s.time_s)) < 1e-9)
    return BranchRecord(
        plan_label=full.plan_label, draw_id=full.draw_id, t0_s=full.t0_s,
        horizon_s=full.horizon_s, dt_s=full.dt_s, free_flow_s=full.free_flow_s,
        steps=one_s, physics_steps=full.steps, tracked=full.tracked,
        cav_ids=full.cav_ids, t0_present_ids=full.t0_present_ids,
    )


def test_branch_delay_is_p4_on_all_half_second_rows_not_the_1s_subsample(p4_lib):
    rows, departs, t0_present = _fixture_rows()
    core, geom, chains = p4_lib
    evaluator = P4DelayEvaluator(
        departs=departs, layout=Layout(), core=core, geom=geom, lane_chains=chains
    )
    record = _subsampled_record(rows, t0_present)
    assert len(record.steps) < len(record.physics_steps)
    out = MetricsExtractor(delay_evaluator=evaluator).extract(
        record, part_a_horizon_s=30.0, trajectory_s=0.0
    )
    expected_all, _w, _n = _direct_p4(p4_lib, rows, departs, t0_present, WINDOW)
    one_s_rows = [r for r in rows if abs(r["time_s"] - round(r["time_s"])) < 1e-9]
    subsampled, _w2, _n2 = _direct_p4(p4_lib, one_s_rows, departs, t0_present, WINDOW)
    assert out["delay_s"] == pytest.approx(expected_all, abs=1e-9)
    # REGRESSION GUARD: the 1 s-subsampled value differs materially
    assert abs(subsampled - expected_all) > 1.0
    assert out["delay_s"] != pytest.approx(subsampled, abs=1.0)


def test_legacy_record_without_physics_steps_still_uses_steps(p4_lib):
    rows, departs, t0_present = _fixture_rows()
    core, geom, chains = p4_lib
    evaluator = P4DelayEvaluator(
        departs=departs, layout=Layout(), core=core, geom=geom, lane_chains=chains
    )
    out = MetricsExtractor(delay_evaluator=evaluator).extract(
        _record_from_rows(rows, t0_present), part_a_horizon_s=30.0, trajectory_s=0.0
    )
    expected_all, _w, _n = _direct_p4(p4_lib, rows, departs, t0_present, WINDOW)
    assert out["delay_s"] == pytest.approx(expected_all, abs=1e-9)

from traffic_shepherd.formation import FormationProposal
from traffic_shepherd.lifecycle_wall import LifecycleProfile
from traffic_shepherd.paired_experiments import PolicyInput
from traffic_shepherd.reporting import CAVReport
from traffic_shepherd.recovery_release import RecoveryController, eligible_packets, estimate_downstream_reference, recovery_plan
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame
from traffic_shepherd.wall import WallMember

def prop():
    return FormationProposal(0., ("L0",), (WallMember("m0", "L0"),), 20., LifecycleProfile())

def rep(vid, distance=100., speed=20., stamp=0., front=None):
    f = SensorFrame(vid, stamp, speed, 0., 30., 0, distance, front=front,
                    front_coverage_m=300., rear_coverage_m=300., lane_id="L0", position_m=1000.)
    return CAVReport(f, stamp, "wall")

def ctx(rows, now=0., history=None):
    if history is None:
        history = [(0., r) for r in rows]
    return PolicyInput(now, tuple(rows), tuple(history), {"m0": 0.})

def test_sign_and_nearest_three():
    p = prop()
    rows = [rep("m0"), rep("a", 80., 21.), rep("b", 70., 22.), rep("c", 60., 23.), rep("d", 50., 99.)]
    x = estimate_downstream_reference(p, ctx(rows))
    assert x["details"]["L0"]["source_ids"] == ["a", "b", "c"] and x["per_lane"]["L0"] == 22.

def test_receipt_filter_and_latest_dedup():
    m = rep("m0", stamp=3.)
    good = rep("a", 80., 22., stamp=3.)
    stale = rep("a", 80., 99., stamp=0.)
    c = ctx([m, good], now=3., history=[(3., m), (3., good), (0., stale)])
    assert [x.frame.vehicle_id for x in eligible_packets(c)] == ["a", "m0"]

def test_front_fallback_and_no_double_count():
    x = estimate_downstream_reference(prop(), ctx([rep("m0", front=NeighborMeasurement(25., -2.))]))
    assert x["details"]["L0"]["mode"] == "local_front_fallback" and x["source_count"] == 1

def test_target_clip_and_front_plan_failure():
    p = prop()
    m = rep("m0", front=NeighborMeasurement(6., -10.))
    fast = rep("a", 70., 40.)
    assert estimate_downstream_reference(p, ctx([m, fast]))["target_speed_mps"] == 30.
    assert recovery_plan(p, [m], now_s=0., previous={"m0": 0.}, target=20., reference=20., remaining_s=20.)["feasible"] is False

def test_ramp_no_early_release_and_missing_sticky():
    p = prop()
    m, a = rep("m0", speed=19.), rep("a", 70.)
    c = RecoveryController(p, {"m0": 0.}, 0.)
    assert c.step(ctx([m, a]))["details"]["target_speed_mps"] == 19.5
    d = RecoveryController(p, {"m0": 0.}, 0.)
    first = d.step(ctx([m]))
    assert first["status"] == "fallback" and d.step(ctx([m])) is first

def test_exact_deadline_fallback():
    p = prop()
    m, a = rep("m0", stamp=20.), rep("a", 70., stamp=20.)
    c = RecoveryController(p, {"m0": 0.}, 0.)
    c.last_time = 19.
    assert c.step(ctx([m, a], now=20., history=[(20., m), (20., a)]))["reason"] == "recovery_deadline"

def test_long_runway_active_and_actual_slew_bound():
    p = prop()
    m = rep("m0", 1500., 20.)
    a = rep("a", 1470., 22.)
    c = RecoveryController(p, {"m0": 0.}, 0.)
    out = c.step(ctx([m, a]))
    assert out["status"] == "active"
    assert abs(out["commands"]["m0"] - 0.) <= .15 + 1e-9

def test_three_measured_ticks_release_and_nonzero_prior_blocks_it():
    p = prop()
    c = RecoveryController(p, {"m0": 0.}, 0.)
    for t in range(3):
        m, a = rep("m0", 1500., 20., t), rep("a", 1470., 20., t)
        out = c.step(ctx([m, a], now=t, history=[(t, m), (t, a)]))
    assert out["status"] == "release"
    d = RecoveryController(p, {"m0": .1}, 0.)
    m, a = rep("m0", 1500., 20.), rep("a", 1470., 20.)
    assert d.step(ctx([m, a]))["status"] == "active"

def test_two_lane_geometry_loss_falls_back():
    p = prop()
    p = FormationProposal(0., ("L0", "L1"),
                          (WallMember("m0", "L0"), WallMember("m1", "L1")),
                          20., LifecycleProfile())
    rows = [rep("m0", 1500.), rep("m1", 1400.), rep("a", 1470., 20.)]
    c = RecoveryController(p, {"m0": 0., "m1": 0.}, 0.)
    assert c.step(ctx(rows))["status"] == "fallback"

def test_short_road_and_empty_history_do_not_synthesize_reference():
    p = prop()
    m = rep("m0", 10.)
    a = rep("a", -20., 20.)
    c = RecoveryController(p, {"m0": 0.}, 0.)
    assert c.step(ctx([m, a]))["status"] == "fallback"
    empty = ctx([m, a], history=[])
    assert estimate_downstream_reference(p, empty)["target_speed_mps"] is None

def test_target_ramp_does_not_release_above_reference():
    p = prop()
    m = rep("m0", 1500., 24.)
    a = rep("a", 1470., 27.)
    c = RecoveryController(p, {"m0": 0.}, 0.)
    out = c.step(ctx([m, a]))
    assert out["status"] == "active"
    assert out["details"]["target_speed_mps"] <= 24.5

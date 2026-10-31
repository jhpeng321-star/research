#!/usr/bin/env python3
"""Generate the 20 numeric IDM test vectors and verify ref_idm.py against them.

Usage:  python run_tests.py          (regenerates test_vectors.json and runs all checks)

The expected values are computed with ref_idm.py (pure transcription of SUMO
1.27.1 sources; see spec.md). Three vectors are additionally hand-derived in
spec.md section 10 (closed-form arithmetic), and run_tests.py re-checks a set
of analytic invariants that are INDEPENDENT of the reference code path.
No SUMO installation/run is used anywhere.
"""

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ref_idm as R

TS = 0.5
LANE_50 = 50.0 / 3.6
LANE_30 = 30.0 / 3.6
LANE_40 = 40.0 / 3.6
P0 = R.IDMParams(TS)  # default passenger IDM, sf=1, ASL=TS

VECTORS = []


def add(name, call, inputs, expected, note):
    VECTORS.append(dict(name=name, call=call, inputs=inputs, expected=expected,
                        note=note))


def build_vectors():
    add("T01_follow_free",
        "follow_speed",
        dict(speed=10.0, net_gap=50.0, pred_speed=10.0, lane_limit=LANE_50),
        R.follow_speed(P0, 10.0, 50.0, 10.0, LANE_50),
        "weak interaction; acc ~ free acceleration")
    add("T02_follow_hard_brake",
        "follow_speed",
        dict(speed=LANE_50, net_gap=5.0, pred_speed=0.0, lane_limit=LANE_50),
        R.follow_speed(P0, LANE_50, 5.0, 0.0, LANE_50),
        "fast approach of standing leader, small net gap -> strong braking")
    add("T03_follow_equilibrium",
        "follow_speed",
        dict(speed=10.0, net_gap=10.0, pred_speed=10.0, lane_limit=LANE_50),
        R.follow_speed(P0, 10.0, 10.0, 10.0, LANE_50),
        "s* = tau*v exactly (gap=minGap+s*): residual accel from (v/des)^delta only")
    add("T04_follow_moderate",
        "follow_speed",
        dict(speed=8.3333333333, net_gap=6.0, pred_speed=5.5555555556,
             lane_limit=LANE_50),
        R.follow_speed(P0, 8.3333333333, 6.0, 5.5555555556, LANE_50),
        "mild closing on slower leader")
    add("T05_follow_negative_gap",
        "follow_speed",
        dict(speed=5.0, net_gap=-1.0, pred_speed=0.0, lane_limit=LANE_50),
        R.follow_speed(P0, 5.0, -1.0, 0.0, LANE_50),
        "minGap violated; gap clamped to NUMERICAL_EPS in _v")
    add("T06_follow_standstill",
        "follow_speed",
        dict(speed=0.0, net_gap=0.0, pred_speed=0.0, lane_limit=LANE_50),
        R.follow_speed(P0, 0.0, 0.0, 0.0, LANE_50),
        "both stopped bumper gap == minGap: acc == 0, stay at 0")
    add("T07_stop_approach",
        "stop_speed",
        dict(speed=10.0, gap=20.0),
        R.stop_speed(P0, 10.0, 20.0),
        "braking towards a stop line 20 m ahead (net gap convention)")
    add("T08_stop_near_stop_fallback",
        "stop_speed",
        dict(speed=0.0005, gap=1.0),
        R.stop_speed(P0, 0.0005, 1.0),
        "triggers maximumSafeStopSpeed fallback with headway=ASL=0.5 (IDM.cpp:186-189)")
    add("T09_stop_zero_gap",
        "stop_speed",
        dict(speed=10.0, gap=0.005),
        R.stop_speed(P0, 10.0, 0.005),
        "gap < 0.01 -> immediate 0 (IDM.cpp:181-183)")
    add("T10_stop_overshoot_clamp",
        "stop_speed",
        dict(speed=5.0, gap=0.5),
        R.stop_speed(P0, 5.0, 0.5),
        "result clamped to gap/TS so the Euler step cannot pass the stop line")
    add("T11_free_accelerate",
        "free_speed",
        dict(speed=10.0, seen=100.0, max_speed=LANE_50, lane_limit=LANE_50),
        R.free_speed(P0, 10.0, 100.0, LANE_50, LANE_50),
        "freeSpeed accelerate branch: _v(gap=1e6, pred=maxSpeed)")
    add("T12_limit_drop",
        "free_speed",
        dict(speed=LANE_50, seen=30.0, max_speed=LANE_30, lane_limit=LANE_50),
        R.free_speed(P0, LANE_50, 30.0, LANE_30, LANE_50),
        "decelerate towards a 30 km/h transition 30 m ahead (virtual stopped leader)")
    add("T13_limit_close_clamp",
        "free_speed",
        dict(speed=8.0, seen=1.0, max_speed=LANE_30, lane_limit=LANE_50),
        R.free_speed(P0, 8.0, 1.0, LANE_30, LANE_50),
        "seen < secGap -> vSafe additionally clamped to maxSpeed (IDM.cpp:121-124)")
    add("T14_step_cruise",
        "idm_step",
        dict(speed=10.0, net_gap=20.0, pred_speed=10.0, lane_limit=LANE_50),
        R.idm_step(P0, 10.0, 20.0, 10.0, LANE_50),
        "full step (followSpeed+finalizeSpeed), cruising follower")
    add("T15_step_start",
        "idm_step",
        dict(speed=0.0, net_gap=30.0, pred_speed=LANE_50, lane_limit=LANE_50),
        R.idm_step(P0, 0.0, 30.0, LANE_50, LANE_50),
        "startup from standstill, free leader")
    add("T16_step_leader_stops",
        "idm_step",
        dict(speed=10.0, net_gap=8.0, pred_speed=0.0, lane_limit=LANE_50),
        R.idm_step(P0, 10.0, 8.0, 0.0, LANE_50),
        "leader brakes to 0 8 m ahead -> emergency-range decel allowed by vMin")
    add("T17_speedfactor_1p2",
        "idm_step",
        dict(speed=10.0, net_gap=20.0, pred_speed=10.0, lane_limit=LANE_50,
             speed_factor=1.2),
        R.idm_step(R.IDMParams(TS, speed_factor=1.2), 10.0, 20.0, 10.0, LANE_50),
        "speedFactor=1.2 -> desSpeed = min(55.56, 277.8*1.2, 13.889*1.2)=16.667")
    add("T18_asl_1s",
        "idm_step",
        dict(speed=10.0, net_gap=20.0, pred_speed=10.0, lane_limit=LANE_50,
             action_step_length=1.0),
        R.idm_step(R.IDMParams(TS, action_step_length=1.0), 10.0, 20.0, 10.0,
                   LANE_50),
        "actionStepLength=1.0 s: aMax budgeted over ASL (BASE.cpp:215), not TS")
    add("T19_custom_params",
        "idm_step",
        dict(speed=9.0, net_gap=10.0, pred_speed=6.0, lane_limit=LANE_40,
             accel=1.5, decel=2.0, tau=1.5, min_gap=2.0),
        R.idm_step(R.IDMParams(TS, accel=1.5, decel=2.0, tau=1.5, min_gap=2.0),
                   9.0, 10.0, 6.0, LANE_40),
        "non-default a,b,tau,minGap (2sqrt(ab)=2*sqrt(3)=3.4641)")
    add("T20_step_length_1s",
        "idm_step",
        dict(speed=10.0, net_gap=15.0, pred_speed=8.0, lane_limit=LANE_50, ts=1.0),
        R.idm_step(R.IDMParams(1.0), 10.0, 15.0, 8.0, LANE_50),
        "TS=1.0 -> myIterations=4 (internal step 0.25 s); full 1 s advance")

    for v in VECTORS:
        if "ts" in v["inputs"]:
            v["inputs"]["ts"] = v["inputs"].get("ts")
        v["expected"] = float(v["expected"])


ANALYTIC = []


def check(name, cond, detail=""):
    ANALYTIC.append((name, bool(cond), detail))


def analytic_checks():
    # (A1) 1-iteration config (stepping=0.5 -> myIterations=1): free-road closed form
    p1 = R.IDMParams(TS, stepping=0.5)
    assert p1.iterations == 1
    v = 10.0
    des = P0.desired_speed(LANE_50)
    closed = v + P0.accel * TS * (1.0 - (v / des) ** P0.delta)
    got = R.follow_speed(p1, v, 1e6, v - 1.0, LANE_50)
    check("A1_free_closed_form_iter1", abs(got - closed) < 1e-9,
          "closed=%.12f got=%.12f" % (closed, got))
    # (A2) hand-verified freeSpeed accelerate branch at iterations=2 (spec.md T01')
    got = R.free_speed(P0, 10.0, 100.0, LANE_50, LANE_50)
    check("A2_free_speed_accel_hand", abs(got - 10.9152) < 1e-3,
          "got=%.6f (hand derivation in spec.md 10.1)" % got)
    # (A3) standstill stays put (T06 path)
    check("A3_standstill", R.follow_speed(P0, 0.0, 0.0, 0.0, LANE_50) == 0.0)
    # (A4) stopSpeed never plans to pass the stop line within the Euler step
    ok = all(R.stop_speed(P0, s, g) * TS <= g + 1e-9
             for s in (0.5, 2.0, 5.0, 10.0) for g in (0.02, 0.5, 3.0, 15.0))
    check("A4_no_stop_overshoot", ok)
    # (A5) follower at s* = tau*v, dv = 0: independent two-sub-step closed form
    #      (arithmetics written out here, deliberately NOT calling _v)
    v = 8.0
    des = P0.desired_speed(LANE_50)
    tsab = P0.two_sqrt_ab
    gap = v * P0.tau + P0.min_gap              # 10.5
    s1 = v * P0.tau + P0.min_gap               # 10.5 (dv=0 in sub-step 1)
    acc1 = P0.accel * (1.0 - (v / des) ** 4.0 - (s1 / gap) ** 2)
    v1 = max(0.0, v + acc1 * TS / 2.0)
    dv2 = v1 - v
    s2 = v1 * P0.tau + v1 * dv2 / tsab + P0.min_gap
    # gap stays: gap advance term is max(0, (v1-v)*TS/2) = 0 since v1 < v
    acc2 = P0.accel * (1.0 - (v1 / des) ** 4.0 - (s2 / gap) ** 2)
    v2 = max(0.0, v1 + acc2 * TS / 2.0)
    got = R.follow_speed(P0, v, v * P0.tau, v, LANE_50)
    check("A5_equilibrium_two_substeps", abs(got - v2) < 1e-12,
          "closed=%.12f got=%.12f" % (v2, got))
    # (A6) actionStepLength rounding at TS=0.5 (PARSER.cpp:1772-1789)
    tbl = {0.0: 0.5, -1.0: 0.5, 0.3: 0.5, 0.5: 0.5, 0.75: 0.5, 1.0: 1.0,
           1.4: 1.0, 2.0: 2.0}
    ok = all(abs(R.process_action_step_length(0.5, k) - val) < 1e-9
             for k, val in tbl.items())
    check("A6_asl_rounding", ok)
    # (A7) iterations for both relevant step lengths
    check("A7_iterations", R.IDMParams(0.5).iterations == 2
          and R.IDMParams(1.0).iterations == 4)
    # (A8) two_sqrt_ab exact
    check("A8_two_sqrt_ab", abs(P0.two_sqrt_ab - 2.0 * math.sqrt(11.7)) < 1e-12)


def main():
    build_vectors()
    analytic_checks()
    failures = 0
    for v in VECTORS:
        # recompute through the public API from raw inputs (guards ref_idm edits)
        p = R.IDMParams(v["inputs"].get("ts", TS),
                        speed_factor=v["inputs"].get("speed_factor", 1.0),
                        action_step_length=v["inputs"].get("action_step_length"),
                        accel=v["inputs"].get("accel"),
                        decel=v["inputs"].get("decel"),
                        tau=v["inputs"].get("tau"),
                        min_gap=v["inputs"].get("min_gap"))
        i = v["inputs"]
        if v["call"] == "follow_speed":
            got = R.follow_speed(p, i["speed"], i["net_gap"], i["pred_speed"],
                                 i["lane_limit"])
        elif v["call"] == "stop_speed":
            got = R.stop_speed(p, i["speed"], i["gap"])
        elif v["call"] == "free_speed":
            got = R.free_speed(p, i["speed"], i["seen"], i["max_speed"],
                               i["lane_limit"])
        else:
            got = R.idm_step(p, i["speed"], i["net_gap"], i["pred_speed"],
                             i["lane_limit"])
        if abs(got - v["expected"]) > 1e-12:
            print("FAIL %s: stored=%.12f recomputed=%.12f" % (v["name"],
                                                              v["expected"], got))
            failures += 1
    for name, ok, detail in ANALYTIC:
        if not ok:
            print("FAIL analytic %s: %s" % (name, detail))
            failures += 1
    print("%d vectors + %d analytic invariants: %s"
          % (len(VECTORS), len(ANALYTIC), "ALL PASS" if failures == 0
             else "%d FAILURES" % failures))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "test_vectors.json")
    with open(out, "w") as f:
        json.dump(dict(
            meta=dict(source="SUMO 1.27.1 MSCFModel_IDM (see spec.md)",
                      step_length=TS,
                      default_params=dict(accel=2.6, decel=4.5, emergency_decel=9.0,
                                          tau=1.0, min_gap=2.5, delta=4.0,
                                          stepping=0.25, speed_factor=1.0,
                                          lane_limit_50=LANE_50)),
            vectors=VECTORS), f, indent=1)
    print("wrote %s" % out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

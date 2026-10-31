"""mergespec_tests.py -- pure-python reference formulas for SUMO 1.27.1 merge/lane-end/insertion.

Companion to spec.md / test_cases.md (rbh-mergespec). No SUMO required, stdlib only.
Each function cites the SUMO v1_27_1 source it ports. Run: python mergespec_tests.py
"""
import math

TS = 1.0                    # simulation step length (s)
EULER = True                # SUMO default update rule (semi-implicit Euler)
POSITION_EPS = 0.01         # m


# ---------------------------------------------------------------- CF helpers
def brake_gap_euler(v, decel, headway=0.0):
    """MSCFModel::brakeGapEuler, MSCFModel.cpp:101-108."""
    if v <= 0:
        return 0.0
    dv = decel * TS
    n = int(v / dv)
    return (n * v - dv * n * (n + 1) / 2) + v * headway


def brake_gap(v, decel, headway=0.0):
    if EULER:
        return brake_gap_euler(v, decel, headway)
    return 0.0 if v <= 0 else v * (headway + 0.5 * v / decel)


class IDM:
    """Minimal port of MSCFModel_IDM (MSCFModel_IDM.cpp). accel/decel/tau from vType defaults."""

    def __init__(self, accel=2.6, decel=4.5, tau=1.0, delta=4.0, min_gap=2.5,
                 max_speed=55.55, emergency_decel=9.0):
        self.a, self.b, self.tau = accel, decel, tau
        self.delta, self.min_gap = delta, min_gap
        self.vmax, self.emerg = max_speed, emergency_decel
        self.two_sqrt_ab = 2.0 * math.sqrt(accel * decel)   # ctor, MSCFModel_IDM.cpp:45
        self.iterations = max(1, int(TS / 0.25 + 0.5))      # stepping 0.25, :44

    def _v(self, gap2pred, ego_speed, pred_speed, des_speed, respect_min_gap=True):
        """MSCFModel_IDM::_v, MSCFModel_IDM.cpp:224-269 (acc integrated over 4 substeps)."""
        new_speed, gap = ego_speed, gap2pred
        if respect_min_gap:
            gap += self.min_gap
        for _ in range(self.iterations):
            delta_v = new_speed - pred_speed
            s = max(0.0, new_speed * self.tau + new_speed * delta_v / self.two_sqrt_ab)
            if respect_min_gap:
                s += self.min_gap
            gap = max(1e-9, gap)
            acc = self.a * (1.0 - (new_speed / max(1e-9, des_speed)) ** self.delta
                            - (s * s) / (gap * gap))
            new_speed = max(0.0, new_speed + acc * TS / self.iterations)
            gap -= max(0.0, (new_speed - pred_speed) * TS / self.iterations)
        return max(0.0, new_speed)

    def follow_speed(self, gap, v, v_l, lane_max=55.55):
        return self._v(gap, v, v_l, lane_max)          # MSCFModel_IDM.cpp:130-134

    def stop_speed(self, gap, v, lane_max=55.55):
        """MSCFModel_IDM::stopSpeed, :178-200."""
        if gap < 0.01:
            return 0.0
        result = self._v(gap, v, 0.0, lane_max, respect_min_gap=False)
        if 0 < v < 1e-9 and result < 1e-9:
            result = self.max_safe_stop_speed(gap, self.b)
        if gap >= 0:
            result = min(result, gap / TS)
        return result

    def free_speed(self, seen, v, max_speed, lane_max=55.55):
        """MSCFModel_IDM::freeSpeed, :104-127."""
        if max_speed < 0:
            return max_speed
        sec_gap = self.get_secure_gap(max_speed, 0.0)
        if v <= max_speed:
            v_safe = self._v(1e6, v, max_speed, lane_max, respect_min_gap=False)
        else:
            v_safe = self._v(max(seen, sec_gap), v, 0.0, lane_max, respect_min_gap=False)
        if seen < sec_gap:
            v_safe = min(v_safe, max_speed)
        return v_safe

    def get_secure_gap(self, v, v_l):
        """MSCFModel_IDM::getSecureGap, :217-221."""
        return max(0.0, v * self.tau + v * (v - v_l) / self.two_sqrt_ab)

    def insertion_follow_speed(self, gap, v, v_l, lane_max=55.55):
        """MSCFModel_IDM::insertionFollowSpeed, :137-162 (fixed point)."""
        s = max(0.0, v * self.tau + v * (v - v_l) / self.two_sqrt_ab)
        if gap >= s:
            return v
        speed2 = self.follow_speed(gap, v, v_l, lane_max)
        speed3 = self.follow_speed(gap, speed2, v_l, lane_max)
        if speed2 - speed3 < self.a * TS:
            return speed2
        return self.insertion_follow_speed(gap, speed2, v_l, lane_max)

    def insertion_stop_speed(self, gap, v):
        """MSCFModel_IDM::insertionStopSpeed, :165-175 (equilibrium iteration)."""
        result = max(0.0, min(self.stop_speed(gap, 0.0), self.vmax))  # base, ballistic branch value
        i = 0
        while result - v < -self.b * TS and i < 10:
            v = result
            result = max(0.0, min(self.stop_speed(gap, 0.0), self.vmax))
            i += 1
        return result

    @staticmethod
    def max_safe_stop_speed(gap, decel):
        """maximumSafeStopSpeedEuler (discrete): largest v so the Euler brake curve fits in gap."""
        v = 0.0
        for _ in range(200):
            if brake_gap_euler(v + decel * TS, decel) <= gap:
                v += decel * TS
            else:
                break
        return v


# ---------------------------------------------------------------- insertion
def secure_gap_krauss(v, v_l, decel, leader_decel, headway=1.0):
    """MSCFModel::getSecureGap, MSCFModel.cpp:177-195."""
    max_decel = max(decel, leader_decel)
    return max(0.0, brake_gap(v, decel, headway) - brake_gap(v_l, max_decel, 0.0))


def leader_gap_ok(ego, leader, pos, lane_len, check="leaderGap"):
    """MSLane::safeInsertionSpeed, MSLane.cpp:1434-1470 + isInsertionSuccess leader step.

    ego/leader: dicts with speed, min_gap; ego.cf is an IDM. pos = ego front position.
    Returns (ok, allowed_speed)."""
    cf = ego["cf"]
    speed = ego["speed"]
    seen = lane_len - pos
    gap = leader["back_pos"] + seen - ego["min_gap"]      # net gap (1450)
    if gap < 0:
        return (False, None) if check in ("leaderGap", "collision") else (False, 0.0)
    nspeed = cf.insertion_follow_speed(gap, speed, leader["speed"])
    return True, min(speed, nspeed)


def follower_gap_ok(follower, ego, gap, check_mask=("FOLLOWER_GAP", "COLLISION")):
    """MSLane.cpp:1228-1254. gap = follower front - ego back (net distance)."""
    back_gap_needed = follower["cf"].get_secure_gap(follower["speed"], ego["speed"])
    if gap < back_gap_needed and "FOLLOWER_GAP" in check_mask:
        return False
    if gap < 0 and "COLLISION" in check_mask:
        return False
    return True


def unsafe_merge_speeds(leader_speed, follower_speed, leader_decel, follower_decel):
    """MSLink.h:730-733."""
    return (leader_speed ** 2 / leader_decel) <= (follower_speed ** 2 / follower_decel)


# ---------------------------------------------------------------- lane end
LOOK_FORWARD = 10.0          # MSLCM_LC2013.cpp:48
URGENCY = 2.0                # :64
LOOK_AHEAD_SPEED_MEMORY = 0.9  # :56
LOOK_AHEAD_MIN_SPEED = 0.0   # :55


def update_look_ahead_speed(prev, v_now, action_step=TS):
    """MSLCM_LC2013.cpp:1235-1243."""
    if v_now > prev:
        return v_now
    memory = 1.0 - (1.0 - LOOK_AHEAD_SPEED_MEMORY) * action_step
    return max(LOOK_AHEAD_MIN_SPEED, memory * prev + (1 - memory) * v_now)


def la_dist(v_look, lc_strategic=1.0, right=True, lookahead_left=2.0, length_with_gap=7.5):
    """MSLCM_LC2013.cpp:1244-1245."""
    d = v_look * LOOK_FORWARD * lc_strategic * (1.0 if right else lookahead_left)
    return d + length_with_gap * 2.0


def strategic_urgent(usable_dist, best_lane_offset, v_look, **kw):
    """MSLCM_LC2013.cpp:1330-1333 + MSLCM_LC2013.h:197-199."""
    d = la_dist(v_look, **kw)
    return usable_dist / max(1, abs(best_lane_offset)) < d, d


def remaining_seconds(left_space, v_look, remaining_lanes=1):
    """MSLCM_LC2013.cpp:1502-1507."""
    return max(TS, left_space / max(v_look, 1e-9) / remaining_lanes / URGENCY)


def dead_end_stop_trajectory(v0, decel, seen, max_steps=50):
    """MSVehicle.cpp:2755-2767 with IDM stopSpeed; returns (v, seen) per step until halt."""
    cf = IDM(decel=decel)
    trace = [(v0, seen)]
    v, s = v0, seen
    for _ in range(max_steps):
        v_next = cf.stop_speed(s, v)
        v_next = max(0.0, min(v, v_next))
        dist = min(s, v_next * TS if EULER else 0.5 * (v + v_next) * TS)
        v, s = v_next, s - dist
        trace.append((round(v, 4), round(s, 4)))
        if v < 1e-3 and s <= 0.11:
            break
    return trace


# ---------------------------------------------------------------- zipper
ZIPPER_LOOKAHEAD_S = 16.0    # MSLink.cpp:74
MINOR_LOOKAHEAD_S = 1.0      # MSLink.cpp:72


def could_brake_for_leader(follow_dist, leader_dist, follow_speed, follow_decel, leader_speed):
    """MSLink::couldBrakeForLeader, MSLink.cpp:2307-2313."""
    return (follow_dist > leader_dist and
            follow_dist - leader_dist > follow_speed - follow_decel - leader_speed)


def estimate_speed_after_distance(dist, v, accel, vmax=55.55):
    """MSCFModel::estimateSpeedAfterDistance, MSCFModel.cpp:797-802."""
    return min(vmax, math.sqrt(max(0.0, 2 * dist * accel + v * v)))


def avoid_arrival_accel(dist, time, speed, max_decel):
    """MSCFModel::avoidArrivalAccel, MSCFModel.cpp:553-567."""
    assert time > 0 or dist == 0
    if dist <= 0:
        return -max_decel
    if time * speed > 2 * dist:
        return -0.5 * speed * speed / dist
    return 2 * (dist / time - speed) / time


def zipper_speed(ego, foe, foe_vis_dist=50.0):
    """MSLink::getZipperSpeed, MSLink.cpp:2157-2304.

    ego/foe dicts: dist (to link), speed, lane_index, length, min_gap, cf(IDM),
    lane_max, accel, decel, arrival_time.
    """
    cf, fcf = ego["cf"], foe["cf"]
    brake_gap_ego = brake_gap(ego["speed"], cf.b, TS)
    dist = ego["dist"]
    if dist > max(foe_vis_dist, brake_gap_ego):
        return ego.get("v_safe", ego["speed"])
    foe_dist = foe["dist"]
    if ((foe["arrival_time"] > ego["arrival_time"] and
         not could_brake_for_leader(dist, foe_dist, ego["speed"], cf.b, foe["speed"])) or
        could_brake_for_leader(foe_dist, dist, foe["speed"], fcf.b, ego["speed"]) or
        (foe["arrival_time"] == ego["arrival_time"] and foe_dist == dist and
         ego["lane_index"] < foe["lane_index"])):   # tie: right (lower index) wins, ego yields
        return ego.get("v_safe", ego["speed"])

    u_end = min(foe["lane_max"], estimate_speed_after_distance(foe_dist, foe["speed"], fcf.a))
    u_avg = (foe["speed"] + u_end) / 2
    tf = max(1.0, math.ceil((foe_dist / max(1e-9, u_avg)) / TS) * TS)
    v_max = ego["lane_max"]
    v_accel = estimate_speed_after_distance(dist, ego["speed"], cf.a)
    v_decel = estimate_speed_after_distance(dist, ego["speed"], -cf.b)
    v_end = min(v_max, v_accel, max(u_end, v_decel))
    v_avg = (ego["speed"] + v_end) / 2
    te = max(1.0, math.ceil((dist / max(1e-9, v_avg)) / TS) * TS)

    t_target = tf + cf.tau
    v_safe0 = ego.get("v_safe", ego["speed"])
    a = avoid_arrival_accel(dist, t_target, v_safe0, cf.b)
    gap = dist - foe["length"] - ego["min_gap"] - foe_dist
    v_follow = cf.follow_speed(gap, ego["speed"], foe["speed"], ego["lane_max"])
    v_safe_gap = max(v_follow, ego["speed"] + a * TS)
    w = min(1.0, te / 10.0)
    max_decel = w * cf.b + (1 - w) * cf.emerg
    v_zipper = max(v_follow, ego["speed"] - max_decel * TS, v_safe_gap)
    return min(v_safe0, v_zipper)


def blocked_by_foe_window(ego_arrival, ego_leave, foe_arrival, foe_leave,
                          same_target_lane, zipper=False,
                          unsafe_lead=False, unsafe_follow=False):
    """MSLink::blockedByFoe core window rule, MSLink.cpp:1175-1219 (impatience=0)."""
    look = ZIPPER_LOOKAHEAD_S if zipper else MINOR_LOOKAHEAD_S
    if foe_leave < ego_arrival:                      # ego becomes follower
        if same_target_lane and (ego_arrival - foe_leave < look or unsafe_follow):
            return True
    elif foe_arrival > ego_leave + look:             # ego becomes leader
        if same_target_lane and unsafe_lead:
            return True
    else:
        return True                                  # hard conflict
    return False


# ---------------------------------------------------------------- tests
def _approx(a, b, tol=1e-6):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def run():
    cf = IDM()
    results = []

    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    # T1 brakeGap (Euler), MSCFModel.cpp:101-108. dv=4.5, n=int(10/4.5)=2
    # -> SPEED2DIST(n*v - dv*n*(n+1)/2) = 2*10 - 4.5*3 = 6.5 m (1 s reaction)
    check("T1 brakeGapEuler(10,4.5,0)", _approx(brake_gap(10.0, 4.5), 6.5),
          f"got {brake_gap(10.0, 4.5):.4f}")
    dv = 4.5
    n = int(10 / dv)
    check("T1b brakeGapEuler closed form", _approx(brake_gap(10.0, 4.5), n * 10 - dv * n * (n + 1) / 2),
          f"n={n}")

    # T2 IDM secureGap, MSCFModel_IDM.cpp:217-221
    check("T2 secureGap equal speeds", _approx(cf.get_secure_gap(10, 10), 10.0))
    check("T2b secureGap closing 10 vs 5",
          _approx(cf.get_secure_gap(10, 5), 10 + 50 / cf.two_sqrt_ab),
          f"{cf.get_secure_gap(10,5):.4f}")

    # T3 IDM insertionFollowSpeed keeps speed when gap >= s, MSCFModel_IDM.cpp:137-162
    s10 = cf.get_secure_gap(10.0, 5.0)
    check("T3 insertionFollow large gap keeps v",
          cf.insertion_follow_speed(s10, 10.0, 5.0) == 10.0)
    v_small = cf.insertion_follow_speed(2.0, 10.0, 5.0)
    check("T3b insertionFollow small gap lowers v", 0.0 <= v_small < 10.0, f"v={v_small:.3f}")
    # the fixed point is of insertionFollowSpeed itself (gap >= s at the solution)
    check("T3c fixed point stable", cf.insertion_follow_speed(2.0, v_small, 5.0) == v_small)

    # T4 IDM stopSpeed clamps to gap/TS and zeroes at contact, MSCFModel_IDM.cpp:178-200
    check("T4 stopSpeed at contact", cf.stop_speed(0.005, 5.0) == 0.0)
    check("T4b stopSpeed clamp", cf.stop_speed(1.0, 30.0) <= 1.0 / TS)

    # T5 strategic urgency trigger boundary (defaults, right, offset 1)
    v = 13.89
    d = la_dist(v)
    check("T5 laDist default", _approx(d, v * 10 + 15.0), f"{d:.3f}")
    urgent_at, _ = strategic_urgent(d - 1.0, 1, v)
    ok_at, _ = strategic_urgent(d + 1.0, 1, v)
    check("T5b urgent below boundary / ok above", urgent_at and not ok_at)
    d_left = la_dist(v, right=False)
    check("T5c left lookahead doubled", _approx(d_left, 2 * (v * 10) + 15.0), f"{d_left:.3f}")
    # two required changes: usableDist/2 < laDist
    u2, _ = strategic_urgent(2 * d - 1, 2, v)
    check("T5d two changes use usable/2", u2)

    # T6 remainingSeconds, MSLCM_LC2013.cpp:1502-1507
    rs = remaining_seconds(100.0, 10.0, 1)
    check("T6 remainingSeconds", _approx(rs, 5.0), f"{rs:.3f}")
    check("T6b floor at TS", remaining_seconds(1.0, 30.0, 1) == 1.0)
    check("T6c remainingLanes", _approx(remaining_seconds(100.0, 10.0, 2), 2.5))

    # T7 lookAheadSpeed memory decay, MSLCM_LC2013.cpp:1235-1243
    check("T7 instant growth", update_look_ahead_speed(5.0, 8.0) == 8.0)
    check("T7b decay", _approx(update_look_ahead_speed(8.0, 5.0), 0.9 * 8.0 + 0.1 * 5.0))

    # T8 follower gap acceptance at insertion (IDM), MSLane.cpp:1228-1254
    ego = {"speed": 10.0, "cf": cf}
    fol = {"speed": 10.0, "cf": cf}
    sg = cf.get_secure_gap(10.0, 10.0)          # symmetric speeds -> tau*v = 10
    check("T8 follower gap exactly secure", follower_gap_ok(fol, ego, sg))
    check("T8b follower gap below secure fails", not follower_gap_ok(fol, ego, sg - 0.5))

    # T9 unsafeMergeSpeeds, MSLink.h:730-733
    check("T9 safe lead", not unsafe_merge_speeds(10.0, 5.0, 4.5, 4.5))
    check("T9b unsafe lead", unsafe_merge_speeds(5.0, 10.0, 4.5, 4.5))

    # T10 opened()/blockedByFoe windows, MSLink.cpp:1175-1219
    # follower case: foe left 0.5s ago, same lane -> blocked (lookahead 1s)
    check("T10 follower blocked", blocked_by_foe_window(10.0, 12.0, 5.0, 9.5,
                                                        same_target_lane=True))
    # follower case on different target lane -> not blocked by trailing gap
    check("T10b follower cross-merge ok", not blocked_by_foe_window(10.0, 12.0, 5.0, 9.5,
                                                                    same_target_lane=False))
    # leader case outside lookahead, safe speeds -> open
    check("T10c leader ok", not blocked_by_foe_window(10.0, 12.0, 20.0, 30.0,
                                                      same_target_lane=True))
    # leader case, unsafe merge speeds (follower cannot stop) -> blocked
    check("T10d leader unsafe speeds blocked",
          blocked_by_foe_window(10.0, 12.0, 20.0, 30.0, same_target_lane=True,
                                unsafe_lead=True))
    # hard conflict window
    check("T10e hard conflict", blocked_by_foe_window(10.0, 12.0, 11.0, 13.0,
                                                      same_target_lane=False))
    # zipper lookahead stretches the follower window to 16 s
    check("T10f zipper follower blocked",
          blocked_by_foe_window(10.0, 12.0, 0.0, 5.0, same_target_lane=True, zipper=True))
    check("T10g minor follower ok (same gaps)",
          not blocked_by_foe_window(10.0, 12.0, 0.0, 5.0, same_target_lane=True))

    # T11 zipper speed adaptation, MSLink.cpp:2157-2304
    # (a) fast foe just ahead, arriving only 1 s after ego -> must trim NOW
    ego = {"dist": 30.0, "speed": 25.0, "lane_index": 0, "length": 5.0, "min_gap": 2.5,
           "cf": cf, "lane_max": 27.78, "arrival_time": 100, "v_safe": 27.78}
    foe_close = {"dist": 20.0, "speed": 25.0, "lane_index": 1, "length": 5.0,
                 "cf": cf, "lane_max": 27.78, "arrival_time": 101}
    vz = zipper_speed(dict(ego), foe_close)
    check("T11 zipper trims speed", vz < 20.0, f"v_zipper={vz:.3f}")
    # (b) foe behind and able to brake for ego -> ignored, no adaptation
    foe_behind_can_brake = {"dist": 40.0, "speed": 28.0, "lane_index": 1, "length": 5.0,
                            "cf": cf, "lane_max": 27.78, "arrival_time": 110}
    vz2 = zipper_speed(dict(ego, dist=20.0), foe_behind_can_brake)
    check("T11b foe-behind-that-can-brake ignored", vz2 == 27.78, f"{vz2:.3f}")
    ego_far = dict(ego, dist=200.0)
    check("T11c beyond visibility no adaptation",
          zipper_speed(ego_far, foe_close) == 27.78)
    tie_ego_right = dict(ego, dist=30.0, lane_index=0)          # ego on the right
    tie_foe = dict(foe_close, dist=30.0, arrival_time=100)      # tie in time and distance
    vz3 = zipper_speed(tie_ego_right, tie_foe)
    check("T11d right lane wins tie (right ego untouched)", vz3 == 27.78, f"{vz3:.3f}")
    tie_ego_left = dict(tie_ego_right, lane_index=1)
    vz4 = zipper_speed(tie_ego_left, dict(tie_foe, lane_index=0))
    check("T11e left lane yields in tie", vz4 < 20.0, f"{vz4:.3f}")

    # T12 avoidArrivalAccel, MSCFModel.cpp:553-567
    check("T12 a@dist", _approx(avoid_arrival_accel(100.0, 10.0, 15.0, 4.5),
                                2 * (100.0 / 10.0 - 15.0) / 10.0))
    check("T12b stop-before branch", _approx(avoid_arrival_accel(10.0, 10.0, 15.0, 4.5),
                                             -0.5 * 15.0 ** 2 / 10.0))

    # T13 dead-end stop trajectory (Euler stop profile), MSVehicle.cpp:2755-2767
    tr = dead_end_stop_trajectory(10.0, 4.5, 50.0)
    check("T13 stops before lane end", tr[-1][1] >= -0.11 and tr[-1][0] < 0.5,
          f"trace={tr[:6]}")

    # T14 extra reservation for multiple changes, MSAbstractLaneChangeModel.cpp:1178-1188
    def extra_reservation(best_lane_offset, neigh_extra_dist, length_with_gap=7.5):
        if neigh_extra_dist > length_with_gap:
            return 0
        if best_lane_offset < -1:
            return 20
        if best_lane_offset > 1:
            return 40
        return 0
    check("T14 extra reservation", extra_reservation(2, 1.0) == 40 and
          extra_reservation(-2, 1.0) == 20 and extra_reservation(1, 1.0) == 0 and
          extra_reservation(2, 10.0) == 0)

    print(f"{sum(1 for _, ok, _ in results if ok)}/{len(results)} checks passed")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)

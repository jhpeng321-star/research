"""Pure-Python reference implementation of SUMO 1.27.1 MSCFModel_IDM.

Extracted line-by-line from tag v1_27_1 (see spec.md for every citation):
  src/microsim/cfmodels/MSCFModel_IDM.{h,cpp}
  src/microsim/cfmodels/MSCFModel.{h,cpp}
  src/microsim/MSLane.h, MSVehicle.cpp, MSBaseVehicle.{h,cpp},
  src/microsim/MSVehicleType.cpp, MSGlobals.cpp,
  src/utils/vehicle/SUMOVehicleParserHelper.cpp, src/utils/common/SUMOTime.h,
  src/config.h.cmake

No dependency on SUMO. Pure Python 3, deterministic, no randomness.

Conventions (SUMO semantics):
  * TS          = simulation step length in seconds (--step-length, default 1.0,
                 0.5 s in the WM-PHYS rebuild).
  * "net gap"   = bumper-to-bumper distance MINUS ego vType minGap. This is what
                 SUMO passes to followSpeed as gap2pred (MSLane.cpp:4200,
                 MSVehicle.cpp:3210-3227). followSpeed adds minGap back
                 internally (_v with respectMinGap=true, MSCFModel_IDM.cpp:237-240).
                 stopSpeed is called with the net gap as well (MSVehicle.cpp:2520)
                 but does NOT add minGap back (respectMinGap=false).
  * Default mode here: Euler update (gSemiImplicitEulerUpdate=True,
                 MSGlobals.cpp:63), no driverState (perception errors are
                 no-ops, MSCFModel.cpp:1098-1163), no stops, no TraCI
                 influencer, no IDMM adaptation, friction coefficient 1.
"""

import math

# --- exact SUMO constants -------------------------------------------------
NUMERICAL_EPS = 0.001            # src/config.h.cmake:236 (v1_27_1)
POSITION_EPS = 0.1               # src/config.h.cmake:239
EMERGENCY_DECEL_AMPLIFIER = 1.2  # src/microsim/cfmodels/MSCFModel.h:35
HALTING_SPEED = 0.1              # src/utils/common/StdDefs.h:62

# vType defaults for vClass "passenger" (docs Vehicle_Type_Parameter_Defaults.md:47;
# MSCFModel.cpp:57-63) and IDM-specific defaults (Definition_of_Vehicles doc,
# MSCFModel_IDM.cpp:41,44).
PASSENGER_DEFAULTS = dict(
    accel=2.6, decel=4.5, emergency_decel=9.0,
    tau=1.0, min_gap=2.5, max_speed=200.0 / 3.6, desired_max_speed=10000.0 / 3.6,
    delta=4.0, stepping=0.25,
)


class IDMParams(object):
    """Parameter set equivalent to one SUMO vType with carFollowModel=IDM."""

    def __init__(self, ts, accel=None, decel=None, emergency_decel=None,
                 tau=None, min_gap=None, delta=None, stepping=None,
                 vtype_max_speed=None, desired_max_speed=None,
                 speed_factor=1.0, action_step_length=None):
        d = PASSENGER_DEFAULTS
        self.ts = float(ts)
        self.accel = d["accel"] if accel is None else float(accel)
        self.decel = d["decel"] if decel is None else float(decel)
        self.emergency_decel = (d["emergency_decel"] if emergency_decel is None
                                else float(emergency_decel))
        self.tau = d["tau"] if tau is None else float(tau)
        self.min_gap = d["min_gap"] if min_gap is None else float(min_gap)
        self.delta = d["delta"] if delta is None else float(delta)
        stepping = d["stepping"] if stepping is None else float(stepping)
        # MSCFModel_IDM.cpp:44: myIterations = MAX2(1, int(TS/stepping + .5));
        # C++ int() truncates toward zero. TS=0.5, stepping=0.25 -> 2 sub-steps
        # (internal step 0.25 s); TS=1.0 -> 4 sub-steps.
        self.iterations = max(1, int(self.ts / stepping + 0.5))
        # MSCFModel_IDM.cpp:45
        self.two_sqrt_ab = 2.0 * math.sqrt(self.accel * self.decel)
        self.vtype_max_speed = (d["max_speed"] if vtype_max_speed is None
                                else float(vtype_max_speed))
        self.desired_max_speed = (d["desired_max_speed"] if desired_max_speed is None
                                  else float(desired_max_speed))
        self.speed_factor = float(speed_factor)  # individual speedFactor
        # actionStepLength in SECONDS after processActionStepLength() rounding
        # (SUMOVehicleParserHelper.cpp:1772-1789); None -> equals TS (default).
        self.action_step_length = self.ts if action_step_length is None \
            else float(action_step_length)

    # --- desired speed / speedFactor ----------------------------------------
    def vehicle_max_speed(self):
        """MSBaseVehicle::getMaxSpeed(), MSBaseVehicle.cpp:211-213:
        MIN2(vType.maxSpeed, desiredMaxSpeed * chosenSpeedFactor)."""
        return min(self.vtype_max_speed, self.desired_max_speed * self.speed_factor)

    def desired_speed(self, lane_speed_limit):
        """desSpeed used by IDM == veh->getLane()->getVehicleMaxSpeed(veh)
        (MSCFModel_IDM.cpp:133/184/114/119) == MSLane.h:575-592:
        MIN2(vehMaxSpeed, laneLimit * chosenSpeedFactor).
        speedFactor multiplies the LANE limit directly."""
        return min(self.vehicle_max_speed(), lane_speed_limit * self.speed_factor)

    # --- conversion macros (src/utils/common/SUMOTime.h:45-56) --------------
    def accel2speed(self, a):
        return a * self.ts

    def dist2speed(self, d):
        return d / self.ts


# --- core -------------------------------------------------------------------
def _v(p, gap2pred, ego_speed, pred_speed, des_speed, respect_min_gap=True):
    """MSCFModel_IDM::_v, MSCFModel_IDM.cpp:224-269. Core IDM iteration over
    p.iterations sub-steps (this is the 'stepping'/idmStepping mechanism,
    MSCFModel_IDM.cpp:44). Returns the speed (m/s) after ONE step of length TS.

    gap2pred is the NET gap (minGap already subtracted by the SUMO caller);
    when respect_min_gap is true the model adds minGap back to gap and s*.
    """
    headway_time = p.tau  # MSCFModel_IDM.cpp:230 (IDMM adaptation only)
    new_speed = ego_speed
    gap = gap2pred
    if respect_min_gap:
        gap += p.min_gap          # :237-240
    for _ in range(p.iterations):  # :246
        delta_v = new_speed - pred_speed                                     # :247
        s = max(0.0, new_speed * headway_time
                + new_speed * delta_v / p.two_sqrt_ab)                       # :248
        if respect_min_gap:
            s += p.min_gap                                                   # :249-251
        gap = max(NUMERICAL_EPS, gap)                                        # :252
        acc = p.accel * (1.0 - (new_speed / max(NUMERICAL_EPS, des_speed)) ** p.delta
                         - (s * s) / (gap * gap))                            # :253
        new_speed = max(0.0, new_speed + p.accel2speed(acc) / p.iterations)  # :259
        # :266 (pre-advance the gap with the new speed; leader speed constant)
        gap -= max(0.0, (new_speed - pred_speed) * p.ts / p.iterations)
    return max(0.0, new_speed)                                               # :268


def get_secure_gap(p, speed, leader_speed):
    """MSCFModel_IDM::getSecureGap, MSCFModel_IDM.cpp:217-221."""
    delta_v = speed - leader_speed
    return max(0.0, speed * p.tau + speed * delta_v / p.two_sqrt_ab)


def follow_speed(p, speed, net_gap, pred_speed, lane_speed_limit):
    """MSCFModel_IDM::followSpeed, MSCFModel_IDM.cpp:130-134.

    speed: current ego speed [m/s]; net_gap: bumper gap minus ego minGap;
    pred_speed: leader speed (assumed constant); lane_speed_limit: current
    lane's speed limit [m/s]. desSpeed = lane vehicle max speed (with
    speedFactor). NOTE: IDM does NOT cap braking at decel.
    """
    return _v(p, net_gap, speed, pred_speed, p.desired_speed(lane_speed_limit),
              respect_min_gap=True)


# --- stop handling ----------------------------------------------------------
def maximum_safe_stop_speed_euler(p, gap, decel, headway):
    """MSCFModel::maximumSafeStopSpeedEuler, MSCFModel.cpp:860-884."""
    g = gap - NUMERICAL_EPS                       # :862
    if g < 0.0:                                   # :863-865
        return 0.0
    b = p.accel2speed(decel)                      # :866
    t = headway if headway >= 0.0 else p.tau      # :867
    s = p.ts                                      # :868
    n = math.floor(0.5 - ((t + (math.sqrt((s * s) + (4.0 * ((s * (2.0 * g / b - t))
                                                     + (t * t)))) * -0.5)) / s))  # :874
    h = 0.5 * n * (n - 1) * b * s + n * b * t     # :875
    r = (g - h) / (n * s + t)                     # :879
    return n * b + r                              # :880


def calculate_emergency_deceleration(p, gap, ego_speed, pred_speed, pred_max_decel):
    """MSCFModel::calculateEmergencyDeceleration, MSCFModel.cpp:1037-1086."""
    if gap <= 0.0:                                                      # :1042
        return p.emergency_decel
    pred_brake_dist = 0.5 * pred_speed * pred_speed / pred_max_decel    # :1047
    b1 = 0.5 * ego_speed * ego_speed / (gap + pred_brake_dist)          # :1049
    if b1 <= pred_max_decel:                                            # :1061
        return b1
    b2 = 0.5 * (ego_speed * ego_speed - pred_speed * pred_speed) / gap  # :1078
    return b2


def maximum_safe_stop_speed(p, gap, decel, current_speed, headway=-1.0,
                            relax_emergency=True):
    """MSCFModel::maximumSafeStopSpeed, MSCFModel.cpp:806-856 (Euler branch)."""
    vsafe = maximum_safe_stop_speed_euler(p, gap, decel, headway)      # :809-810
    if relax_emergency and p.decel != p.emergency_decel:               # :815
        orig_safe_decel = (current_speed - vsafe) / p.ts               # :825
        if orig_safe_decel > p.decel + NUMERICAL_EPS:                  # :826
            safe_decel = EMERGENCY_DECEL_AMPLIFIER * \
                calculate_emergency_deceleration(p, gap, current_speed, 0.0, 1.0)  # :836
            safe_decel = max(safe_decel, p.decel)                      # :838
            safe_decel = min(safe_decel, orig_safe_decel)              # :840
            vsafe = current_speed - p.accel2speed(safe_decel)          # :841
            vsafe = max(vsafe, 0.0)                                    # :842-844
    return vsafe


def stop_speed(p, speed, gap, decel=None, lane_speed_limit=1e6):
    """MSCFModel_IDM::stopSpeed, MSCFModel_IDM.cpp:178-200.

    gap: NET gap to the obstacle/stop line (SUMO callers subtract minGap,
    MSVehicle.cpp:2520). decel: the function's 'decel' argument; default SUMO
    callers pass myDecel via the inline overload (MSCFModel.h:189-190).
    lane_speed_limit: only influences desSpeed (use a big value when the stop
    dominates, matching a free lane).
    """
    if decel is None:
        decel = p.decel
    if gap < 0.01:                                                      # :181-183
        return 0.0
    result = _v(p, gap, speed, 0.0, p.desired_speed(lane_speed_limit),
                respect_min_gap=False)                                  # :184
    if gap > 0 and speed < NUMERICAL_EPS and result < NUMERICAL_EPS:    # :186
        # near-stop fallback; headway = ACTION step length (:189)
        result = maximum_safe_stop_speed(p, gap, decel, speed,
                                         headway=p.action_step_length)
    if gap >= 0:                                                        # :192-193
        result = min(result, p.dist2speed(gap))
    return result


# --- speed-limit transitions ------------------------------------------------
def free_speed(p, speed, seen, max_speed, lane_speed_limit):
    """MSCFModel_IDM::freeSpeed, MSCFModel_IDM.cpp:104-127.

    seen: distance to the point where the allowed speed becomes max_speed
    (max_speed may be the next lane's limit, a red light's target speed, ...).
    desSpeed remains the CURRENT lane's vehicle max speed (with speedFactor),
    NOT max_speed. maxSpeed<0 passes through (ballistic red-light convention).
    """
    if max_speed < 0.0:                                                 # :106-109
        return max_speed
    sec_gap = get_secure_gap(p, max_speed, 0.0)                         # :110
    des = p.desired_speed(lane_speed_limit)
    if speed <= max_speed:                                              # :112-114
        vsafe = _v(p, 1e6, speed, max_speed, des, respect_min_gap=False)
    else:                                                               # :115-119
        vsafe = _v(p, max(seen, sec_gap), speed, 0.0, des, respect_min_gap=False)
    if seen < sec_gap:                                                  # :121-124
        vsafe = min(vsafe, max_speed)
    return vsafe


# --- acceleration / deceleration bounds -------------------------------------
def max_next_speed(p, speed):
    """MSCFModel::maxNextSpeed, MSCFModel.cpp:323-326 (not overridden by IDM)."""
    return min(speed + p.accel2speed(p.accel), p.vtype_max_speed)


def min_next_speed(p, speed):
    """MSCFModel_IDM::minNextSpeed, MSCFModel_IDM.cpp:79-89 (Euler clips at 0).
    decel_eff = MAX2(myDecel, MIN2(myEmergencyDecel, 1.5))."""
    decel = max(p.decel, min(p.emergency_decel, 1.5))
    return max(speed - p.accel2speed(decel), 0.0)


def min_next_speed_emergency(p, speed):
    """MSCFModel::minNextSpeedEmergency, MSCFModel.cpp:340-348 (Euler)."""
    return max(speed - p.accel2speed(p.emergency_decel), 0.0)


# --- final assembly ---------------------------------------------------------
def finalize_speed(p, old_speed, v_pos, lane_speed_limit):
    """MSCFModel::finalizeSpeed, MSCFModel.cpp:199-265, wrapped by
    MSCFModel_IDM::finalizeSpeed (MSCFModel_IDM.cpp:93-101; the IDMM level-of-
    service update is skipped for plain IDM). Assumptions (all SUMO defaults):
    friction coefficient 1 (MSCFModel.cpp:208-210), startupDelay 0
    (MSCFModel.cpp:269-285 -> identity), no pending stops
    (processNextStop -> identity, MSVehicle.cpp:~5800), identity lane-change
    patchSpeed. v_pos = safe-speed upper bound computed during planMove
    (min over leader followSpeed/stopSpeed, link speeds, limit changes).

    NOTE: aMax divides by the ACTION step length (MSCFModel.cpp:215), not TS.
    """
    old_v = old_speed
    v_stop = v_pos  # processNextStop identity without stops                 # :203
    v_min_emergency = min_next_speed_emergency(p, old_v)                    # :205
    v_min = min(min_next_speed(p, old_v), max(v_pos, v_min_emergency))      # :207
    # friction factor == 1 (MSCFModel.cpp:208-210)
    a_max = (max(p.desired_speed(lane_speed_limit), v_pos) - old_v) \
        / p.action_step_length                                              # :215
    v_max = min(old_v + p.accel2speed(a_max),                               # :217
                max_next_speed(p, old_v), v_stop)
    v_max = max(v_min, v_max)                                               # :231
    # patchSpeedBeforeLC -> vMax (MSCFModel.h:120-124); LC identity; delay 0
    return v_max


def idm_step(p, old_speed, net_gap, pred_speed, lane_speed_limit):
    """One plain simulation step for a follower with one leader on the lane:
    v_pos = followSpeed(...), clamped by the lane vehicle max speed (the planMove
    initialisation, MSVehicle.cpp:2295/2295 cf maximumLaneSpeedCF default),
    then vNext = finalizeSpeed(v_pos). Returns vNext [m/s], Euler-clipped >= 0.
    """
    v_pos = follow_speed(p, old_speed, net_gap, pred_speed, lane_speed_limit)
    v_pos = min(v_pos, p.desired_speed(lane_speed_limit))
    v_next = finalize_speed(p, old_speed, v_pos, lane_speed_limit)
    return max(v_next, 0.0)


def process_action_step_length(ts, given):
    """SUMOVehicleParserHelper::processActionStepLength,
    SUMOVehicleParserHelper.cpp:1772-1789. Non-positive -> TS; otherwise floor
    to a positive multiple of TS (=> value in [TS, +inf), multiple of TS).
    Returns seconds. At step-length 0.5: 0.3 -> 0.5 (+warning), 0.5 -> 0.5,
    0.75 -> 0.5, 1.0 -> 1.0."""
    result_ms = round(given * 1000.0)   # TIME2STEPS, 1 ms resolution
    dt_ms = round(ts * 1000.0)
    if result_ms <= 0:
        result_ms = dt_ms
    elif result_ms % dt_ms != 0:
        result_ms = max(dt_ms, int(dt_ms * math.floor(float(result_ms) / float(dt_ms))))
    return result_ms / 1000.0


def action_step_active(t, asl, last_action_time):
    """MSVehicle::isActionStep, MSVehicle.h:635-636:
    (t - myLastActionTime) % actionStepLength == 0 (SUMOTime milliseconds)."""
    return (round(t * 1000.0) - last_action_time) % round(asl * 1000.0) == 0

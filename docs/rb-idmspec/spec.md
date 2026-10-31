# SUMO 1.27.1 MSCFModel_IDM — Exact Behaviour Spec (WM-PHYS rebuild reference)

Author: rbh-idmspec helper, 2026-10-31. Sources: SUMO tag **v1_27_1**
(https://github.com/eclipse-sumo/sumo/tree/v1_27_1), downloaded read-only into
`_sumo_src/` (this folder; filenames there use `__` in place of `/`).
Companion artifacts: `ref_idm.py` (pure-Python reference implementation),
`test_vectors.json` + `run_tests.py` (20 numeric leader/follower vectors).
No SUMO was run; everything below is source-derived.

Primary source files (line numbers refer to v1_27_1):

| abbrev | file |
|---|---|
| IDM.cpp / IDM.h | `src/microsim/cfmodels/MSCFModel_IDM.{cpp,h}` |
| BASE.cpp / BASE.h | `src/microsim/cfmodels/MSCFModel.{cpp,h}` |
| VEH.cpp / VEH.h | `src/microsim/MSVehicle.{cpp,h}` |
| BASEVEH.{cpp,h} | `src/microsim/MSBaseVehicle.{cpp,h}` |
| LANE.h / LANE.cpp | `src/microsim/MSLane.{h,cpp}` |
| VT.cpp | `src/microsim/MSVehicleType.cpp` |
| GL.cpp | `src/microsim/MSGlobals.cpp` |
| PARSER.cpp | `src/utils/vehicle/SUMOVehicleParserHelper.cpp` |
| TIME.h | `src/utils/common/SUMOTime.h` |
| CONFIG | `src/config.h.cmake` |

Conversion macros (TIME.h:42-56): `TS = DELTA_T/1000` (step length in s),
`ACCEL2SPEED(a)=a*TS`, `SPEED2DIST(v)=v*TS`, `DIST2SPEED(d)=d/TS`,
`ACCEL2DIST(a)=a*TS^2`, `SPEED2ACCEL(v)=v/TS`.
Constants (CONFIG:235-243): **NUMERICAL_EPS = 0.001**, POSITION_EPS = 0.1,
SPEED_EPS = 0.01. `EMERGENCY_DECEL_AMPLIFIER = 1.2` (BASE.h:35).

---

## 1. Parameters and defaults

From IDM.cpp:38-51 (constructor) and BASE.cpp:55-67:

| param | source attr | default | cite |
|---|---|---|---|
| accel `a` | accel | 2.6 m/s² (passenger) | BASE.cpp:57; docs Vehicle_Type_Parameter_Defaults.md:47 |
| decel `b` | decel | 4.5 m/s² | BASE.cpp:58; docs index.md:59 |
| emergencyDecel `bmax` | emergencyDecel | 9 m/s² | BASE.cpp:59-60; docs index.md:63 |
| apparentDecel | apparentDecel | = decel | BASE.cpp:61 |
| tau `T` | tau | 1.0 s | BASE.cpp:63 |
| minGap `s0` | minGap | 2.5 m | docs Vehicle_Type_Parameter_Defaults.md:47 |
| delta | cfIdmDelta (`delta`) | **4.0** | IDM.cpp:41; docs Definition_of_Vehicles.md:827 |
| stepping | cfIdmStepping (`stepping`) | **0.25 s** | IDM.cpp:44; docs Definition_of_Vehicles.md:828 |
| maxSpeed | maxSpeed | 200/3.6 m/s | docs Vehicle_Type_Parameter_Defaults.md:47 |
| desiredMaxSpeed | desiredMaxSpeed | 10000/3.6 m/s (no constraint) | utils/vehicle/SUMOVTypeParameter.cpp:71 |
| speedFactor (per vehicle) | speedFactor | 1.0 | BASEVEH.cpp:109 |
| actionStepLength | actionStepLength | = step length TS | PARSER.cpp:1776-1780 |

Derived (IDM.cpp:44-45):
- `myIterations = MAX2(1, int(TS/stepping + 0.5))` — C++ truncation.
  - TS=0.5 s → `int(2.0+0.5)=2` sub-steps (internal step **0.25 s**)
  - TS=1.0 s → `int(4.0+0.5)=4` sub-steps
  - Warning if internal step > 0.25 s (IDM.cpp:48-50).
- `myTwoSqrtAccelDecel = 2*sqrt(a*b)` (the classic `2*sqrt(a*b)`).
- `myCollisionMinGapFactor = 0.1` default for IDM (IDM.cpp:46-47; used only in
  collision detection, MSLane.cpp:1999-2004).

This `myIterations` loop IS the mechanism that was called "idmStepping"/
`_updateSpeed` in older SUMO versions; in 1.27.1 the sub-stepping lives inside
`_v` (IDM.cpp:246-267). There is no `_updateSpeed` symbol anymore.

## 2. Core iteration `_v` (IDM.cpp:224-269)

Signature: `_v(veh, gap2pred, egoSpeed, predSpeed, desSpeed, respectMinGap=true)`
(default arg IDM.h:208-209). `gap2pred` arrives as the **NET gap** (bumper-to-bumper
minus ego minGap; LANE.cpp:4200, VEH.cpp:3210-3227, VEH.cpp:569). With
`respectMinGap=true` the model adds minGap back (IDM.cpp:237-240).

Exact loop (plain IDM, i.e. `myAdaptationFactor==1`, IDM.cpp:230-234 is a no-op):

```
v  = egoSpeed
gap = gap2pred + (s0 if respectMinGap else 0)          # :236-240
repeat myIterations times:                             # :246
    dv  = v - predSpeed                                # :247
    s   = max(0, v*T + v*dv/(2*sqrt(a*b)))             # :248
    if respectMinGap: s += s0                          # :249-251
    gap = max(NUMERICAL_EPS, gap)                      # :252
    acc = a * ( 1 - (v / max(NUMERICAL_EPS, desSpeed))**delta
                  - (s*s)/(gap*gap) )                  # :253
    v   = max(0, v + acc*TS/myIterations)              # :259
    gap = gap - max(0, (v - predSpeed)*TS/myIterations)# :266
return max(0, v)                                       # :268
```

Key semantics:
- Sub-step length is `TS/myIterations` and the **speed advance per sub-step is
  `acc*TS/iterations`** (IDM.cpp:259), i.e. effectively simulating at the
  `stepping` resolution. `acc` is recomputed each sub-step with the updated `v`.
- The **gap is pre-advanced inside the loop** by ego's projected motion
  `(v-predSpeed)*TS/iterations` (IDM.cpp:266) — an approximate intra-step
  kinematics with leader speed held constant. This makes IDM at gap≈s* brake
  slightly earlier than the textbook continuous IDM.
- `s*` clamped below at 0 (IDM.cpp:248), gap clamped at NUMERICAL_EPS=0.001
  (IDM.cpp:252) — no division-by-zero, so negative/zero gaps yield huge
  deceleration, bounded only by per-step `a*TS` per sub-step; IDM can exceed
  `decel` (docs index.md:59 confirms the bound is "less strict" for IDM).
- v is clipped at 0 each sub-step (IDM.cpp:259) — Euler-style, never negative.
  If that happens mid-step, the next sub-step sees s*=0 and re-accelerates by
  a*TS/iterations (see §10 artifact note; visible in vectors T02/T10).
- desSpeed = `veh->getLane()->getVehicleMaxSpeed(veh)` at every call site
  (IDM.cpp:133, 184, 114, 119) — see §5 for its exact form (this is where
  speedFactor enters).

## 3. Entry points

### 3.1 followSpeed (IDM.cpp:130-134)
```
applyHeadwayAndSpeedDifferencePerceptionErrors(veh, speed, gap2pred, predSpeed, ...)  # no-op without driverState, BASE.cpp:1098-1131
return _v(veh, gap2pred, speed, predSpeed, veh->getLane()->getVehicleMaxSpeed(veh))   # respectMinGap=true
```
- `predMaxDecel` (leader apparentDecel) is **ignored** by IDM (also
  `leaderMaxPredSpeed` does not exist in this API version).
- No decel cap, no minNextSpeed clamp applied here.

### 3.2 stopSpeed (IDM.cpp:178-200)
```
applyHeadwayPerceptionError(veh, speed, gap)          # no-op default, BASE.cpp:1134-1163
if gap < 0.01: return 0                               # :181-183
result = _v(veh, gap, speed, 0, laneVehMaxSpeed, respectMinGap=false)  # :184
if gap > 0 and speed < NUMERICAL_EPS and result < NUMERICAL_EPS:       # :186
    result = maximumSafeStopSpeed(gap, decel, speed, false,
                                  veh->getActionStepLengthSecs())      # :189 ("switching to krauss")
if gap >= 0: result = min(result, DIST2SPEED(gap))                     # :192-193 (no overshoot of stop line)
return result
```
- Callers pass the **net gap** (e.g. VEH.cpp:2520 `leader.second - minGap`;
  stop-line distance at VEH.cpp:2636/2759/2904). minGap is NOT added back.
- `decel` argument = myDecel via the inline overload (BASE.h:189-190); used only
  in the maximumSafeStopSpeed fallback.
- The fallback headway is the **action step length**, not tau (:189).
- Near-stop creep is prevented by the `min(result, gap/TS)` clamp (Euler: cannot
  cross the stop line within one step).

### 3.3 freeSpeed (IDM.cpp:104-127) — speed-limit changes / red lights
```
if maxSpeed < 0: return maxSpeed                     # :106-109 (ballistic red-light convention)
secGap = getSecureGap(maxSpeed, 0)                   # :110  (§3.4)
if speed <= maxSpeed:
    vSafe = _v(veh, 1e6, speed, maxSpeed, laneVehMaxSpeed, respectMinGap=false)   # :114 (free accelerate)
else:
    vSafe = _v(veh, MAX2(seen, secGap), speed, 0, laneVehMaxSpeed, respectMinGap=false)  # :119 (decelerate to the transition point)
if seen < secGap: vSafe = min(vSafe, maxSpeed)       # :121-124 (avoid overshoot)
return vSafe
```
- desSpeed stays the **current** lane's vehicle max speed even when maxSpeed is
  the next limit (:114/:119).
- In the decelerate branch the leader is a virtual stopped object at the
  transition point (predSpeed=0) at distance `MAX2(seen, secGap)`.
- The static `MSCFModel::freeSpeed` red-light formula (BASE.cpp:112-174, with
  `actionStepLength` in the ballistic branch :150-171) is NOT used by IDM.

### 3.4 getSecureGap (IDM.cpp:217-221)
`max(0, speed*T + speed*(speed-leaderSpeed)/(2*sqrt(a*b)))` — same expression as
`s*` without minGap. Used by freeSpeed (:110) and lane changing (LANE.cpp:502,550,581).

### 3.5 insertionFollowSpeed (IDM.cpp:137-162), insertionStopSpeed (IDM.cpp:165-175)
Used at insertion only. insertionFollowSpeed returns `speed` unchanged if
`speed*T + speed*(speed-predSpeed)/(2*sqrt(a*b)) <= gap2pred` (:140-143), else
iterates followSpeed on its own output until the per-step speed drop is below
`ACCEL2SPEED(1) = TS` m/s (:151-159; recursive). insertionStopSpeed iterates the
base `insertionStopSpeed` (BASE.cpp:374-381; Euler branch = stopSpeed) until the
drop per iteration ≤ `decel*TS`, max 10 iterations (:168-174).

### 3.6 minNextSpeed (IDM.cpp:79-89)
`decel_eff = MAX2(myDecel, MIN2(myEmergencyDecel, 1.5))` — i.e. with defaults
`max(4.5, min(9, 1.5)) = 4.5`. Euler: `max(speed - decel_eff*TS, 0)`; ballistic:
`speed - decel_eff*TS` (may go negative = stop-within-step flag).
minNextSpeedEmergency (BASE.cpp:340-348): `max(speed - bmax*TS, 0)` (Euler).
maxNextSpeed (BASE.cpp:323-326, NOT overridden by IDM): `min(speed + a*TS, vType.maxSpeed)`.

## 4. finalizeSpeed — how a raw safe speed becomes vNext

Chain per action step (VEH.cpp:4692-4697 → IDM.cpp:93-101 → BASE.cpp:199-265):

```
vPos (input)  = safe-speed bound from planMove:
                min over { followSpeed/stopSpeed to each leader (VEH.cpp:3341,3395,3337),
                           link approach speeds, freeSpeed at limit changes/red lights,
                           arrival speeds }  (VEH.cpp:2295 initialises v=maximumLaneSpeedCF)
vNext = IDM.finalizeSpeed(veh, vPos):
   vNext = BASE.finalizeSpeed(veh, vPos)           # IDM.cpp:95
   # (IDMM level-of-service update skipped when adaptationFactor==1, IDM.cpp:96-99)

BASE.finalizeSpeed (BASE.cpp:199-265):
   oldV = veh.speed                                                   # :201
   vStop = min(vPos, processNextStop(vPos))                           # :203 (identity without stops)
   vMinEmergency = minNextSpeedEmergency(oldV)                        # :205
   vMin = min(minNextSpeed(oldV), max(vPos, vMinEmergency))           # :207
   factor = 1 if friction==1 else quadratic polyfit                   # :208-210
   aMax = (MAX2(laneVehMaxSpeed, vPos)*factor - oldV) / actionStepLengthSecs   # :215  <-- ASL, not TS!
   vMax = min(oldV + aMax*TS, maxNextSpeed(oldV), vStop)              # :217
   vMax = max(vMin, vMax)                                             # :231
   vNext = patchSpeedBeforeLC(vMin, vMax) = vMax                      # :232, BASE.h:120-124
   vNext = LC.patchSpeed(vMin, vNext, vMax)   # identity by default
   vNext = applyStartupDelay(vNext)           # identity, startupDelay default 0, BASE.cpp:269-285
```

Then at VEH level (VEH.cpp:4692-4727):
- `if vNext > 0: vNext = max(vNext, vSafeMin)` (:4694-4696) — junction minimum.
- `if |vNext| < NUMERICAL_EPS_SPEED (= 1e-4*TS, VEH.cpp:127) and stopDist > POSITION_EPS: vNext = 0` (:4701-4703).
- Euler: `vNext = max(vNext, 0)` (:4723-4724).
- Movement integration (VEH.cpp:4953-4967, 3710-3730): Euler
  `pos += vNext*TS` (the NEW speed is applied during the coming step);
  ballistic `pos += (v + vNext)/2*TS` (negative vNext = stop inside step).
- Non-action steps (actionStepLength > TS): planMove returns early
  (VEH.cpp:2173-2182) and the vehicle continues with the previous plan's
  acceleration: `vSafe = speed + myAcceleration*TS` (VEH.cpp:4654-4656) — the
  cfmodel is NOT called; no reactivity between action points.

### Practical identity for a follower with one leader and no junction constraints
```
vPos = min( followSpeed(v, netGap, vL), laneVehMaxSpeed )          # VEH.cpp:2295 + 3352
vNext = finalize_speed(vPos)  # §4; with defaults:
      # vMin = min(max(v - max(b, min(bmax,1.5))*TS, 0), max(vPos, max(v-bmax*TS,0)))
      # vMax = min(v + (max(laneVehMaxSpeed, vPos) - v)/ASL * TS, min(v + a*TS, vTypeMax), vPos)
      # vNext = max(vMin, vMax)
```
Note `vMin` allows braking beyond decel only up to emergencyDecel per step;
`_v` itself may request more than decel but less than emergencyDecel*TS/step
typically. `aMax` limits ACCELERATION: ego cannot exceed
`max(laneVehMaxSpeed, vPos)` reached linearly over the action step.

## 5. speedFactor application (exact)

IDM never multiplies by speedFactor directly. It reads
`veh->getLane()->getVehicleMaxSpeed(veh)` as desSpeed:
- LANE.h:575-592: `MIN2(vehMaxSpeed, laneLimit * veh->getChosenSpeedFactor())`
  (with per-vClass lane restrictions `r`: `MIN2(vehMaxSpeed, r*vClass * sf)`)
- vehMaxSpeed = `MSBaseVehicle::getMaxSpeed()` = `MIN2(vType.maxSpeed,
  desiredMaxSpeed*speedFactor)` (BASEVEH.cpp:211-213)
- individual speedFactor: `pars.speedFactor < 0 ? vtype-sampled : pars.speedFactor`
  (BASEVEH.cpp:109); sampled from vType distribution `norm/normc(mean,dev)`
  once at insertion (BASEVEH.cpp:2776) and written to tripinfo (BASEVEH.cpp:1268).

Therefore, with defaults (maxSpeed 55.56, desiredMaxSpeed 277.8):
**desSpeed = speedFactor * laneSpeedLimit**. Both lanes'
`getVehicleMaxSpeed` are pre-multiplied — a vType with speedFactor>1 exceeds
limit changes by the same factor.

planMove additionally uses `laneMaxV = MAX2(laneMaxV, vMinComfortable)` (VEH.cpp:2289-2290)
so a sudden limit drop does not force emergency braking, and initialises
`v = cfModel.maximumLaneSpeedCF(this, maxV, laneMaxV)` (VEH.cpp:2295; base
implementation clamps to laneMaxV) before taking the min with leader/link speeds.

## 6. actionStepLength semantics (at step length 0.5 s)

Parsing/rounding (PARSER.cpp:1772-1789): milliseconds = `round(given*1000)`;
`<=0 → DELTA_T`; else if not a multiple of DELTA_T →
`floor-to-multiple`, floored at DELTA_T, with a warning. At TS=0.5 s:
`0.3→0.5(+warn), 0.5→0.5, 0.75→0.5, 1.0→1.0, 2.0→2.0`. Per-vType attribute or
global option `--default.action-step-length` (docs index.md:26-38).

Gating (VEH.cpp:2115-2122, VEH.h:635-636): action happens iff
`(t - myLastActionTime) % actionStepLength == 0` (integer ms). On action steps
planMoveInternal runs (cfmodel called); between action points the vehicle holds
its last planned acceleration (VEH.cpp:4654-4656) and does NOT sense new leaders.
At insertion `myLastActionTime = current step + DELTA_T` (VEH.cpp:5793).

Coupled effects (important for exact rebuild):
1. **aMax divides by ASL** (BASE.cpp:215): acceleration is budgeted over the
   whole action interval, so with ASL=1.0, TS=0.5 the max per-step speed
   increase is `aMax*0.5 = ((max(des, vPos)-v)/1.0)*0.5` — i.e. effectively
   accel-limited in *desired-speed* terms but relaxed vs the TS=ASL case.
2. **stopSpeed fallback headway = ASL** (IDM.cpp:189): near stops, the
   Krauss-style safe stop speed is computed with headway=ASL.
3. **Auto ballistic**: if a vType sets ASL ≠ DELTA_T and `--step-method.ballistic`
   was not set, SUMO switches the whole simulation to ballistic update
   (VT.cpp:463-476) and warns. With ASL==TS (default) Euler stays.
   → Rebuild note: to mimic SUMO with ASL≠TS one must also switch the position
   update to ballistic (`pos += (v+vNext)/2*TS`, VEH.cpp:4957-4963).
4. Warning if ASL > tau ("may lead to collisions", VT.cpp:452-462); docs note
   (index.md:41): at its action step the vehicle still reacts to the previous
   simulation step's state, so perception-reaction granularity stays TS.

## 7. What IDM does NOT do (frequent mis-modellings)

- No decel cap in followSpeed: braking is only limited by
  `minNextSpeed(Emergency)` in finalizeSpeed (§4); `_v` may demand more.
- leader apparentDecel ignored (IDM.cpp:131 signature uses it only for the
  perception-error hook).
- minGap is **desired**, not enforced: `myCollisionMinGapFactor=0.1`
  (IDM.cpp:46-47) means collisions are only registered below 0.1*minGap
  (LANE.cpp:1999-2004); near-zero gaps are clamped to NUMERICAL_EPS in `_v`.
- No dawdle/randomness anywhere in IDM.
- equilibrium gap: `s* = s0 + v*T + v*dv/(2*sqrt(a*b))` (with minGap added
  back when respectMinGap=true), but the intra-loop gap pre-advance (§2) and
  Euler kinematics shift the realised equilibrium slightly.

## 8. Out of scope (defaults documented as no-ops)

driverState perception errors (BASE.cpp:1089-1163), IDMM adaptation
(IDM.cpp:231-234), stops/processNextStop (VEH.cpp:~5800), TraCI influencer
(VEH.cpp:2302-2321), lane-change patchSpeed, startupDelay>0, friction≠1,
maxAccel/desAccel profiles (BASE.cpp:65-66, BASE.cpp:1166-1177),
insertion (`departSpeed` procedures), junction right-of-way (separate helper
deliverable).

## 9. ref_idm.py API map

| function | SUMO source |
|---|---|
| `IDMParams(ts, ...)` | §1 defaults; iterations/two_sqrt_ab (IDM.cpp:44-45) |
| `IDMParams.desired_speed(limit)` | LANE.h:575-592 + BASEVEH.cpp:211-213 |
| `_v(...)` | IDM.cpp:224-269 |
| `get_secure_gap` | IDM.cpp:217-221 |
| `follow_speed` | IDM.cpp:130-134 |
| `stop_speed` | IDM.cpp:178-200 (+ BASE.cpp:806-884 fallback) |
| `free_speed` | IDM.cpp:104-127 |
| `maximum_safe_stop_speed(_euler)`, `calculate_emergency_deceleration` | BASE.cpp:806-884, 1037-1086 |
| `min_next_speed`, `min_next_speed_emergency`, `max_next_speed` | IDM.cpp:79-89, BASE.cpp:340-348, 323-326 |
| `finalize_speed` | BASE.cpp:199-265 (+ IDM.cpp:93-101) |
| `idm_step` | §4 identity, one leader, no junctions |
| `process_action_step_length`, `action_step_active` | PARSER.cpp:1772-1789, VEH.h:635-636 |

## 10. Numeric spot checks (hand-verified against ref_idm.py; full set in test_vectors.json)

Defaults: a=2.6, b=4.5, bmax=9, T=1.0, s0=2.5, delta=4, stepping=0.25, TS=0.5
→ iterations=2, 2sqrt(ab)=6.841052550594828, desSpeed=13.888888888888889
(50 km/h lane, sf=1).

1. **Free accelerate** (freeSpeed accelerate branch, T11: `_v(1e6, 10, 13.889,
   des, rMG=false)`; note predSpeed=maxSpeed here, so dv<0 shrinks s*):
   - sub-step 1: s1 = 10*1 + 10*(10-13.889)/6.841 = 4.315364690;
     (10/13.889)^4 = 0.268503; acc1 = 2.6*(1-0.268503-~0) = 1.901279744;
     v1 = 10 + 1.901279744*0.25 = 10.475319936.
   - sub-step 2: s2 = 5.248313325; acc2 = 1.758658100;
     v2 = 10.914984461 m/s = `free_speed(...)` = test vector T11.
2. **Follower at s* = tau*v** (A5: v=8, leader 8, netGap=8 → gap = s0+s* = 10.5,
   respectMinGap=true): sub-step 1: s1 = 8+2.5 = 10.5 → interaction term = 1
   exactly → acc1 = -2.6*(8/13.889)^4 = -0.286195817 → v1 = 7.928451046.
   Sub-step 2 (dv = v1-8 = -0.071549): s2 = 10.345529247, acc2 = -0.200156760
   → v2 = **7.878411856** = `follow_speed(P0, 8, 8, 8, 50/3.6)`. The realised
   decel over the step (0.263 m/s²) exceeds the drive term -a*(v/des)^4/TS
   because s2/gap dropped below 1 while v dropped less.
3. **stopSpeed near-standstill** (T08: v=0.0005, gap=1.0, respectMinGap=false):
   sub-step 1: acc1 ≈ +2.6 (s1≈0.0005, term≈0) → v1 = 0.650499837,
   gap pre-advanced to 0.837375041; sub-step 2: s2 = 0.712354360,
   acc2 = 0.718395317 → **0.830101782** (v2 differs in the 5th decimal from the
   exact loop because gap clamps; code is authoritative). The
   maximumSafeStopSpeed fallback (IDM.cpp:186-189) does NOT trigger here
   because `_v` returned ≥ NUMERICAL_EPS. It triggers when `_v` returns < 0.001,
   e.g. gap=0.02: `stop_speed(P0, 0.0005, 0.02) = 0.038` — exactly
   `maximumSafeStopSpeed(0.02, 4.5, 0.0005, headway=ASL=0.5)` = r = (0.019-0)/
   (0*0.5+0.5) = 0.038 (n=floor(...)=0). Direct call
   `maximum_safe_stop_speed(gap=1.0, decel=4.5, v=0, headway=0.5)` = 1.998
   (n=0 branch, r = 0.999/0.5; relaxEmergency not engaged since origSafeDecel<0).
4. **finalizeSpeed vMin floor** (T16: v=10, netGap=8, leader braking to 0):
   followSpeed → 5.145655423; minNextSpeed(10) = 7.75; vMin =
   min(7.75, max(5.1457, 10-9*0.5=5.5)) = **5.5**; vMax = 5.1457 →
   vNext = max(vMin, vMax) = 5.5: IDM may brake beyond decel down to the
   emergencyDecel rate per step (Euler), never below.

All 20 vectors live in `test_vectors.json` (inputs + expected vNext from
`ref_idm.py`); `run_tests.py` re-checks them, plus 8 independent closed-form
invariants A1-A8 (single-sub-step formulas, two-sub-step equilibrium A5,
no-overshoot of stop lines A4, ASL rounding A6, iteration counts A7).

**Mid-loop zero-clamp artifact** (§2, from IDM.cpp:259): if v hits 0 at a
sub-step boundary (strong braking), the NEXT sub-step sees v=0 → s*=0 →
interaction 0 → acc=+a → v jumps back to a*TS/iterations within the SAME step
(visible in T02 = 0.5778 and T10 = 0.65, both = a*TS/2 after braking to 0 in
sub-step 1). This is faithful SUMO behaviour, not a bug.

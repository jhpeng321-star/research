# SUMO 1.27.1 merge / lane-end / insertion behaviour — implementable spec

Author: rbh-mergespec (helper), 2026-10-31/11-01. Target consumers: S-RB-MERGE (primary, `src/wms/physics/junction_merge.py`), S-RB-LANE, S-RB-INT, validators.
Companion artifacts in this folder: `test_cases.md` (scenario matrix), `mergespec_tests.py` (pure-python reference formulas + numeric vectors, runnable, no SUMO), `_sumo_src/` (downloaded SUMO source files, flat names `src__<path-with-__>.cpp`).

Provenance: SUMO tag `v1_27_1` (github.com/eclipse-sumo/sumo), downloaded read-only into `_sumo_src/`; every claim below cites `file:line` of that tag. Docs quotes cite sumo.dlr.de. Nothing was executed; no SUMO runs.

Conventions: TS = simulation step (default 1 s); Euler update is the SUMO default (`MSGlobals::gSemiImplicitEulerUpdate == true`); `POSITION_EPS = 0.01 m`; speeds m/s, distances m, accelerations m/s². `brakeGapEuler(v, b, h) = SPEED2DIST(n*v - dv*n*(n+1)/2) + v*h` with `n = int(v / (b*TS))`, `dv = b*TS` (MSCFModel.cpp:101-108); ballistic form `v*(h + 0.5*v/b)` (MSCFModel.cpp:90-97).

---

## 1. Parameter defaults (SUMO "physics", allowed to hard-code)

### 1.1 Default vType (docs: sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html, "Available vType Attributes"; code SUMOVTypeParameter.cpp:940,973,1036)

| param | default | code |
|---|---|---|
| carFollowModel | Krauss (we target IDM; see §5) | docs |
| laneChangeModel | LC2013 | docs |
| length / minGap | 5.0 / 2.5 | docs |
| width | 1.8 | docs |
| maxSpeed | 55.55 (vClass-dependent) | docs |
| accel / decel | 2.6 / 4.5 | SUMOVTypeParameter.cpp:940,973 |
| emergencyDecel | max(decel, 9.0) | SUMOVTypeParameter.cpp:979-1014 |
| apparentDecel | = decel | docs |
| sigma (driver imperfection) | 0.5 | SUMOVTypeParameter.cpp:1036 |
| tau (headway) | 1.0 | docs |
| speedFactor | normc(1.0, 0.1, 0.2, 2) (passenger) | docs |

### 1.2 Departure defaults (docs same page, "Available Vehicle attributes"; code MSEdge.cpp:62-63, SUMOVehicleParameter.cpp:40-43, MSLane.cpp:654-658,678-685)

| attribute | default | note |
|---|---|---|
| departLane | `"first"` = rightmost lane the vehicle may use | MSEdge.cpp:62 (`myDefaultDepartLaneDefinition(DEFAULT)`, `myDefaultDepartLane(0)`) → FIRST_ALLOWED → getFirstAllowed (MSEdge.cpp:741-762) |
| departPos | `"base"` (front at pos = length) | docs; MSLane.cpp:737-758 (`veh.basePos(myEdge)`) |
| departSpeed | `0` | MSLane.cpp:654-658 (`speed = 0 was set before`, patchSpeed=false) |
| departPosLat | `"center"` (0) | MSLane.cpp:678-685 |
| arrivalLane / arrivalPos / arrivalSpeed | `current` / `max` / `current` | docs |
| insertionChecks | `all` | docs; per-vehicle attr; bitmask `InsertionCheck` used in MSLane.cpp:1233-1235,1256,1330,1348,1384 |
| max-depart-delay | option `--max-depart-delay`, vehicle discarded after it | MSInsertionControl.cpp:168-171 |

### 1.3 LC2013 defaults (MSLCM_LC2013.cpp:108-122; MSAbstractLaneChangeModel.cpp:139,145,147)

| param | default |
|---|---|
| lcStrategic | 1.0 |
| lcCooperative | 1.0 (also reused for lcCooperativeRoundabout, lcCooperativeSpeed) |
| lcSpeedGain / lcKeepRight / lcOpposite | 1.0 |
| lcLookaheadLeft | 2.0 |
| lcSpeedGainRight | 0.1 |
| lcSpeedGainRemainTime | 20 |
| lcSpeedGainUrgency | 50 |
| lcKeepRightAcceptanceTime | -1 |
| lcStrategicLookahead | UNDEFINED (disabled) |
| lcSigma | 0.0 (code default; docs mention 0.5 for the *CF* sigma — do not conflate) |
| lcAssertive | 1.0 |

### 1.4 Hard-coded model constants (LC2013 / junction)

| constant | value | site |
|---|---|---|
| LOOK_FORWARD | 10 s | MSLCM_LC2013.cpp:48 |
| JAM_FACTOR | 1.0 | MSLCM_LC2013.cpp:50 |
| LOOK_AHEAD_MIN_SPEED | 0.0 | MSLCM_LC2013.cpp:55 |
| LOOK_AHEAD_SPEED_MEMORY | 0.9 / actionStep | MSLCM_LC2013.cpp:56 |
| URGENCY (strategic) | 2.0 | MSLCM_LC2013.cpp:64 |
| OPPOSITE_URGENCY | 5.0 | MSLCM_LC2013.cpp:65 |
| TURN_LANE_DIST | 200 m | MSLCM_LC2013.cpp:71 |
| REACT_TO_STOPPED_DISTANCE | 100 m | MSLCM_LC2013.cpp:75 |
| MSLink::myLookaheadTime | 1 s | MSLink.cpp:72 |
| MSLink::myLookaheadTimeZipper | 16 s | MSLink.cpp:74 |
| DIST_TO_STOPLINE_EXPECT_PRIORITY | 1.0 m (default jmStoplineGap for major links) | MSVehicle.cpp:125,2791 |
| jmStoplineGapMinor default | 0 | MSVehicle.cpp:2814 |
| jmTimegapMinor default | 1 s (myLookaheadTime) | MSLink.cpp:1063 |
| jmAllwaystopWait / jmStopsignWait default | 1 step (TS) | MSLink.cpp:944-948 |
| BEST_LANE_LOOKAHEAD | 3000 m (departLane=best cap) | MSEdge.cpp:54 |

### 1.5 IDM parameters (MSCFModel_IDM.cpp:38-51)

`delta = 4`, substep `stepping = 0.25 s` → `iterations = TS/0.25 = 4` at TS=1, `collisionMinGapFactor = 0.1`, `2*sqrt(accel*decel)` precomputed. IDMM only: adaptFactor 1.8, adaptTime 600 (not used here).

---

## 2. Insertion (departure) pipeline

### 2.1 Call chain (who does what)

1. `MSInsertionControl::emitVehicles` (MSInsertionControl.cpp:127-152) walks pending emits, oldest first, and calls `tryInsert` per vehicle per step.
2. `MSInsertionControl::tryInsert` (156-188): `MSEdge::insertVehicle(veh, time, checkOnly=false, forceCheck)`. On failure: if `time - depart > maxDepartDelay` → vehicle deleted; else it waits one step (line 183-185). `myLastFailedInsertionTime`/`myFailedInsertionMemory` (MSEdge.cpp:901-921): once a lane rejected a vehicle in a step, later vehicles skip that lane that step (order matters for platoons).
3. `MSEdge::insertVehicle` (MSEdge.cpp:803-922): route validity; `validateDepartSpeed` (766-799 — for `departSpeed=GIVEN` above lane max, speedFactor is raised or it errors); lane choice via `getDepartLane` (§2.2); then `MSLane::insertVehicle` (§2.4).

### 2.2 departLane resolution — `MSEdge::getDepartLane` (MSEdge.cpp:677-751)

| departLane | behaviour |
|---|---|
| `>=0` (GIVEN) | that lane if it allows vClass, else fail (686-689) |
| `random` | uniform over allowed lanes (690-691); only one tested per step |
| `free` | `getFreeLane(nullptr, vClass, departPosBound)` (692-693) |
| `allowed` (ALLOWED_FREE) | free lane among lanes that allow continuation to next route edge (694-699) |
| `best` (BEST_FREE) | among best lanes (below), the free one (733-734) |
| `best_prob` (BEST_PROB) | as best, plus never insert left of a vehicle with larger speedFactor (736, getProbableLane 579-613) |
| `first` (default) | rightmost allowed lane (741-762) |

`getFreeLane` (547-575): pick lane maximizing `lastVehiclePos - departPos` (gap ahead of insertion point; empty lane ⇒ `edgeLength - departPos`); if none has positive gap, least brutto-occupancy.
`best lanes` (700-731): from the vehicle's LaneQ continuations (`updateBestLanes`), keep lanes whose continuation length ≥ max length; cap length at `BEST_LANE_LOOKAHEAD = 3000` minus departPos (713-717); keep vClass-allowed.
`departPosBound` (617-661): base → 0/edgeLength; GIVEN → the value; LAST → 0..min last-vehicle pos.

### 2.3 departSpeed resolution — `MSLane::getDepartSpeed` (MSLane.cpp:605-661)

`speed` initial 0; `patchSpeed` = whether downstream checks may *lower* the speed (§2.5).

| departSpeed | speed | patchSpeed |
|---|---|---|
| `0`/default | 0 | false |
| `>=0` (GIVEN) | given value | false |
| `random` | uniform [0, laneMax] | true |
| `max` | `getVehicleMaxSpeed` = min(laneLimit * chosenSpeedFactor, vType maxSpeed * speedFactor) | true |
| `desired` | same as max | false |
| `speedLimit` (LIMIT) | laneLimit / chosenSpeedFactor (≈ raw limit) | false |
| `last` | min(laneMax, speed of last vehicle on lane) | false if a vehicle present |
| `avg` | min(laneMax, lane mean speed) | false if a vehicle present |

`LAST`/`AVG` fall back to DESIRED when the lane is empty (docs; the MIN2-with-laneMax in code achieves it). Note `departSpeed=max` *may* be lowered by insertion checks (docs: "the speed may be lowered to accommodate a leader vehicle or intersections"), while `desired`/`speedLimit`/`last`/`avg` delay the departure instead (§2.5).

### 2.4 Position choice — `MSLane::insertVehicle` (MSLane.cpp:691-814)

- departPos `base`/default → `pos = veh.basePos(edge)` (= vehicle length: back at lane start, 757). GIVEN negative → +laneLength (702-704). `random` → uniform (706-708). `random_free` → 10 random tries then `freeInsertion` (709-725). `free` → `freeInsertion` (726-727): scan from behind-last-vehicle forward, at each pair (follower, leader) try `pos = followPos + backGapNeeded + POSITION_EPS` where `backGapNeeded = follower.secureGap(follower→ego)` (561-598); fallback behind-last (546-558); `last` → `lastInsertion` (481-512): at lane end or behind leader minus `secureGap(ego→leader) + minGap + POSITION_EPS` (502-505).
- `departPosLat` free/random_free (substep lat scanning) at 762-789; default center = 0.
- Then `isInsertionSuccess(..., NOTIFICATION_DEPARTED)` (791). On success with substep depart delay, the vehicle may be moved fractionally forward (`myExtrapolateSubstepDepart`, 797-812).

### 2.5 `MSLane::isInsertionSuccess` (MSLane.cpp:851-1420) — the acceptance oracle

Ordered checks; all operate on the *tentatively* placed vehicle (`setTentativeLaneAndPosition`, 875):

0. pos outside [0, len] → clamp to lane end with warning (855-860).
1. Setup: `seen = len - pos` (distance from insertion to end of lane); horizon `dist = brakeGap(speed) + minGap` (880-881). `dist` is the lookahead window: everything within it must be safely stoppable.
2. Scheduled stop on the lane (943-961): required speed `stopSpeed(speed, endPos - pos)` (or `freeSpeed` for a waypoint); must be stoppable else fail (`InsertionCheck::STOP`).
3. Leader on the lane (963-984): `nspeed = safeInsertionSpeed(veh, -pos, leaders, speed)` (1434-1470): for each leader, `gap = leaderBack + seen - minGap` (i.e. net gap ego-front to leader-back); `gap < 0` → INVALID_SPEED (collision) or 0 depending on `insertionChecks::COLLISION` (1450-1461); else `nspeed = min(nspeed, insertionFollowSpeed(ego, speed, gap, leaderSpeed, leaderMaxDecel, leader))` (1462-1463).
4. Route-continuation walk (997-1226): follow bestLane continuation links while `seen < dist`:
   - lane does not continue for the route (junction unreachable/last edge): require ability to stop before it: `insertionStopSpeed(ego, speed, seen)` → `checkFailure(..., JUNCTION)` (1021-1028); on the last edge only if the junction may be blocked (`mayContinue && hasUnsafeLink`, 1013-1020).
   - link not opened / no priority / `LINKSTATE_ZIPPER` / minor (1052-1056): must be able to stop before the junction (insertionStopSpeed at `remaining = seen - laneStopOffset`, 1070-1090); hard error (departure descheduled) when state is MINOR/EQUAL/STOP/ALLWAY_STOP and speed is fixed ("unpriorised junction too close", 1058-1065).
   - next lane internal → junction leader check via `checkLinkLeader` (1101-1124).
   - leader on next lane (1157-1183): same `safeInsertionSpeed` logic with `seen` offset.
   - pedestrians on next lane (1184-1186); next-lane speed limit (1187-1206, lowers speed when patchSpeed allowed); approaching junction foes: `link->hasApproachingFoe(arrivalTime, leaveTime, speed, maxDecel)` (1207-1216, MSLink.cpp:1288-1296) — insertion yields to committed junction traffic.
5. Followers (1228-1254): for each follower on consecutive lanes (incl. shadow), `backGapNeeded = follower.secureGap(follower→ego, followerSpeed, egoSpeed, egoMaxDecel)`; fail if `gap < backGapNeeded` (FOLLOWER_GAP) or `gap < 0` (COLLISION). This is the "will the follower hit me" rule — asymmetric: a *slower than follower* ego needs a larger gap.
6. Shadow-lane leader/follower checks (1260-1324); `missingRearGap` from predecessor lanes when followers are free (1325-1347).
7. `speed < 0` → fail (1351-1366); `insertionChecks == NONE` clamps speed to ≥ 0 without failing (1348-1350).
8. Lane-change reservation (1367-1389): `extraReservation = LCModel.getExtraReservation(bestLaneOffset)` (20 m if bestLaneOffset < -1, 40 m if > 1, 0 otherwise; only when the neighbour lane is not longer: MSAbstractLaneChangeModel.cpp:1178-1188) — insertion near a place where ≥2 more lane changes are needed must be able to stop at `distToStop = bestLaneDist - pos - extraReservation - speed*actionStep` (1373-1388, LANECHANGE check).
9. Accept: `incorporateVehicle` (MSLane.cpp:456-477) at the sorted position.

### 2.6 `checkFailure` — how fixed departSpeed delays departure (MSLane.cpp:818-847)

If a check produced `nspeed < speed`:
- `patchSpeed=true` → `speed = min(nspeed, speed)`, extend `dist` to the new brakeGap; insertion proceeds at the lower speed (this is how `departSpeed=max` degrades gracefully).
- `patchSpeed=false` (given/desired/limit/last/avg) → by default the check *fails* (return true ⇒ insertion false) unless the vehicle's `insertionChecks` mask disables it; with option `--emergency-insert`, insertion is allowed if emergency braking (`0.5*v²/emergencyDecel`) can stop within `dist` (827-836). Fatal messages deschedule the vehicle (838-843).

### 2.7 What an on-ramp insertion looks like (summary for S-RB-MERGE)

Ramp vehicle with `departSpeed=max`, `departLane=best/free` on a 1-lane ramp:
- enters at `pos = len` (base) if empty; speed starts at min(ramp limit × speedFactor, vType max);
- is lowered only by: leader on ramp (IDM insertionFollowSpeed), stop-ability before the merge junction (`insertionStopSpeed`) if the merge link is zipper/minor, or an approaching committed foe at the merge (`hasApproachingFoe`);
- never checks the *mainline* — insertion is lane-local (plus internal-lane leaders). Post-merge interaction is purely the junction machinery of §4.

---

## 3. Lane-end forcing (strategic urgency + stopping)

State: LC2013 computes per vehicle a `bestLanes` set (LaneQ): for the current lane chain, `length` = drivable distance along the route before a mandatory change, `bestLaneOffset` = signed number of lane changes to the route continuation (0 = on it), `occupation` = queued jam length. This is the sole source of "lane ends soon".

### 3.1 Look-ahead distance — when does the lane count as "ending"

`MSLCM_LC2013::_wantsChange` (MSLCM_LC2013.cpp:1096-1333):
- `myLookAheadSpeed` (1235-1243): grows instantly to ego speed; decays with memory factor `0.9` per action step; floor 0.
- `laDist = myLookAheadSpeed * LOOK_FORWARD(10) * lcStrategic * (1 if right else lcLookaheadLeft) + 2 * lengthWithGap` (1244-1245). With defaults (lengthWithGap 7.5): rightward `laDist = 10*v_look + 15`; leftward `20*v_look + 15` (the lookahead doubles, the +15 constant is added once).
- `lcStrategic < 0` disables strategic changes (`laDist = -1e3`, 1275-1276).
- Special laDist overrides: bidi leader (0, 1250-1252); stopped leader on own lane → half the space needed to pass it (1253-1259); stopped leader on the best target lane (1260-1273).
- `usableDist = max(currentDist - posOnLane - occupation, driveToNextStop)` (1294).
- **Trigger** (1330-1333): if the target direction is toward the best lane (`changeToBest`) and `usableDist / |bestLaneOffset| < laDist` (`currentDistDisallows`, MSLCM_LC2013.h:197-199) ⇒ `LCA_STRATEGIC | LCA_URGENT` (plus direction flag). I.e. with defaults, a vehicle with one required change starts "urgent" strategic attempts when remaining drivable distance < `10 * v_look + 15` m (rightward; `20 * v_look + 15` leftward).
- Additional strategic rules: stopped-leader avoidance → urgent strategic (1377-1400); guards that prevent moving *away* from the best lane when it cannot be recovered (`neighLeftPlace * 2 < laDist` → LCA_STAY|STRATEGIC, 1401-1424; TURN_LANE_DIST=200 for turn lanes, 1425-1443).

### 3.2 Urgency magnitude and courtesy messages

When `LCA_URGENT` (1470-1545):
- `myLeftSpace = currentDist - posOnLane` (space left before the mandatory-change point) (1473).
- Unseen-blocker reservation: if `|bestLaneOffset| > 1`, reserve `max(extraReservation, blocker)` length (1474-1482, getExtraReservation §2.5-8); counter-direction leaders/followers wanting the opposite change reserve `myLeadingBlockerLength` via `MSLCHelper::updateBlockerLength` (1484-1500); if reservation impossible ⇒ `myDontBrake = canContinue` (1496-1500).
- `remainingSeconds = max(TS, myLeftSpace / max(v_look, eps) / max(1, |bestLaneOffset|) / URGENCY(2.0))` (1502-1507). This scalar is the "urgency" passed to `informLeader`/`informFollower` (1509-1517): the neighbour computes its speed advice as if it had this many seconds before ego stops.
- For TraCI-requested changes the influencer's remaining seconds are used instead (1507).

### 3.3 Speed adaptation while urgent (patchSpeed)

`MSLCM_LC2013::_patchSpeed` (234-432), called each step with (min, wanted, max):
- Leading-blocker space reservation (247-281): if `myLeadingBlockerLength > 0`, `space = myLeftSpace - blocker - POSITION_EPS`; `safe = stopSpeed(space)`; `max = min(max, max(safe, vMinEmergency))`; if `safe < wanted` adopt `nVSafe = max(min, safe)` — i.e. the merging vehicle *brakes to open a hole for the blocker* (this is the classic ramp courtesy slowdown of the mainline vehicle when *it* is the one that must change into a blocked lane; on the ramp side the analog is `informLeader` line 478: `stopSpeed(myLeftSpace - blocker)`).
- Accumulated LC speed advices (284-322) cap `nVSafe` (own advices unscaled, neighbour advices blended with `lcCooperativeSpeed` weight, 283, 295-300).
- State-based fallbacks: `LCA_WANTS_LANECHANGE && LCA_BLOCKED && LCA_STRATEGIC` → `(max + wanted) / 2` (speed up, "no vSafe advice", 334-343); cooperative blocked-by-leader → `(max(0,min) + wanted)/2`, blocked-by-follower → `(max+wanted)/2` (344-365); `LCA_AMBLOCKINGLEADER` → `(max+wanted)/2` (404-411).
- On a single-lane edge, change info is reset (427-431).

### 3.4 Physical stop at the lane end (no route beyond the lane)

`MSVehicle::planMoveInternal` (MSVehicle.cpp:2265-3110):
- If the lane (or shadow lane in sublane mode) is a dead end for the bestLane continuation (`lane->isLinkEnd(link)`, 2755): `va = stopSpeed(ego, v, seen)` where `seen` = distance to lane end; `v = min(v, va)` (2759-2767); the planned drive item ends there (2775-2778). `myLaneChangeModel->getCommittedSpeed()` (set during a committed continuous LC) overrides (2763-2764). **This is the mechanism that stops a vehicle at the end of an acceleration lane if it failed to merge** — SUMO has no special "ramp meter": it is an ordinary stop target at `seen` with the CF model's stopSpeed.
- Stop-line offsets for junction stops (2790-2829): major priority links stop at `jmStoplineGap` (default 1 m, DIST_TO_STOPLINE_EXPECT_PRIORITY); minor links at `max(jmStoplineGapMinor, jmStoplineCrossingGap - distToFoePedCrossing)`, never below `foeVisibilityDistance`; and `laneStopOffset = min(laneStopOffset, seen - brakeDist)` when `canBrakeBeforeLaneEnd` (i.e. `seen >= brakeGap(v, stopDecel, 0)`, 2793-2795, 2824-2827) — avoid emergency braking if comfortable braking suffices.
- Continuous LC must finish before the shadow lane ends: `va = max(stopSpeed(seen - eps), (seen - eps)/timeRemaining)` (2870-2895).

### 3.5 Observable macro-behaviour (for validation targets)

- A ramp vehicle that cannot merge decelerates along the ending lane (stopSpeed profile, Euler: discrete decel = `decel` per step until halt), queues at the ramp end, and re-attempts every step (LC decision each action step; gap acceptance by the *mainline follower* — see §4.4 follower-side rules). Teleporting (jam removal) is the only other relief valve (option `--time-to-teleport`, not modelled here).

---

## 4. Junction / zipper / priority rules at an on-ramp merge

### 4.1 Link & junction machinery

- Each (lane → next lane) transition is an `MSLink` with `state` from the net (TL green/red, `LINKSTATE_MINOR/EQUAL/STOP/ALLWAY_STOP`, `LINKSTATE_ZIPPER`), `myFoeLinks` (conflicting links), `myFoeLanes`, cont flag, `myFoeVisibilityDistance`, built by `MSRightOfWayJunction::init` from the junction logic request/response matrices (MSRightOfWayJunction.cpp:100-183, `setRequestInformation` at 152; internal-junction foes appended 100-151). A **zipper** junction is a right-of-way junction whose links all carry `LINKSTATE_ZIPPER` and each other as foes (state consumed at MSLink.cpp:938,1175,2167; MSVehicle.cpp:3882).
- Approaching vehicles register `arrivalTime, leaveTime, arrivalSpeed, leaveSpeed, arrivalSpeedBraking, willPass(=setRequest), dist` on every foe link each step: `MSVehicle::setApproachingForAllLinks` (MSVehicle.cpp:5482-5550) → `MSLink::setApproaching` (MSLink.cpp:717-736). `leaveTime = arrivalTime + (linkLength + vehLen) / max(0.5*(arr+leave speed), eps)` (MSLink.cpp:830-833). `arrivalTime` comes from `getArrivalTime` (MSVehicle.cpp:3158-3173) via `MSCFModel::getMinimalArrivalTime(dist, v, arrivalSpeed)` (MSCFModel.cpp:461-465). `arrivalSpeedBraking` = speed after covering `dist - v*tau` at max decel (`estimateSpeedAfterDistance`, MSCFModel.cpp:573-586, 797-802: `sqrt(max(0, 2*d*a + v²))` capped by vType maxSpeed).

### 4.2 Right of way — `MSLink::opened` (MSLink.cpp:837-1022)

1. red → closed (845-847); cont (already past internal junction) → open (848-850).
2. sublane sibling-foe checks (852-924): among links from the same edge with crossing paths, the *later arrival must yield*; equal arrival → the one further **left** yields (860-866).
3. Priority short-circuit (935-943): if ego has priority (`havePriority()` or last-was-TL-green-major) and state is **not** ZIPPER → open (collectFoes empty). **Zipper links never take this shortcut — they always collect and check foes.**
4. `LINKSTATE_ALLWAY_STOP`: closed until `waitingTime > jmAllwaystopWait` (default TS), then first-come by foe waiting time (944-945, 1132-1153). `LINKSTATE_STOP`: closed until `jmStopsignWait` (946-948). This is the *give-way ramp* case.
5. Otherwise (minor/equal, and zipper): blocked if **any** foe link has an approaching vehicle that blocks at ego's time window: `blockedAtTime` (1025-1109) → `blockedByFoe` (§4.4).

### 4.3 The merge time-window rule — `MSLink::blockedByFoe` (MSLink.cpp:1112-1221)

Given ego `[arrivalTime, leaveTime]` and foe `[foeArrival, foeLeave]`, lookahead `L = jmTimegapMinor (default 1 s)`, but **`L = myLookaheadTimeZipper = 16 s` for zipper links** (1175-1179; MSLink.cpp:72-74). Foe considered only if `foe.willPass` (it has issued a request) (1129-1131). Impatience pulls the foe's arrival earlier assuming it brakes with `maxDecel * impatience` (`computeFoeArrivalTimeBraking`, 1154-1172, 1224-1285).

- Foe leaves before ego arrives (`foeLeave < arrivalTime`): ego becomes **follower**. Blocked iff same target lane AND (`arrivalTime - foeLeave < L` OR `unsafeMergeSpeeds(foeLeaveSpeed, egoArrivalSpeed, foeDecel, egoDecel)`) (1188-1199). Different target lanes (a crossing merge) → not blocked by trailing-gap.
- Foe arrives after ego leaves (`foeArrival > leaveTime + L`): ego becomes **leader**. Blocked iff same target lane AND `unsafeMergeSpeeds(egoLeaveSpeed, foeBrakingArrivalSpeed, egoDecel, foeDecel)` (1200-1210).
- Otherwise: **hard conflict → blocked** (1211-1219).

`unsafeMergeSpeeds(leaderSpeed, followerSpeed, leaderDecel, followerDecel) = leaderSpeed²/leaderDecel <= followerSpeed²/followerDecel` (MSLink.h:730-733): the follower cannot stop in the leader's stopping distance.

### 4.4 Zipper speed adaptation — `MSLink::getZipperSpeed` (MSLink.cpp:2157-2304)

Applied at move time (`MSVehicle::processLinkApproaches`, MSVehicle.cpp:3882-3884) when the link is not opened and `state == LINKSTATE_ZIPPER`:

1. If `dist > max(foeVisibilityDistance, brakeGap(vSafe))` → no adaptation (2166-2174). Else evaluate each collected foe.
2. Skip foe if (2211-2231):
   - foe arrives after ego AND ego cannot brake for it (`couldBrakeForLeader(egoDist, foeDist, ego, foe)`, 2307-2313: `followDist > leaderDist AND followDist - leaderDist > followSpeed - followMaxDecel - leaderSpeed`), or
   - foe is behind and *can* brake for ego, or
   - tie in arrival and distance → the vehicle on the **higher lane index (more left)** yields (ego continues if `ego.laneIndex < foe.laneIndex`).
3. Else compute a zipper-safe speed:
   - foe end speed `uEnd = min(foeLaneMax, estimateSpeedAfterDistance(foeDist, foeArrSpeed, +maxAccel))`; `tf = ceil(foeDist / uAvg / TS) * TS` with `uAvg = (foeArrSpeed + uEnd)/2`, min 1 s (2246-2251);
   - ego end speed `vEnd = min3(egoLaneMax, estSpeed(+maxAccel), max(uEnd, estSpeed(-maxDecel)))`; `te` analog (2253-2259);
   - `tTarget = tf + egoHeadway` (2261); `a = avoidArrivalAccel(egoDist, tTarget, vSafe, maxDecel)` (2262; formula MSCFModel.cpp:553-567: `dist<=0 → -maxDecel; time*v > 2*dist → -v²/(2*dist); else 2*(dist/time - v)/time`);
   - `gap = egoDist - foeLength - egoMinGap - foeDist`; `vFollow = followSpeed(ego, v, gap, foeArrSpeed, foeMaxDecel, foe)` (2264-2266);
   - `vSafeGap = max(vFollow, v + TS*a)` (2267);
   - blend deceleration harshness with proximity: `w = min(1, te/10)`, `maxDecel_blend = w*maxDecel + (1-w)*emergencyDecel` (2269-2271);
   - **`vZipper = max3(vFollow, v - TS*maxDecel_blend, vSafeGap)`** (2272); `vSafe = min(vSafe, vZipper)` (2274).

Net effect: within visibility/brake distance of the zipper, both ramp and mainline vehicles trim speed so that whoever arrives first can be followed with `tau` headway; ties favour the right-hand lane; the correction is gentle far away (emergency decel blend near).

### 4.5 Priority (give-way) merge behaviour

- Minor-link vehicle plans a stop at `seen - laneStopOffset` (§3.4 offsets; default `jmStoplineGapMinor = 0`, MSVehicle.cpp:2814) with `vLinkWait = stopSpeed(stopDist, stopDecel)` (2904-2905) and only issues a request (`setRequest`) if it can still move (`v > eps` and not aborted after an earlier minor stop, 2897-2902).
- At move time (`processLinkApproaches`, MSVehicle.cpp:3831-3865): if not opened and foe presence undetermined (`dist > foeVisibilityDistance`) and braking still possible → wait at `vLinkWait`; if past the point of no return (`canBrake == false`, `impatience` forced to 1, 3785-3788) → it must clear the junction: `vSafeMin` ≈ speed to cover the remaining link distance (3853-3858) — prevents dawdling on the crossover.
- Waiting vehicles block *behind* the stop line and re-check every step; a blocked vehicle aborts its request when blocked by a sibling/stopped/long-waiting leader (JUNCTION_BLOCKAGE_TIME) to avoid deadlock (checkLinkLeader, MSVehicle.cpp:3660-3691).
- Junction leaders: `checkLinkLeader` (3573-3707) treats the first vehicle on the conflict area as leader (`adaptToJunctionLeader`, 3375-3485: followSpeed at gap, else stop at crossing point, else follow-into-gap after leader passes with `vFinal = 2*(distToCrossing - minGap)/leaderPastCPTime - v` capped by `stopSpeed`); oncoming vehicles split free gap in proportion to brakeGaps (`adaptToOncomingLeader`, 3488-3553).

### 4.6 Insertion-time junction check (already covered in §2.5-4): a vehicle may only depart if it can stop before a closed/minor/zipper junction or if the junction is clear of committed foes (`hasApproachingFoe`, MSLane.cpp:1052-1090, 1207-1216).

---

## 5. Car-following pieces used by merge (IDM target)

Full IDM numerics: `rb-shared/idmspec/` (S-RB-CF). Only the merge-relevant surface:

- `_v` (MSCFModel_IDM.cpp:224-269): `s*(gap) = max(0, v*tau + v*(v-vL)/(2√(a·b)))` (+ minGap when `respectMinGap`); `acc = a*(1 - (v/v_des)^4 - (s/gap)²)`; integrated with 4 substeps (stepping 0.25 s); `gap` shrinks within the substep loop (266).
- `followSpeed = _v(gap, v, vL, laneMax)` (130-134); `stopSpeed` (178-200): `_v(gap, v, 0)`; `gap < 0.01 → 0`; near halt switch to Krauss `maximumSafeStopSpeed` (186-190); clamp `result <= gap/TS` (192-197).
- `freeSpeed` (104-127): accelerate branch `_v(1e6, v, maxSpeed)`; decelerate branch `_v(max(seen, secGap), v, 0)`; clamp to target when `seen < secGap`.
- `getSecureGap` (217-221): `max(0, v*tau + v*(v-vL)/(2√(a·b)))` — *no* leader-decel term (contrast Krauss base, MSCFModel.cpp:177-195: `max(0, brakeGap(v, myDecel, tau) - brakeGap(vL, max(myDecel, leaderDecel), 0))`).
- `insertionFollowSpeed` (137-162): if `gap >= s(v)` return `speed` unchanged (allows full-speed insertion at large gaps); else fixed-point iterate `followSpeed` until the speed change per iteration < 1 m/s step — yields the *actually* safe insertion speed rather than one step of braking.
- `insertionStopSpeed` (165-175): iterate base `insertionStopSpeed` until within one step of `myDecel` braking (equilibrium insertion near stops).
- Base-Krauss fallbacks (MSCFModel.cpp): `insertionFollowSpeed = maximumSafeFollowSpeed(gap, (Euler ? v : 0), vL, predMaxDecel, onInsertion)` (363-371); `insertionStopSpeed` (374-381).

## 6. Implementable per-step algorithm (what S-RB-MERGE should reproduce)

Per vehicle per step (Euler, TS=1):
1. **planMove**: walk continuation lanes; candidate stops = route stops, arrival, dead-end lane end (stopSpeed at seen), junction stop lines (offsets §3.4); for each link compute (arrivalTime, vLinkPass, vLinkWait) with `opened()`/foe checks; keep first binding stop.
2. **junction follower-speed**: apply §4.4 zipper speed (if zipper, within visibility), §4.5 minor-link wait speeds.
3. **lane change decision** (each action step): §3.1 strategic trigger; if urgent compute remainingSeconds (§3.2), exchange speed advices with neighbour leader/follower (§3.3), attempt change if accepted (gap acceptance with target-leader `insertionFollowSpeed`-style and target-follower `getSecureGap`; full LC acceptance mechanics belong to lcspec/S-RB-LANE).
4. **CF speed**: min over leader-followSpeed, stop constraints, LC advices; clamp by accel/decel; move.
5. **Insertion attempts** (departing vehicles): §2.5 oracle; patch speed or retry next step.

Named constants to expose (all §1): `URGENCY=2.0`, `LOOK_FORWARD=10`, `LOOK_AHEAD_SPEED_MEMORY=0.9`, `lcLookaheadLeft=2`, `zipperLookahead=16 s`, `minorLookahead=1 s`, `EXTRA_RESERVATION_LEFT=-1→20 m / >1→40 m`.

## 7. Known gaps / notes for integrators

- Best-lane computation (`MSVehicle::updateBestLanes`) lives in MSVehicle.cpp (LaneQ) — S-RB-LANE owns it; this spec treats `currentDist`/`bestLaneOffset` as inputs.
- `MSLCHelper::updateBlockerLength` (courtesy blocker reservation) details are in `_sumo_src/src__microsim__lcmodels__MSLCHelper.cpp`; summarized here, not fully specified.
- Pedestrian checks (MSLane.cpp:4620+) are out of scope (no pedestrians in the WM scenarios).
- `foeVisibilityDistance` is a per-link net attribute (netconvert); for generated networks record what the net sets (affects zipper visibility + minor-stop offsets).
- LC2013 `lcSigma` default is 0.0 in code (MSAbstractLaneChangeModel.cpp:145) although some docs mention 0.5; use the code value.
- Departure is *ordered*: vehicles with earlier depart time are tried first each step, and a lane that rejected once is skipped for the rest of that step (MSEdge.cpp:901-921) — this creates the bursty "emit or wait a whole step" insertion pattern at saturated ramps.

# SUMO 1.27.1 LC2013 Lane-Change Model — Implementable Spec + Test Vectors

Author: rbh-lcspec (helper). Date: 2026-11-01.
Scope: exact decision logic of `MSLCM_LC2013` (SUMO v1_27_1 tag), written so a from-scratch
reimplementation (e.g. `src/wms/physics/lc2013.py`) can be built and tested without SUMO.

All line references are to the GitHub tag `v1_27_1` of eclipse-sumo/sumo. Local copies of the
cited files were fetched read-only into `E:/agent-work/_sol/sumo-src-1271/` (never executed).
Web doc: lane-changing vType attributes table
https://sumo.dlr.de/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.html
(repo copy: `docs/web/docs/Definition_of_Vehicles,_Vehicle_Types,_and_Routes.md`, cited below
as VTypes_routes.md; the old `Simulation/Lane_Change_Model` wiki page 404s as of this tag).

Primary sources:
- `src/microsim/lcmodels/MSLCM_LC2013.cpp` (LC13:x) — model logic
- `src/microsim/lcmodels/MSLCM_LC2013.h` (LC13H:x) — members
- `src/microsim/lcmodels/MSAbstractLaneChangeModel.cpp` (ALCM:x) — base defaults/helpers
- `src/microsim/MSLaneChanger.cpp` (CHG:x) — gap-acceptance gate + commit
- `src/microsim/lcmodels/MSLCHelper.cpp` (HELP:x) — blockers/roundabout/bidi helpers
- `src/microsim/cfmodels/MSCFModel.cpp` (CF:x), `MSCFModel_IDM.cpp` (IDM:x) — secure gap
- `tools/traci/constants.py:1469-1544` — canonical `LCA_*` bit values

---

## 1. Execution pipeline (who calls what, in order)

Per simulation step, `MSLaneChanger::laneChange(t)` (CHG:190) runs `change()` (CHG:330) for one
candidate vehicle per edge, on its action step:

1. `getRealLeader/getRealFollower` on current lane; `getRealLeader/getRealFollower` on the
   target lane (= `neighLead`, `neighFollow`) (CHG:781-796). If neighLead == neighFollow,
   neighFollow is cleared (CHG:789-793).
2. `checkChange(laneOffset, ...)` (CHG:798) computes the `blocked` bitset (see §7), then calls
   `wantsChange(...)` → `MSLCM_LC2013::_wantsChange` (LC13:1096) and ORs in `blocked`.
3. Post-gates in `checkChange`: critical-leader re-check (CHG:960-986), zipper-link safety
   (CHG:987-1004), continuous-LC feasibility `LCA_INSUFFICIENT_SPACE` / `LCA_INSUFFICIENT_SPEED`
   (CHG:1006-1115, only when `--lanechange.duration > step`), entry-link openness (CHG:1117-1135),
   TraCI influence, `checkChangeBeforeCommitting` (CHG:1153-1156).
4. Commit rule (CHG:399, 416): change executes iff
   `(state & LCA_WANTS_LANECHANGE) != 0 && (state & LCA_BLOCKED) == 0`.
   `LCA_WANTS_LANECHANGE = LCA_LEFT | LCA_RIGHT` (constants.py:1511).
   **Right is checked first** (CHG:396-409 before CHG:413-426); if BOTH directions are urgent,
   right wins (`stateLeft = 0`, CHG:428-432).
5. A vehicle whose request was urgent but blocked is registered as `lastBlocked`/`firstBlocked`
   on the neighbor lane's ChangeElem (CHG:403-408, 420-425). Those pointers are fed back into
   `wantsChange` of vehicles on that neighbor lane, where `slowDownForBlocked()` (LC13:1955)
   makes them decelerate to open a gap (`LCA_AMBACKBLOCKER` / `LCA_AMBACKBLOCKER_STANDING`).

Default update scheme assumed throughout: ballistic (`--step-method.ballistic`), step length
TS = 1 s, action step length = 1 s, instant lane change (`--lanechange.duration` default 0).
For euler mode the few differing branches are flagged inline.

## 2. State (per vehicle, persists across steps)

| member | init | reset by `changed()` (LC13:1063) |
|---|---|---|
| `mySpeedGainProbabilityLeft` (fixed-point int) | 0 | 0 |
| `mySpeedGainProbabilityRight` (fixed-point int) | 0 | 0 |
| `myKeepRightProbability` (fixed-point int, negative = want right) | 0 | 0 |
| `myLookAheadSpeed` | 0 (`LOOK_AHEAD_MIN_SPEED`, LC13:55) | 0 |
| `myOwnState` (LCA_* bits) | 0 | 0 |
| `myLeadingBlockerLength`, `myLeftSpace` | 0 | 0 (unless bestLaneOffset!=0, LC13:1069-1074) |
| `myLCAccelerationAdvices` | empty | empty |

`HYST_PRECISION = 1e7` (LC13:78): all probabilities are floats × 1e7 stored as `long long`.

`prepareStep()` (LC13:1023-1060): while not changing lanes, `myOwnState` keeps only
`LCA_STRATEGIC` (anded with `LCA_WANTS_LANECHANGE`) — i.e. strategic motivation persists one
step, everything else (incl. AMBLOCKING*) is cleared; blocker/length/bookkeeping cleared;
lateral `mySigma` OU-noise applied if `lcSigma > 0` (default 0 → none).

### LCA_* bit values (tools/traci/constants.py:1469-1544; mirrored by the C++ enum)

```
LCA_STAY=1<<0  LCA_LEFT=1<<1  LCA_RIGHT=1<<2
LCA_STRATEGIC=1<<3  LCA_COOPERATIVE=1<<4  LCA_SPEEDGAIN=1<<5  LCA_KEEPRIGHT=1<<6
LCA_TRACI=1<<7  LCA_URGENT=1<<8
LCA_BLOCKED_BY_LEFT_LEADER=1<<9   LCA_BLOCKED_BY_LEFT_FOLLOWER=1<<10
LCA_BLOCKED_BY_RIGHT_LEADER=1<<11 LCA_BLOCKED_BY_RIGHT_FOLLOWER=1<<12
LCA_OVERLAPPING=1<<13 LCA_INSUFFICIENT_SPACE=1<<14 LCA_SUBLANE=1<<15
LCA_AMBLOCKINGLEADER=1<<16 LCA_AMBLOCKINGFOLLOWER=1<<17
LCA_MRIGHT=1<<18 LCA_MLEFT=1<<19
LCA_AMBLOCKINGFOLLOWER_DONTBRAKE=1<<21 LCA_CHANGE_TO_HELP=1<<23
LCA_AMBACKBLOCKER=1<<26 LCA_AMBACKBLOCKER_STANDING=1<<27
LCA_INSUFFICIENT_SPEED=1<<28
# composites
LCA_WANTS_LANECHANGE = LEFT|RIGHT = 6
LCA_BLOCKED_BY_LEADER   = 1<<9 | 1<<11
LCA_BLOCKED_BY_FOLLOWER = 1<<10 | 1<<12
LCA_BLOCKED = BLOCKED_BY_LEADER | BLOCKED_BY_FOLLOWER | INSUFFICIENT_SPACE | INSUFFICIENT_SPEED
```

## 3. Parameters (defaults; LC13:108-122, ALCM:139-148, VTypes_routes.md:955-985)

| param (vType attr) | default | used for |
|---|---|---|
| lcStrategic | 1.0 | scales strategic lookahead (×; 0 → lookahead 0, −1 → never strategic, LC13:1275) |
| lcCooperative | 1.0 | [0,1]; coop-change inconvenience gate, default of the two below |
| lcSpeedGain | 1.0 | divides speedGain/keepRight thresholds |
| lcKeepRight | 1.0 | multiplies keepRight probability |
| lcOpposite | 1.0 | safety factor for opposite-direction driving |
| lcLookaheadLeft | 2.0 | left strategic lookahead factor |
| lcSpeedGainRight | 0.1 | right speedGain threshold asymmetry |
| lcSpeedGainLookahead | 0 (LC2013) | anticipate slow leader (s); 0 = off |
| lcSpeedGainRemainTime | 20 s | min drive time on new lane after speedGain change |
| lcSpeedGainUrgency | 50 | speedGain probability → LCA_URGENT |
| lcKeepRightAcceptanceTime | −1 (legacy) | −1 → legacy `7 * speedFactor * max(1,v)` formula |
| lcOvertakeDeltaSpeedFactor | 0 | informLeader pre-overtake on neighbor lane |
| lcCooperativeRoundabout | = lcCooperative | roundabout inner-lane bonus |
| lcCooperativeSpeed | = lcCooperative | weight of others' speed advice |
| lcAssertive | 1 | divides required secure gaps (via getSafetyFactor) |
| lcSigma | 0.0 | lateral OU noise |
| lcOvertakeRight | 0 | probability to allow overtake-right |
| lcMaxSpeedLatStanding | maxSpeedLat | lateral speed while standing |
| lcMaxSpeedLatFactor | 1.0 | bound `lcMaxSpeedLatStanding + f*speed` |
| lcMaxDistLatStanding | 1.6 (0 for 2-wheelers) | anti-slide while standing |
| lcCooperativeHelpTime | 60 s | getCooperativeHelpTime (ALCM:148) |
| lcStrategicLookahead | 3000 m (UNDEFINED_LOOKAHEAD=−1 → 3000) | bestLanes only, not _wantsChange |

### Hard-coded constants (LC13:48-77 unless noted)

```
LOOK_FORWARD=10.0            JAM_FACTOR=1.0
LOOK_AHEAD_MIN_SPEED=0.0     LOOK_AHEAD_SPEED_MEMORY=0.9
HELP_DECEL_FACTOR=1.0        HELP_OVERTAKE=10/3.6≈2.7778   MIN_FALLBEHIND=7/3.6≈1.9444
RELGAIN_NORMALIZATION_MIN_SPEED=10.0
URGENCY=2.0                  OPPOSITE_URGENCY=5.0
KEEP_RIGHT_TIME=5.0          KEEP_RIGHT_HEADWAY=2.0 (unused in LC2013)
MAX_ONRAMP_LENGTH=200.0      TURN_LANE_DIST=200.0
REACT_TO_STOPPED_DISTANCE=100  BLOCKER_IS_BLOCKED_TIME_THRESHOLD=5
HYST_PRECISION=1e7           LC_RESOLUTION_SPEED_LAT=0.5 (LC13:73)
highway on-ramp guard lane speed: > 80/3.6 (LC13:1621)
congested(): both lanes speedLimit > 70/3.6 and both vehicles congested (ALCM:269-284)
predInteraction(): leader speed > 80/3.6 and gap < interactionGap (ALCM:299-308)
```

## 4. Strategic motivation (wish + urgency) — LC13:1233-1277, 1290-1445

**Look-ahead speed memory** (LC13:1235-1243):

```
if v > laSpeed:  laSpeed = v                       # growth instantaneous
else:            m = 1 - (1-0.9)*actionStepLen     # 0.9 per 1 s
                 laSpeed = max(0, m*laSpeed + (1-m)*v)
```

**Look-ahead distance** (LC13:1244-1245):

```
laDist = laSpeed * 10 * lcStrategic * (1 if right else lcLookaheadLeft) + 2*lengthWithGap
```
- if a bidi leader blocks and bestLaneOffset==0: laDist = 0 (LC13:1250)
- if a stopped leader is on the current lane (and bestLaneOffset==0):
  `laDist = 0.5*(lenWithGap_ego + lenWithGap_leader + gap_leader)` (LC13:1253-1259)
- lcStrategic < 0 → laDist = −1e3 (never strategic)

**Distances** (LaneQ from `updateBestLanes`, recomputed per new lane; LC13:1111-1160):
- `currentDist` / `neighDist` = drivable length along the route continuation of the
  current / target lane (includes subsequent lanes of the best-lanes continuation),
  plus `driveToNextStop` extension if the next stop is on this edge (LC13:1167-1188).
- roundabout bonus (LC13:1288-1292, HELP:42-238): on multi-exit roundabouts adds
  `distanceInRoundabout * jamFactor * lcCooperativeRoundabout * 9` to both distances;
  0 off-roundabout.
- `usableDist = max(currentDist - pos - best.occupation * 1.0, driveToNextStop)` (LC13:1294)
- `neighLeftPlace = max(0, neighDist - pos - maxJam)` (LC13:1303; capped at stopped
  neighLead gap, LC13:1304-1306)

**bestLaneOffset resolution** (LC13:1132-1160): `bestLaneOffset` = offset of the best lane
from the current lane (positive = left). When examining direction `laneOffset` and BOTH the
current and neighbor entries have `bestLaneOffset == 0`, it is set to `laneOffset` itself.
`changeToBest = (right && bestLaneOffset<0) || (!right && bestLaneOffset>0)` (LC13:1192).

**Urgent strategic change** (LC13:1329-1333):

```
if changeToBest && bestLaneOffset == curr.bestLaneOffset
   && usableDist / |bestLaneOffset| < laDist:      # currentDistDisallows, LC13H:197
       return LCA_<dir> | LCA_STRATEGIC | LCA_URGENT
```

**Overtake stopped leader** (LC13:1373-1400): if the leader is stopped (or bidi) within
`REACT_TO_STOPPED_DISTANCE=100 m` and the target lane (or a lane beyond it, `hasFreeLane`
LC13:2086) leaves room to pass (`MIN2(neighDist, currentDist) - pos > overtakeDist` with
`overtakeDist = leaderGap + ego.length + leader.lengthWithGap`), the same urgent-strategic
state is returned and `currentDist` is shortened to the stopped leader.

**Strategic stay** (in order, LC13:1401-1444):
- `!changeToBest && neighLeftPlace/(|bestLaneOffset|+2) < laDist` → `LCA_STAY|LCA_STRATEGIC`
- `bestLaneOffset==0 && 2*neighLeftPlace < laDist` (cannot come back) → `LCA_STAY|LCA_STRATEGIC`
- VARIANT_21 stayOnBest: target lane leads elsewhere (`bestContinuations` differ),
  `neighDist < TURN_LANE_DIST=200` and bestLaneOffset==0, leader not stopped, no roundabout
  → `LCA_STAY|LCA_STRATEGIC`

**Urgent bookkeeping** (LC13:1470-1545): on `LCA_URGENT`:
```
myLeftSpace = currentDist - pos
remainingSeconds = max(TS, myLeftSpace / max(laSpeed,eps) / max(1,|bestLaneOffset|) / URGENCY)
                   # URGENCY=2.0 (5.0 when driving opposite); TraCI request → its own remaining seconds
```
then `informLeader(...)` and `informFollower(...)` negotiation runs (§6), and the saved
(canceled) state is returned.

## 5. Speed-gain and keep-right motivations — LC13:1547-1898

**Anticipated lane speeds** (LC13:1549-1552, ALCM:1901-1952):

```
thisLaneVSafe  = min(vMax,          anticipateFollowSpeed(leader,     currentDist, ...))
neighLaneVSafe = min(neighVMax,     anticipateFollowSpeed(neighLead,  neighDist,   ...))

anticipateFollowSpeed(lead, dist, vMax, acceleratingLeader):
  if acceleratingLeader:                       # any leader acc > 0 (LC13:1549)
     maxSpeed1s = v + a_max - a_max            # ≈ v  (see LC13:1908)
     futureSpeed = lead==nullptr ? followSpeed(ego, maxSpeed1s, dist, 0, 0)
                                 : followSpeed(ego, maxSpeed1s, gap, lead.v, lead.maxDecel)
  else:
     futureSpeed = lead==nullptr ? maximumSafeStopSpeed(dist, maxDecel, v, onInsertion=true)
                                 : maximumSafeFollowSpeed(gap, v, lead.v, lead.maxDecel, onInsertion=true)
  futureSpeed = min(vMax, futureSpeed)
  # optional slow-leader forecast when lcSpeedGainLookahead > 0 (default 0): LC13:1934-1950
```

**Convenience gate** (LC13:1561-1569):

```
relSpeedDiff  = (thisLaneVSafe==0) ? 0 : (thisLaneVSafe - neighLaneVSafe) / max(thisLaneVSafe, neighLaneVSafe)
inconvenience = clamp01( max( relSpeedDiff, probOppositeDirection/threshOppositeDirection ) )
speedGainInconvenient = inconvenience > lcCooperative
```
with (LC13:1561-1563): examining right → `probLeft/threshRight`; examining left →
`probRight/threshLeft`.

**relativeGain** (LC13:1694-1695):

```
relativeGain = (neighLaneVSafe - thisLaneVSafe) / max(neighLaneVSafe, 10.0)
```

**Right examination** (LC13:1710-1817):
1. if `thisLaneVSafe - 5/3.6 > neighLaneVSafe` (current lane ≥5 km/h faster):
   `probRight *= 0.5^asl` (decay)
2. else: `probRight += round(asl * relativeGain * 1e7)`
   and keep-right accumulation (below); if `probRight * lcKeepRight < -threshRight`
   → return `LCA_RIGHT | LCA_KEEPRIGHT` (never urgent)
3. speedGain commit (LC13:1808-1817):
   `probRight > threshRight && neighDist/max(0.1, v) > lcSpeedGainRemainTime`
   → `LCA_RIGHT | LCA_SPEEDGAIN` (+ `LCA_URGENT` iff `probRight > lcSpeedGainUrgency*1e7`)

**Left examination** (LC13:1818-1872):
1. `thisVSafe > neighVSafe` → `probLeft *= 0.5^asl`;
   `thisVSafe == neighVSafe` → `probLeft *= 0.8^asl`;
   else `probLeft += round(asl * relativeGain * 1e7)`
2. commit: `probLeft > threshLeft && (relativeGain > eps || changeLeftToAvoidOvertakeRight)
   && neighDist/max(0.1,v) > lcSpeedGainRemainTime` → `LCA_LEFT|LCA_SPEEDGAIN`
   (+URGENT iff `probLeft > lcSpeedGainUrgency*1e7`); when urgent AND blocked by the left
   follower, ask that follower to slow to `max(minNextSpeed_f, v_ego − 1)` (LC13:1861-1868)

**Thresholds** (LC13:142-151):

```
if lcSpeedGain <= 0:  threshRight = threshLeft = LLONG_MAX   (feature off)
else:
  threshRight = (0.2 / lcSpeedGainRight) / lcSpeedGain * 1e7      # default (0.2/0.1) = 2.0e7
  threshLeft  = (0.2 / 1)               / lcSpeedGain * 1e7      # default 2.0e6
```

**Keep-right accumulation** (legacy default path, LC13:1720-1790, only in the right/else
branch above):

```
roadSpeedFactor = vMax / laneSpeedLimit            # ≤ 1 normally (LC13:1721)
acceptanceTime  = 7 * roadSpeedFactor * max(1.0, v)          # lcKeepRightAcceptanceTime == -1
                # non-legacy: lcKeepRightAcceptanceTime * roadSpeedFactor, then reduced if a
                # faster follower is within 2*its brakeGap (LC13:1729-1738)
fullSpeedGap    = max(0, neighDist - brakeGap(vMax))
fullSpeedDrivingSeconds = min(acceptanceTime, fullSpeedGap / vMax)
if neighLead with v < vMax:
    fullSpeedGap = min(fullSpeedGap, max(0, neighLeadGap - getSecureGap(ego, neighLead, vMax, neighLead.v, neighLead.maxDecel)))
    fullSpeedDrivingSeconds = min(..., fullSpeedGap / (vMax - neighLead.v))
if avoidOvertakeRight and slow current-lane leader: capped by leader gap (LC13:1749-1753)
deltaProb = threshRight * (fullSpeedDrivingSeconds / acceptanceTime) / KEEP_RIGHT_TIME   # 5 s
probKeepRight -= round(asl * deltaProb)
change right  iff  probKeepRight * lcKeepRight < -threshRight
```

**Strategic non-urgent toward best lane** (LC13:1874-1883):
`changeToBest && bestLaneOffset == curr.bestLaneOffset && lcStrategic >= 0 && relativeGain >= 0
&& probDirection > 0` → `LCA_<dir> | LCA_STRATEGIC`.

**On-ramp / dead-end guard** (LC13:1619-1632, runs BEFORE cooperative and speedGain):
examining right, currently on the best lane (`bestLaneOffset == 0` after resolution — i.e. the
neighbor is NOT as good as the current lane) and `laneSpeedLimit > 80/3.6` and `laSpeed >
haltingSpeed` → return `LCA_STAY | LCA_STRATEGIC`.

**Cooperative help** (LC13:1643-1671, runs after the on-ramp guard):

```
if amBlockingFollowerPlusNB()            # myOwnState has AMBLOCKINGFOLLOWER|DONTBRAKE (set by inform)
   && !speedGainInconvenient
   && (myOwnState & LCA_M<dir>)          # the "help" direction matches the changer's request
   && (changeToBest || neighDist/(|bestLaneOffset|+1) > laDist):
       return LCA_<dir> | LCA_COOPERATIVE | LCA_URGENT
```

**Roundabout cooperative** (VARIANT_15, LC13:1586-1616): inside multi-exit roundabouts:
left request `LCA_LEFT|LCA_COOPERATIVE` when moving inward is fine; else `LCA_STAY|LCA_COOPERATIVE`.

## 6. Blocker negotiation (messaging) — LC13:469-1020

Called only within the urgent-strategic block (§4). `dir = LCA_MLEFT/LCA_MRIGHT` (bits 18/19).

**plannedSpeed** (LC13:475-485): `min(v, stopSpeed(ego, v, myLeftSpace - myLeadingBlockerLength))`
further lowered by own acceleration advice.

**informLeader** (LC13:469-657) — target-lane leader nv:
- `neighNextGap` extrapolates the gap 1 s ahead for both ballistic/euler (LC13:498-504)
- not blocked: keep distance — `addLCSpeedAdvice(followSpeed(ego, v, neighNextGap, ...))` (LC13:637-656)
- blocked by leader: decide overtake-vs-follow:
  - `overtakeDistance(follower, leader, gap)` (LC13:456-466):
    `gap + leader.lengthWithGap + follower.length + getSecureGap(leader, follower, leaderSpeed, followerSpeed, follower.maxDecel)`
  - overtake if `plannedSpeed - nv.speed > lcOvertakeDeltaSpeedFactor * laneSpeedLimit`
    (default factor 0 → any dv>0), or the leader waits > 5 s and both lengths fit into
    myLeftSpace; then tell the leader not to accelerate
    (`informNeighLeader(nv.speed, dir|AMBLOCKINGLEADER)` unless it already waits > 5 s,
    LC13:632-634) and return −1
  - otherwise follow: smooth decel at least `MIN_FALLBEHIND=7/3.6` per remaining second,
    `addLCSpeedAdvice(nextSpeed)` (LC13:559-615)

**informFollower** (LC13:659-1020) — target-lane follower nv, when blocked by it:
- cut-in without help if `plannedSpeed − nv.speed >= HELP_OVERTAKE (10/3.6)` and the gap
  closes fast enough; ask nv not to exceed `plannedSpeed − HELP_OVERTAKE` (LC13:689-703)
- else with help decel `helpDecel = nv.maxDecel * HELP_DECEL_FACTOR (1.0)`:
  - euler: `decelGap = neighFollowGap + (plannedSpeed − (nv.speed − helpDecel))`;
    ballistic: `decelGap = gapExtrapolation(1s, gap, ego.v, nv.v, plannedAccel, −helpDecel, ...)` (LC13:721-746)
  - `secureGap = nv.getSecureGap(nv, ego, max(nv.speed − helpDecel, 0), plannedSpeed, ego.maxDecel)`
  - if `decelGap >= secureGap > 0`: request nv to drive `vsafe = followSpeed(...)` (cut in now,
    LC13:775-876)
  - elif the gap opens within remainingSeconds (`dv*(rem−1) > secureGap − decelGap + eps`
    ballistic): request `nv.speed − helpDecel` (cut in eventually, LC13:877-899)
  - else be overtaken: `vhelp = max(nv.speed, ego.speed + HELP_OVERTAKE)`;
    `overtakeDist = overtakeDistance(nv, ego, neighFollowGap, vhelp, plannedSpeed)`;
    `needDV = overtakeDist / remainingSeconds`; slow ego smoothly:
    `addLCSpeedAdvice(max(vhelp − needDV, v − maxDecel))` (LC13:987-1007)
- on-ramp special case (needs lcExperimentalParam1, default 0 → off, LC13:751-773)
- non-blocking follower: `MSLCHelper::getSpeedPreservingSecureGap` (HELP:333-353) — reply with
  the follower's own followSpeed that preserves the secure gap (LC13:1009-1019)

**slowDownForBlocked** (LC13:1197-1200, 1955-2009): for `lastBlocked`/`firstBlocked` vehicles
behind ego on the same lane with gap > POSITION_EPS and ego almost stopped
(`v < maxDecel`): set `LCA_AMBACKBLOCKER[_STANDING]` and follow them at
`followSpeed(ego, v, gap−eps, blocked.v, blocked.maxDecel)`.

**Blocker-space reservation** (LC13:1474-1500, HELP:241-321): when urgent with
`|bestLaneOffset| > 1`, reserve `getExtraReservation` (ALCM:1178-1188: 20 m if 2+ lanes right,
40 m if 2+ lanes left) for unseen blockers; `MSLCHelper::updateBlockerLength` reserves
`blocker.lengthWithGap + ego.minGap` in `myLeadingBlockerLength` when a neighbor that wants to
counter-change fits (`required <= myLeftSpace − brakeGap(v, maxDecel, 0)`), else asks the
blocker to reserve for ego via its `saveBlockerLength` (LC13:2065-2083).

## 7. Secure gaps and the blocked test (gap acceptance)

**Blocked pre-test in `checkChange`** (CHG:818-927), before `wantsChange` is even called:

```
overlap:   neighFollow.gap < 0 → blocked |= BLOCKED_BY_<side>_FOLLOWER | LCA_OVERLAPPING
           neighLead.gap   < 0 → blocked |= BLOCKED_BY_<side>_LEADER   | LCA_OVERLAPPING
           (<side> = RIGHT for laneOffset=-1, LEFT for +1)
tauRemainder = 0 if actionStep == DELTA_T else max(headway − TS, 0)   (CHG:851)
vNextFollower = f.speed + max(0, tauRemainder*f.acc);  vNextLeader(ego) = v + min(0, tauRemainder*acc)

secureBackGap  = followerCF.getSecureGap(follower, ego, vNextFollower, vNextLeader(ego), ego.maxDecel)
blocked_by_follower  iff  neighFollow.gap < secureBackGap  * getSafetyFactor()
secureFrontGap = egoCF.getSecureGap(ego, neighLead, vNextFollower(ego), vNextLeader, lead.maxDecel)
blocked_by_leader    iff  neighLead.gap   < secureFrontGap * getSafetyFactor()
```

`getSafetyFactor() = 1 / lcAssertive` (LC13:2056-2058); opposite driving:
`1/lcOpposite` (LC13:2060-2063).

**MSCFModel::getSecureGap (base, Krauss-style)** (CF:177-195):

```
maxDecel = max(myDecel, leaderMaxDecel)
bgLeader   = brakeGap(leaderSpeed, maxDecel, headway=0)
secureGap  = max(0, brakeGap(egoSpeed, myDecel, myHeadwayTime) - bgLeader)
if gComputeLC and ego.accel < 0:      # reacting next step anyway; lcAssertive enters here
    sg2 = max(0, brakeGap(egoSpeed, min(-ego.accel, myDecel), 0) - bgLeader)
    secureGap = min(secureGap, sg2 / lcAssertive)      # = sg2 * getSafetyFactor()
brakeGap (ballistic, CF:87-98):   v*(headway + 0.5*v/decel);  euler: discrete sum + v*headway (CF:102-108)
```

**IDM override (what our rebuild uses)** (IDM:217-221):

```
secureGap_IDM(egoSpeed v, leaderSpeed vL) = max(0, v*T + v*(v − vL) / (2*sqrt(a*b)))
```
— exactly the IDM desired gap `s*` without minGap. `2*sqrt(a*b) = myTwoSqrtAccelDecel`.

**maximumSafeFollowSpeed / maximumSafeStopSpeed** (used by anticipateFollowSpeed; CF:806-1000):

```
maximumSafeFollowSpeed(gap, ego, pred, predMaxDecel, onInsertion=true):
    maximumSafeStopSpeed(gap + brakeGap(pred, max(myDecel,predMaxDecel), 0), myDecel, ego, onInsertion, headway, relaxEmergency=false)
maximumSafeStopSpeed (ballistic, onInsertion=true):  v0 = −b*τ + sqrt((b*τ)² + 2*b*g),  τ=headway
maximumSafeStopSpeed (ballistic, driving):           v1 = −b*τ/2 + sqrt((b*τ)²/4 + b*(2g − τ*v0)); return v0 + (v1−v0)*TS/τ
```

## 8. patchSpeed (longitudinal adaptation while wishing/blocked) — LC13:233-432

In order:
1. `myLeadingBlockerLength > 0` (counter-LC space reservation, step 2): cap speed with
   `stopSpeed(ego, v, myLeftSpace − myLeadingBlockerLength − POSITION_EPS)`, allow emergency
   braking (LC13:251-281)
2. apply `myLCAccelerationAdvices` (each stored as accel = `SPEED2ACCEL(vSafe − v)`,
   ALCM:1111-1115): own advice used directly; foreign advice blended
   `v*coopWeight + (1−coopWeight)*wanted` with `coopWeight = clamp01(lcCooperativeSpeed)` (LC13:283-331)
3. if `state & LCA_WANTS_LANECHANGE && state & LCA_BLOCKED`:
   - `LCA_STRATEGIC` → return `(max + wanted)/2`  (speed up toward the choke point)
   - `LCA_COOPERATIVE` + blocked by leader → `(max(0,min) + wanted)/2`;
     + blocked by follower → `(max + wanted)/2`
4. `LCA_AMBLOCKINGLEADER && lcCooperativeSpeed >= 0` → return `(max + wanted)/2` (pull ahead)
5. else return `wanted`.

## 9. Ten numeric test vectors

Common setting for all cases: ballistic update, TS = 1 s, actionStep = 1 s, instant lane
change, `gComputeLC = true`, no TraCI. Straight 3-lane edge, lane indices 0=right … 2=left;
ego length 5 m, minGap 2.5 m → lengthWithGap 7.5. CF = IDM with a = 1.5 m/s², b = 4.5 m/s²,
T = 1 s (→ `2*sqrt(a*b) = 5.1962`), maxDecel 4.5. All LC params default. `posOnLane = 0`,
`best.occupation = 0`, no stops, no roundabout, no pedestrians, no bidi. "steps" = consecutive
`_wantsChange` evaluations with unchanged inputs (self-state apart from the accumulated
probabilities). Result flags are the return of `_wantsChange` OR-ed with the changer's
`blocked` bits; "COMMIT" = changer executes the change.

Reference computations used repeatedly:
- `maximumSafeStopSpeedBallistic(onInsertion=true, b=4.5, τ=1)` for gap g:
  `v0 = −4.5 + sqrt(20.25 + 9g)`
- `brakeGap(v, 4.5, 0) = 0.5v²/4.5 = v²/9`;  `brakeGap(v, 4.5, 1) = v(1 + v/9)`
- `secureGap_IDM(v, vL) = max(0, v + v(v−vL)/5.1962)`

| # | name | inputs (beyond common) | expected decision |
|---|---|---|---|
| 1 | strategic urgent, right | ego lane 1, v=20, laSpeed=20; best lane = lane 0 (exit; `curr.bestLaneOffset=−1`); examining right; `currentDist−pos = 150`; no neighbors | `laDist = 20·10·1·1 + 15 = 215 > 150/1` → `LCA_RIGHT|LCA_STRATEGIC|LCA_URGENT`; `myLeftSpace=150`, `remainingSeconds = max(1, 150/20/1/2)=3.75`; target lane empty → **COMMIT right** |
| 2 | strategic not urgent | case 1 with `currentDist−pos = 300` | `300 ≥ 215` → no strategic request; equal lane speeds (both vMax 30, empty lanes) → `relGain=0` → probRight += 0; keepRight accrues (see case 3) but after 1 step prob = −828571 ≮ −2e7 → **no request (ret = 0)** |
| 3 | keepRight after 25 steps | case 2 repeated 25 steps | each step `deltaProb = 2e7·(29/140)/5 = 828571.43` (fullSpeedGap = 1000−brakeGap(30,4.5,1)=1000−130=870; drivingSeconds = min(7·1·20=140, 870/30=29) = 29); after 25 steps prob = −25·828571 = −20 714 275 < −2e7 → **`LCA_RIGHT|LCA_KEEPRIGHT`** (not urgent); after 24 steps (−19 885 704) still no request |
| 4 | speedGain left | ego lane 1, v=25, laSpeed=25, vMax=30 both lanes; current-lane leader gap 0 m at 20 m/s (not accelerating); neighbor lane empty, neighDist=1000; examining left; bestLaneOffset resolves to +1 | thisVSafe = maxSafeFollowSpeed(gap 0, 25, 20) = maxSafeStopSpeed(0 + brakeGap(20,4.5,0)=44.444) = −4.5+sqrt(20.25+400) = 16.0; neighVSafe = maxSafeStopSpeed(1000) ≈ 90.4 → min(vMax)=30; relGain = (30−16)/30 = 0.46667; probLeft += round(1·0.46667·1e7) = 4 666 666 > threshLeft 2e6, relGain > 0, 1000/25 = 40 > 20 s → **`LCA_LEFT|LCA_SPEEDGAIN`** (prob ≪ 5e8 → not urgent); empty lanes → **COMMIT left** |
| 5 | speedGain decay (left blocked-off) | case 4 mirrored: current lane empty (thisVSafe=30), neighbor leader gap 0 at 20 (neighVSafe=16), probLeft = 0 | `thisVSafe(30) > neighVSafe(16)` → probLeft *= 0.5 → 0 → **no request (ret = 0)** |
| 6 | speedGain urgency | case 4 with `probLeft` preloaded to 5.2e8 (setParameter `speedGainProbabilityLeft=52`) | `5.2e8 > threshLeft(2e6)` and `5.2e8 > lcSpeedGainUrgency·1e7 = 5e8` → **`LCA_LEFT|LCA_SPEEDGAIN|LCA_URGENT`**; empty target → **COMMIT** |
| 7 | on-ramp / turn-lane guard | ego lane 1 on best lane, road speedLimit 30 m/s (> 80/3.6), laSpeed 20; examining right where the right lane ENDS in 100 m (right lane's own bestLaneOffset = +1, so resolved `bestLaneOffset` stays 0) | guard at LC13:1621 fires before speedGain/keepRight → **`LCA_STAY|LCA_STRATEGIC`** |
| 8 | secure-gap blocks (follower too close) | case 3 state (keepRight mature, requesting right); target-lane follower 10 m behind ego, follower v=25 (ego v=20, both IDM) | secureBackGap = secureGap_IDM(25, 20) = 25 + 25·5/5.1962 = 49.06; `10 < 49.06·(1/1)` → changer sets `LCA_BLOCKED_BY_RIGHT_FOLLOWER` → **NO commit**; final state `LCA_RIGHT|LCA_KEEPRIGHT|LCA_BLOCKED_BY_RIGHT_FOLLOWER`. Note: keepRight never sets LCA_URGENT, so no informFollower negotiation runs — the wish simply stays blocked (patchSpeed returns `wanted`; the follower itself only reacts once it becomes a blocker via its own model / MSLaneChanger registration) |
| 9 | same, gap sufficient → commit | case 8 with follower gap 60 m | `60 ≥ 49.06` → not blocked, front clear → **`LCA_RIGHT|LCA_KEEPRIGHT` and COMMIT right** |
| 10 | patchSpeed while blocked-strategic | `myOwnState = LCA_RIGHT|LCA_STRATEGIC|LCA_URGENT|LCA_BLOCKED_BY_RIGHT_FOLLOWER` (set via setOwnState after a blocked urgent strategic wish); patchSpeed inputs `max = 30` (lane cap), `wanted = 28` (IDM) | rule LC13:334-343 → return `(30+28)/2 =` **29.0 m/s** (accelerate toward the lane end instead of dawdling) |

Hand-arithmetic notes (verify any case in seconds):
- case 1: `currentDistDisallows(150, −1, 215)` ⇔ `150/|−1| = 150 < 215` ✓
- case 3: `brakeGap(30,4.5,1) = 30·(1 + 30/9) = 130`; `threshRight = (0.2/0.1)/1·1e7 = 2e7`
- case 4: `maxSafeFollowSpeed(0,25,20)`: leader brakeGap = 20²/9 = 44.444; `sqrt(20.25+9·44.444)=sqrt(420.25)=20.5` → exactly 16.0
- case 8: `25·(25−20) = 125`; `125/5.1962 = 24.06`; secureGap = 49.06

Caveats for implementers: (a) the urgent-strategic test compares against
`curr.bestLaneOffset` BEFORE the both-zero override (LC13:1330), so a vehicle already on its
best lane never gets urgent-strategic in that branch; (b) keepRight only accumulates in the
"right, lane not 5 km/h faster" branch; (c) all probability thresholds scale with
`1/lcSpeedGain` and the right threshold additionally with `1/lcSpeedGainRight`; (d) euler
update changes brakeGap, decelGap, maximumSafeStopSpeed and the `remainingSeconds − 1` term
in informFollower (LC13:877-879).

## 10. Resets & persistence

- `changed()` (LC13:1063-1079): zero probabilities, state, advice; keep blocker reservation
  only if `bestLaneOffset != 0`.
- `saveState/loadState` (LC13:2236-2262): persists the 3 probabilities, laSpeed, dontBrake.
- TraCI: `cancelRequest()` (ALCM:568-574) stores the un-canceled state per direction; TraCI
  `right/keepRight` requests re-enter as `LCA_TRACI` and bypass URGENCY timing
  (LC13:1504-1507).
```

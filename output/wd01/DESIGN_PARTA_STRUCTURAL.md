# DESIGN note: structural lane-change fixes for native Part A (v2 candidate)

Design text only; no ZCode dispatched. 2026-09-27, world model session.

## Motivation (measured, p3q, 17 TRAIN roots, k = 5)

- **Merges:** +57% at H=30, and too early: median 7.5 s vs 15 s into the window; 26.5 m vs 18 m from the lane end.
- **Discretionary changes:** a constant 6–10 per window, while the truth goes from 9–14 at 2000 veh/h down to 3–6 at 2700 veh/h. There is no density dependence.
- **No parameter setting fixes this.** Every merge-reducing lever raises queue TTC exposure, because the merge/queue mechanism itself is wrong.

## Fix 1: speed-gain discretionary lane change (LC2013-like)

Replace the logistic hazard for 0↔1 with SUMO LC2013's speed-gain logic, emulated per vehicle and step:

- **Keep-right motive:** a vehicle in lane 0 (left) returns to lane 1 when the gain vanishes.
- **Speed-gain motive:** an accumulator
  `sg += (v_safe(target) − v_safe(current)) / v_max · dt`
  that decays when negative. A change is triggered when `sg > threshold · (1 / lcSpeedGain)`.
  `v_safe(lane)` is the IDM safe speed behind that lane's leader.
- **Safety:** the target-lane leader and follower gaps must satisfy the IDM safe-braking bound. This is the same code as the merge follower check fixed in p3l.
- **Parameters:**
  - lcSpeedGain and lcKeepRight per vType (SUMO defaults 1.0 / 1.0, adjustable per vType);
  - the accumulator threshold;
  - a decay rate.
  All are fitted on TRAIN.
- **Expected effect:** density dependence emerges naturally. In light traffic there are many speed-gain opportunities; in congestion the lanes equalise and changes become rare.

## Fix 2: strategic late merging with speed matching

Replace the merge hazard plus end-force with an LC2013-like strategic merge.

- The urgency of a merge rises as the remaining distance to the lane end shrinks: the lookahead is ~ v · lcLookaheadLeft.
- **Before** the urgency threshold, the vehicle keeps up with its own lane (cooperative gap search) and merges only into gaps that satisfy full safety. That produces late merges near the end, as observed.
- **After** the threshold, it adapts speed to the best target gap: it decelerates to fall in behind a leader in the target lane, then merges. This speed matching is what avoids stop-and-go at the lane end.
- **Cooperation:** a main_right follower whose predicted leader is an urgent merger yields by adopting it as an IDM leader, with probability lcCooperative per vType.
- **Remove:** end_force_s, merge_accept_prob and the merge cooldown hacks.
- **Expected effect:** fewer, later merges; less stop-and-go in the queue; queue TTC exposure closer to the truth. This removes the p3q trade-off at its source.

## Implementation plan (when approved)

1. Engine kind `idm_lc2013`, with default-off flags and tests. Test scenes:
   - an empty target lane → speed-gain change;
   - an equal-speed lane → no change;
   - an urgent merger with a speed-matching trajectory;
   - a cooperative yield.
2. Fit on chunk-1-era native TRAIN roots with k-fold, as in p3q.
   - Targets: per-type LC counts, merge position and time, the accel speed profile, queue TTC, physics delay.
   - All scoring is TRAIN-internal.
3. When chunk-2 native data is available, refit or check on chunk-2 TRAIN worlds (speedMode 27). CAVs are native here, so this is mostly human behaviour.

## Validation

- **One pass** on chunk-2 VALIDATION worlds under the frozen registration (REGISTRATION_NATIVE_PARTA_CHUNK2_DRAFT.md, after the owner's (A)/(B) decision).
- **Gate:** gate v2 (a)–(d), with H=30 primary and H=60.
- **First-class metrics** alongside the gate: per-type LC deviation (merges and discretionary each within ±20%), merge position/time KS statistics, and delay/TTC CRPS vs climatology.
- **The comparison of record:** v2 (`idm_lc2013`) vs v1 (`idm_merge`, the frozen p3o candidate), both evaluated in the SAME single pass on the same worlds. This makes it one look for both, and the owner's registration names v2 as the candidate under test.

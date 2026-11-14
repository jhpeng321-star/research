# Design discussion — how to move forward (2026-10-08)

Status: discussion brief for the owner. **Options only - nothing is being built.** Training is postponed by owner decision until a direction is chosen.

## Findings (all CPU, zero simulation, TRAIN data or unseen-PI test sets as stated)

### 1. When does each controller's benefit show? (EFFH-02, runs/active/EFFH-02-20261008/RESULT.md)
- **Data:** 282 state-matched pairs from chunk-4 and the wall pilot, TRAIN only.
- **Delay at 600 s vs native (median):**
  - damping +1723 veh-s (worse);
  - PI -260 (not significant);
  - walls -1057 (better).
- **Horizon needed for the 600 s winner to agree in 80% / 90% of pairs:**
  - damping 290 / 470 s;
  - PI 300 / 410 s;
  - walls 260 / 420 s (nothing before ~140 s, while the wall assembles).
- **Short horizons (30-120 s):** only ~60-70% agreement, and <3% of the 600 s effect has appeared by 120 s.
- **Leading indicators at 30-60 s:** weak, except for walls (merge load at 60 s, AUC 0.71).
- **Effects still growing at 600 s;** nothing is recorded beyond.

### 2. Why do predictions blow up at 60-120 s? (DIAG120, E:/rd-wt/ws-diag120/_out/DIAG120.md)
- **Missed jams and systematic under-braking**, not phantom jams. Speed is too high by ~1-2 m/s upstream and ~2.6 m/s at the merge; this bias correlates 0.71 with the 120 s position error.
- **The error starts upstream (1-2 km before the merge) and propagates backward.** Each follower's error starts 1-1.5 s after its leader's.
- **Growth is gradual, then accelerates.** Median crossings: 20 m at ~58 s, 50 m at ~84 s, 100 m at ~99 s. A late jump around 108 s coincides with a recorded braking event in ~53% of cases.
- **Not the cause:** entries, per-driver params.

### 3. Is a cell transmission model a good long-horizon anchor? (CTM-alone, E:/rd-wt/ws-ctm/_out/CTM_ALONE.md)
- **Stable** (never blows up).
- **Merge delay error 4-5x larger than C1 struct.** At the per-car blow-up states it also misses the jams (55-65 missed, ~0 phantom).
- **Crude calibration:** one 75 m merge cell; the merge FD is inherited from the mainline; the capacity drop is estimated from only 36 congested bins.
- **Better than C1 only downstream:** speed error ~half of C1's at 120 s.

### 4. Can the current model rank controllers? (RANKING, E:/rd-wt/ws-ranking/_out/RANKING.md; CAVs teacher-forced, closed loop not run)
- **C1 struct, native vs PI on network delay, u_same 140 states:**

  | horizon | correct choices | regret (veh-s) | always-native | always-PI |
  |---|---|---|---|---|
  | 60 s | ~68-72% | 4-6 | 16 | 19 |
  | 120 s | ~71-74% | 8-16 | 51 | 69 |
  | **300 s** (coordinator computation from the H300 run) | **80%** | **19** | **128** | **194** |

  The coordinator's 60/120 s numbers differ slightly from the harness report (delay accumulation convention); the 300 s numbers are the coordinator's only.
- **u_new (21 states):** 81-90% correct at 60-120 s; 71% at 300 s, regret 66 vs 149-176.
- **Trajectory black box:** ranks better at 60-120 s (u_same 120 s: 81% correct, regret 2.6). It was not run at 300 s.
- **Short call vs settled outcome:** the 60/120 s predicted call agrees with the recorded 300 s winner only 61-68%.

## What this means
1. **The traffic-level delay difference is more robust than car positions.** The model picks the better of native/PI in ~80% of situations at 300 s with far lower regret than any fixed policy, even though car positions blow up at 120 s. The planner can use this.
2. **Decisions must be judged at ~5 min** (EFFH-02 and the cross-horizon result), so 300 s traffic-level accuracy is what matters most.
3. **The main model error is under-braking / missed upstream waves** (DIAG120). It caps long-horizon accuracy.
4. **The CTM as calibrated cannot fix the merge;** at most it is a downstream anchor.
5. **All ranking so far replays recorded CAV motion.** Closed loop (the model runs the controller code) is still untested, and it is the real planner setting.

## Candidate directions (owner to choose; pros / cons)

**A. Fix the braking response (the DIAG120 lever).**
- What: loss weighting on decelerating HDVs; an asymmetric penalty on "too fast" errors; a follower loss after a leader brakes (string propagation); possibly a physics decel floor matched to recorded braking.
- Pros: targets the measured dominant mechanism; cheap; keeps the per-step structure and C1.
- Cons: may over-brake in free flow; needs a retrain to confirm.

**B. Better macro anchor.**
- What: finer merge cells, a separately calibrated merge FD, a second-order model (ARZ) for waves; or use the CTM only downstream.
- Pros: stability and conservation; novel coupling (lit check: docs/LITCHECK_MICRO_MACRO_20261008.md).
- Cons: the current CTM misses the same jams; substantial calibration work before coupling can help.

**C. Planner built on traffic-level deltas at per-controller horizons.**
- What: score each candidate by its predicted network-delay difference at its own horizon (~260-470 s from EFFH-02), re-plan often, and add a small switching penalty.
- Pros: uses what already works (~80% correct at 300 s); no retrain needed to start.
- Cons: teacher-forced so far; must be validated in closed loop first.

**D. Validate closed-loop ranking first (evaluation only).**
- What: re-run the ranking with the controller code executed inside the model (the closed-loop mode on wms/switch).
- Pros: answers whether the planner can use the model today; CPU only.
- Cons: none beyond CPU time (a few hours).

A plausible sequence is D (closed-loop check) -> A (braking fix, one retrain) -> C (planner prototype), with B as a longer-term research item. The choice is the owner's.

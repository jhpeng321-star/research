# S-R14 preparation (D1 lane-change secure gap, D2 merge timing, D3 low-share data)

Branch `c4/v10r14` (from 97d22246). Build and offline inference only; **no training launched**.

## D1: `physics.lane_change_secure_gap` (default off)

Under `lane_learnable`, an HDV mainline lane change (ramp merge and CAVs exempt) must also pass the R13E gs law. It is checked on the body envelope's anticipated configuration:

- follower: `gap_f >= T_f v_f + relu(v_f²/2b_f - v_e²/2b_e)`;
- ego behind its new leader: the same law.

T and b are the learned per-slot settings, gathered by neighbour index (`secure_gap.py`). There is no s0 term and no new constant. A missing neighbour passes; invalid settings fail closed. The mask is a discrete rollout lane-logit mask with no new differentiable term. Teacher-forced lane feasibility is unchanged; lane heads are frozen in R14.

Offline runs: S-R12C plus the 13E switches ("13E"), with and without D1 ("13EL"). Common random numbers, gpuR CPU, nice 19. Evidence: `R14_OFFLINE_REPORT.json`.

| metric | SUMO | 13E | 13EL | 13EL-13E [CI] |
|---|---|---|---|---|
| 120 s, 48 TRAIN states: lane-1->0/run | 5.4 | 5.0 | 2.4 | -2.6 [-3.6,-1.8] |
| 120 s: up_near delay veh-s | 880 | 940 | 984 | +44 [21,71] |
| 300 s, 5 pairs: lane-1->0/run (native arm) | 8.0 | 14.0 | 3.8 | (all arms) -6.9 [-11,-3] |
| 300 s: up_near delay / spill m | 2140/79 | 1120/9 | 1610/53 | +496 [-140,1130] / +44 [-21,108] |
| 300 s: merges/min after 120 s | 6.6 | 2.8 | 2.3 | -0.4 [-1.4,0.5] |
| action-minus-native spillback m | -136 | -30 | -109 | r .63 vs -.15 |
| action-minus-native up_near | -1080 | -162 | -1020 | sign 4/5 vs 3/5 |

What this shows:
- D1 removes the excess queue bypass but undershoots SUMO by about half.
- At 300 s it restores most of the upstream queue and the late action payoff (D4).
- At 120 s it adds a little delay.
- The 13E late-merge deficit remains.

The 300 s evidence is 5 pairs and 1 seed, so it is indicative only.

## D2: early merges (diagnosed, not fixed)

Tools: `d2_merge_timing.py`, `d2_pivot_hazard.py`.

**Where.** SUMO merges spread along the accel lane: median 34 m before the end, 33% from standstill. 13E merges at the end: 8-11 m, 62-68% in the last 15 m, 67-82% from standstill.

**When.** In-zone time is similar (21 vs 26 s SUMO/WM). The earliness is in the approach: first seen on the ramp to merge takes 51 vs 40 s. SUMO's ramp queue spills upstream of the zone; the WM's compacts at the lane end.

**Acceptance.** Given the pivot margin, WM upstream hazards match SUMO within noise. Upstream gaps are 3x rarer in the WM (1.3% vs 4.3% of rows), because ramp cars enter faster (2.7-3.4 vs 1.0 m/s).

**Conclusion.** No missing urgency or distance term in acceptance. The missing piece is LC2013-like longitudinal speed adaptation of ramp cars; that is the next-round candidate.

## D3: low-CAV-share data

**Sources.** Three lowpen packs are added as `data.c4_sources`. `world_roles` keeps their validation worlds on the validation side. Shares split each v5 share by TRAIN windows:

| source | v5 | lowpen |
|---|---|---|
| native | .4694 | .0306 |
| plain | .2382 | .0118 |
| branch | .0824 | .0176 |

pairs300 keeps .15.

**Weight.** `data.replay.low_share_oversample = 1.69` applies at stages of 120 s or more, to windows whose first-row CAV share is <= 0.15 (midway between the 10% and 20% levels). Derivation:
- v5 TRAIN is 32% low share; base-weighted low mass is 0.3723.
- 1.69 = 0.6277 / 0.3723, which gives equal draw mass to both halves of the evaluation grid.
- The training receipt logs the realized draw fraction.

**Loader check** (cpuN, `loader_check.json`):
- TRAIN windows come only from TRAIN-role worlds.
- native: 1975 / 1525 windows at 120 / 300 s.
- 2.7% of windows have realized share above 0.15.

**Staging: not on gpuR.**
- Moved 8 diagnosis dirs (1.5 GB, sha256 verified) to cpuN `~/c4/archive_gpuR_diag/`.
- Kept diag_brake: it is a live 13E gate dependency.
- Free space is now 3.6 GB. The 2.4 GB rsync would leave 1.2 GB, so it was skipped.

## Configs and gate

`c4_wm_v10round14{,_stage_b}.toml` are the 13E A / B recipes plus D1 and D3; `wms.train.round14_gate` asserts exactly that difference.

- A starts from the 13E B FINAL, which must have both 13E B checks CONTINUE.
- The stop reference is the 13E B FINAL with the same switches.
- **Lesson applied:** B launches after A's verdict and A **midrun** CONTINUE. A's postrun runs in parallel, and B's midrun stops B unless that postrun says CONTINUE.
- The script adds GPU and disk guards, `--chain` and `--registration-template`.

## Verification

- Bit identity vs 65cc714e: 6 cases, 1222 tensors.
- New tests: FD gradients and threshold cases.
- R13E/R14 suites: 269 passed.
- Full suite: no new failures.

## Risks

- D1 undershoot.
- n=5 at 300 s, possibly bistable.
- Late-merge starvation.
- Unchanged TF mask.
- Packs need about 5.4 GB free.
- Gate not run end to end.

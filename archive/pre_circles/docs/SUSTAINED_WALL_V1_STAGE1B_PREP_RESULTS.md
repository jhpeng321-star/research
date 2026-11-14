# Stage 1b-prep results — hard-brake attribution, band locator, sustained-hold arms

Date: 2026-09-02. Status: **development-only preparation; no gate is applied
and no cell or treatment is chosen or endorsed** (see "claims not made").

## Part A — artifact-only analysis of Stage 1a (no SUMO)

Source: `runs/probe/sustained_wall_v1_stage1a_m2200_p20_s6001_6030/` only.

### A1. Hard-brake attribution of Delta_W

Granularity contract (stored data determines what is possible): the
three-phase split of hard-brake **steps** is exact (per-step network totals
from the stored series); the **location** split of all hard-brake steps is at
road granularity (per-step lane positions were not persisted, so the 200-400 m
band split is only available on **event starts**); the full location x phase x
leader-member split is event-start level (≤200 events per branch,
truncation flag recorded; no branch was truncated except 6006-W).

Per seed, C's absolute hard-brake steps and Delta_W = W − C (all positive; ratio
W/C 1.01-2.02; median seed 6011 and worst seed 6006):

| seed | C steps | ΔW steps | ratio | excess by phase (during / release-60 / after-60) | excess by road (approach / merge / post-merge) |
| --- | ---: | ---: | ---: | --- | --- |
| 6011 (median ΔW = +35) | 133 | +35 | 1.26 | +19 / +14 / +2 | +23 / −8 / +20 |
| 6006 (worst ΔW = +154) | 169 | +154 | 1.91 | +67 / +50 / +37 | +115 / −71 / +110 |

Median across seeds (from Part A per-seed rows): ΔW median +35 (p10 +8.8,
p90 +62.6); phase excess concentrates **during the wall** (median +30.0,
p10 +10.8, p90 +54.8) followed by the release+60 window (median +1.5,
p10 −5.0, p90 +11.3), and is near zero after release+60 (median 0.0,
p10 −2.0, p90 +9.1). Location excess is concentrated on the **mainline
approach edge** (median +33.0; +3…+115 across seeds) while the **merge edge
often decreases** (median −3.5; −71…+41) — i.e., the wall replaces some
merge-edge braking with more approach-edge braking. Post-merge is near zero
except on the worst seed (+110); ramp contribution is zero on every seed.

Leader-member attribution (all-step exposure counts; event-start level in
brackets): Delta of steps whose braking follower had a **wall member leader**
median +5 (6006: +18; 6011: +5); event starts with a member leader: 4 (6011),
9 (6006) vs 0 in C by construction.

Upstream-of-band braking speeds (event starts with road = approach edge and
distance > 400 m): median 29.8-30.1 m/s, p10 26.7-27.6, p90 32.5-32.7, and
**0 of 47 events below the 10 m/s wall speed** — the upstream hard-brake
events are free-flow-speed slowdown events, not wall-queue events
(recoverable from the stored event records; reported at event-start
granularity).

### A2. Low-TTC direction for the 15 seeds outside the P band

The P (one-step nudge) band for low-TTC steps is [0, 0] (28 seeds exactly 0,
6006 +2, 6007 −? — p5 = p50 = p95 = 0). Of the 15 seeds whose Delta_W lies
outside that band: **5 are increases** (6001 +1, 6003 +1, 6006 +6, 6019 +1,
6023 +1) and **10 are decreases** (6007, 6008, 6011, 6012, 6013, 6015, 6018,
6021, 6025, 6028; −1…−4). The location delta is **entirely on the merge edge**
(road 1487606179) for every one of the 15 seeds — no approach-edge or
post-merge low-TTC change.

### A3. Onset vs decision (C arm)

Among the 30 seeds (6 no-onset in C: 6003, 6005, 6009, 6016, 6025, 6029):

| bucket | count | seeds |
| --- | ---: | --- |
| onset < t_d (post-onset decision) | 8 | 6004, 6007, 6013, 6015, 6018, 6019, 6022, 6028 |
| t_d <= onset < t_d + 60 | 4 | 6001, 6002, 6012, 6023 |
| onset >= t_d + 60 | 12 | 6006, 6008, 6010, 6011, 6014, 6017, 6020, 6021, 6024, 6026, 6027, 6030 |

h(t) on the C prefix: h(100) p10/p50/p90 = 0.097/0.683/1.0; h(200)
p10/p50/p90 = **0.0/0.0/0.713**, with 20/30 seeds at h(200) <= 0.45 —
i.e., by t = 200 the merge health is already at or below 0.45 on most seeds.

### A4. Seeds 6012 (prevented) and 6025 (induced)

Full listings (h at 10 s for C and W, wall member count per 10 s, and the
hard-brake / low-TTC event times) are in
`runs/probe/sustained_wall_v1_stage1b_prep/part_a.json → a4_case_trajectories`.
Highlights:

- **6012** (t_d = 200, W prevented a C collapse at onset 252; W had no
  collapse run in the whole horizon): the C h series is near 0 during
  250-320 s while the W h series recovers to 0.25-0.94 in the same window;
  the W members (max 3, mean ~1.2) are present only during 210-330 s.
  W still added 11 hard-brake steps versus C (all during the wall window,
  239-288 s) and one fewer low-TTC event (C 1, W 0).
- **6025** (t_d = 200, W induced a collapse at onset 379 vs no C collapse):
  W members peaked at 7 during 260-330 s; the W h series then stays at 0
  for the 340-500 s range (vs C's oscillating 0.14-0.78), consistent with
  the induced collapse. W added 45 hard-brake event starts (mostly
  250-400 s, on the approach edge, several with a wall-member leader) and
  has fewer low-TTC events (C 1, W 0).

## Part B — band locator (native IDM Arm C only)

Cells: 1900/2000/2100 veh/h mainline, 400 ramp, p = 0.20, seeds 6031-6060 /
6061-6090 / 6091-6120 (registered in `idm/contract.py` as
`sustained_wall_v1_stage1b_band_locator` and added to the prospective
exclusion set; grep audit: no prior assignment). 2200 reuses the Stage 1a
Arm C; not rerun. All 90 branches ended by verified natural clearance.

| cell | health incidence (CP95) | physical incidence | onset median (min-max) | fraction h(300) > 0.45 | delay s/veh median |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1900 | 5/30 (0.056-0.347) | 0/30 | 649 (454-761) | 0.367 | 9.91 |
| 2000 | 9/30 (0.147-0.494) | 0/30 | 383 (241-636) | 0.500 | 11.37 |
| 2100 | 17/30 (0.374-0.745) | 0/30 | 303 (196-581) | 0.333 | 12.82 |
| 2200 (reused) | 24/30 (0.614-0.923) | 0/30 | 258.5 (119-879) | 0.167 | 12.43 |

**Cell identification paragraph.** By point estimates, exactly two cells
satisfy both "health incidence in [0.3, 0.7]" and "median onset >= 300 s":
**m2000** (0.30, median onset 383 s) and **m2100** (0.567, median onset
303 s). m1900 (0.167, onset 649) fails the incidence criterion; m2200 (0.800,
onset 258.5) fails both. The 95% Clopper-Pearson intervals add a caveat:
m2000's interval [0.147, 0.494] straddles the 0.3 boundary, and m2100's
[0.374, 0.745] straddles the 0.7 boundary — so with interval-level strictness
no single cell is unambiguously INSIDE [0.3, 0.7]; m2100 is the only cell
whose interval lies mostly within the band, while m2000 is borderline. This
is a description, not a selection: no cell is chosen here. The physical label
remains 0/30 in every cell, and the delay median is essentially flat
(9.9-12.8 s/veh; 2200 is NOT the worst delay cell at 12.43 vs 2100's 12.82).

## Part C — sustained-hold arms at 2200/p20 (t_d = 200 rule-blind)

Arms: WHOLD_soft (200-400 m, 10 m/s, tau 3.0, 0.3 clip; Stage 1a Arm W
semantics held to t = 900 s) and WHOLD_strong (RUNGS[2] from
`env/wall_supervisor.py`: 200-800 m, 12 m/s, tau 3.0; held to 900 s), both
with the 5 s taper and tau restore at 900 s. 60 new branches plus 30 Arm C
replays (used for prefix identity; every replayed C trace is digest-identical
to the stored Stage 1a C trace: 30/30). Integrity: prefix identity through
t_d = 200 and first divergence at exactly step 201 on 60/60 hold branches;
zero collisions/teleports in all 90 branches; tau restored on 60/60; terminal
classification natural clearance in all branches.

### Effects versus Arm C (n = 30; the Stage 1a 180 s pulse on the same axis)

| metric | 180 s pulse (Stage 1a) | WHOLD_soft (to 900 s) | WHOLD_strong (to 900 s) |
| --- | ---: | ---: | ---: |
| health incidence C → arm | 24 → 24 | 24 → **12** | 24 → **18** |
| paired incidence reduction (bootstrap CI90 / CI95 of the mean) | 0.0 [−0.067, +0.067] / [−0.100, +0.100] | **0.40** [0.233, 0.567] / [0.200, 0.600] | **0.20** [0.033, 0.367] / [0.000, 0.400] |
| prevented / induced seeds | 1 / 1 | 13 / 1 | 9 / 3 |
| total duration delta, median (CI90 of median) | −30 s [−54, −18] | **−180 s** [−203, −158] | **−96.5 s** [−148, −80] |
| longest-run delta, median (CI90) | 0 s [−5.5, 0] | −42 s [−72.5, −19.5] | −24 s [−75, −1.5] |
| onset delay among co-collapsing seeds (n) | — | 0 s (n=11; −341…+401) | 26 s (n=15; −111…+557) |
| delay s/veh median (p10 / p90; CI90 of median) | +0.655 (−0.329 / +1.952; [0.203, 0.797]) | **+3.09** (−0.41 / +4.63; [2.43, 3.58]) | **+13.98** (+9.50 / +15.61; [11.89, 14.43]) |
| fraction of seeds with delay benefit | 23.3% | 13.3% | 0.0% |
| hard-brake excess, median (p10 / p90) | +35 (+9 / +63) | **+166.5** (+81 / +230) | **+388.5** (+236 / +552) |
| low-TTC excess, median (p10 / p90) | 0 (−1 / +1) | 0 (−2 / +1) | 0 (−3 / +1) |

Hard-brake excess phase split (median; A1 granularity): WHOLD_soft **+164.5
during the wall**, +1.5 in [release, release+60], 0.0 after release+60;
WHOLD_strong +307 during, +38 in the release window, +30 after. Location
split (median road marginals): WHOLD_soft +190.5 approach / −21.5 merge /
0.0 post-merge; WHOLD_strong +352.5 approach / −14.5 merge / +42.0
post-merge. So both hold arms keep the same signature as the pulse — the
hard-brake excess is almost entirely while the wall is active and on the
mainline approach edge, with merge-edge braking decreasing.

Dose: both arms held the full 700 s (201-900) and released at the 900 s cap
(time_cap on 30/30 per arm). WHOLD_soft member-seconds median 1205.5, mean
members/step 2.01, per-seed-mean per-lane coverage 0.589, fraction of the
[200, 900] window with a member in every occupied lane 0.359. WHOLD_strong
member-seconds 1335.0, per-lane coverage 0.675, full-coverage fraction
0.449.

Reading across the one axis: extending the wall from 180 s to 700 s converts
a zero incidence effect into a substantial one (24 → 12) with a modest delay
cost (+3.1 s/veh) and a much larger hard-brake excess (+166 vs +35); the
stronger rung costs 4.5x more delay, prevents fewer collapses, and induces
three collapses that Arm C did not have, at a further multiple of hard-brake
excess. Noteworthy seed-level detail: seed 6012's collapse was prevented by
the 180 s pulse but NOT by the soft hold (C decayed there anyway under the
longer wall), so "longer is better" is not uniform across seeds. No gate is
applied; these are reported effect sizes with intervals.

## Artifacts

- `runs/probe/sustained_wall_v1_stage1b_prep/part_a.json` — Part A analyses
  (per-seed rows, median/worst seeds, low-TTC outside-band list, onset
  buckets, A4 listings). SHA-256 `e2d787164369fb0b775263e8d2dc3b9adc5f87ff8c573ac347f6f2700e029b62`.
- `runs/probe/sustained_wall_v1_stage1b_prep/band_locator_summary.json` —
  Part B per-cell statistics (includes `m2200_stage1a_reused`). SHA-256
  `1238648bd712d492b1fb26ea44b2e7e861b8f22645b7320813b85fa4756f4840`.
- `runs/probe/sustained_wall_v1_stage1b_prep/hold_summary.json` — Part C
  per-arm paired statistics + integrity summary + Stage 1a pulse reference.
  SHA-256 `52daa542d3d7e868cccd895fb0671311f5c432b11974cfbb9d30895d7e33302c`.
- `runs/probe/sustained_wall_v1_stage1a_m2200_p20_s6001_6030/summary.json` —
  Stage 1a summary (unchanged, pulse reference). SHA-256
  `bef7a1a61e225d8f61fb38a727c3fb2ef0edde83e35b8a6ea3fd7e749302f4cf`.
- Band locator per-seed artifacts:
  `runs/probe/sustained_wall_v1_stage1b_band_locator_m1900_s6031_6060/`,
  `..._m2000_s6061_6090/`, `..._m2100_s6091_6120/` (manifest + seeds each).
- Hold arms: `runs/probe/sustained_wall_v1_stage1b_hold_m2200_p20_s6001_6030/`
  (manifest + seeds, three branches per seed + integrity).
- Code: `experiments/idm/stage1b_common.py`, `stage1b_runner_core.py`,
  `analyze_sustained_wall_v1_stage1b_part_a.py`,
  `run_sustained_wall_v1_stage1b_band_locator.py`,
  `analyze_sustained_wall_v1_stage1b_band_locator.py`,
  `run_sustained_wall_v1_stage1b_hold_arms.py`,
  `analyze_sustained_wall_v1_stage1b_hold.py`, plus
  `tests/test_stage1b_common.py` (5 tests; full suite **1005 passed** before
  compute and after the final code, 1000 Stage-1a tests unchanged).

## Claims not made

- No predictor/model was fitted or screened; no cell was chosen; no gate was
  applied and no treatment is endorsed. The m2000/m2100 identification in
  Part B is a description of where the criteria hold, not a selection.
- No validation or prospective claim: all blocks are development seeds
  registered in the prospective evaluation exclusion set.
- Part A's location x phase x leader-member split is event-start granularity
  (per-step lane positions were not persisted); the phase and road-marginal
  splits are exact step counts.
- Part C paired comparisons use the replayed Arm C with the rule-blind
  t_d = 200 window; the Stage 1a pulse column uses the Stage 1a landmark
  windows (reported alongside, same cell and C arm).
- No generalization outside 1900-2200 veh/h, 400 veh/h ramp, p = 0.20, IDM.

# DIAG 1 — where does the 120 s world-model error come from?

Analysis only (CPU, no training/SUMO/GPU). Set: `wall_same` h60, 67 states (1 recorded arm ends
before t0+120 s -> 66 states at the 120 s horizon). Models: C1 (`data_cache/ckpt_c1fast/wm_s.pt`)
and R4 (`runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt`, = R4-last). Entries OFF,
teacher-forced CAVs, seed 0 — identical to the R4 scoring harness.

**Verification:** per-state HDV `ds_med` at 60/120 s recomputed here matches the R4 score cache
bit-exactly (266/266 checks, max |diff| = 0.0; 2 cache-missing values for the truncated state).
State-median metric reproduced: C1 8.71 / 99.84 m, R4 9.96 / 110.69 m at 60 / 120 s. 0 blow-ups
(rule >50 m @60 s AND >100 m @120 s) for both models, 0/66.

Definitions (recorded data unless stated): *entrant* = vehicle id absent from the t0 modelled set
(lane>=0 at t0) appearing later in the recorded arm; *affected* = t0 HDV whose same-lane recorded
leader OR follower at any tick <= h is an entrant (junction lane -1 rows excluded from adjacency);
*matched* = t0 HDV present in both predicted and recorded frames at t0+h (HDV = recorded is_cav
False); *zones* from t0 position: on_ramp = lane 2, merge_zone = lane 0/1 with s in [0, 75.44]
(SECTIONS bounds, `src/wms/eval/metrics.py`), upstream_mainline s<0, downstream s>75.44.
Attribution uses POOLED per-HDV values; the headline metric is the median over states of
per-state medians (both given — they differ).

## 1. Entrants: the recorded truth contains none (a)

- Across all 67 wall_same states, the recorded wall arms contain **0 new vehicle ids after t0**
  (state 1: 95 ids at t0, 95 over the whole 120 s). The only entrant-like vehicles are
  **junction-transit-at-t0** cars (on lane -1 at t0, invisible to the model): mean 0.78/state
  (median 1, p90 2); all of them land on mainline lanes 0/1.
- wall_pilot: also **0 new ids** across all 53 states (id-set scan only, no rollouts).
- The "70-110 entrants per state by 120 s" in WM_CHANGE_LOG #14 counts entry-process injections
  when entries are switched ON at scoring — not vehicles in the recorded truth.
- t0 HDVs entrant-affected (leader/follower ever an entrant): mean 2.01/state by 60 s and by 120 s
  (median 1, p90 5.4).

**Consequence:** missing-entry interactions CANNOT be a direct source of the 120 s error on this
benchmark — the ground truth itself was generated without entries. Affected HDVs do go badly wrong
(C1 median |ds| 118 m, R4 82 m at 60 s; E-group time profile reaches 475 m median at 70 s), but
they are ~10 pooled vehicles at 60 s (0.4% share; +0.8/+0.6 m of the pooled mean |ds|) and all
exit before 120 s (E-group n=0 at 120 s).

## 2. Attrition and exits (e, consistency)

Per state (mean over 67): 73.7 t0 HDVs; **32.4 exit before 60 s; 60.6 exit before 120 s**
(median 63); matched HDVs 41.2/state at 60 s, 13.2/state at 120 s; survivors-at-120 13.0/state.
Presence sequences of the predictions follow the recorded exits exactly: **0 presence mismatches**
for both models over all states at every 10 s tick -> exits are excluded identically and
consistently in C1 and R4. Blow-ups: 0/66 both; the 120 s metric is not blow-up-driven. Top C1
states at 120 s: 307.8 / 284.1 / 274.6 / 263.3 / 263.3 m medians (R4 on the same states: 39.9 /
18.0 / 256.2 / 142.3 / 142.3) — state-level heterogeneity, not one mode.

## 3. Attribution of pooled mean |ds| (both models; contributions sum to the pooled mean)

Pooled per-HDV mean |ds| at 60 s: C1 44.50 m, R4 48.50 m (pooled median 10.13 / 9.88).
At 120 s: C1 175.36 m, R4 176.06 m (pooled median 112.81 / 93.50). n = 2763 (60 s) / 873 (120 s).

| group | model | n | share | med \|ds\| | mean \|ds\| | contrib to pooled mean |
|---|---|---|---|---|---|---|
| 60 s E_affected | C1 / R4 | 10 / 10 | 0.4% | 118.3 / 81.6 | 219.8 / 173.2 | 0.80 / 0.63 m |
| 60 s N_wrong_lane | C1 / R4 | 377 / 239 | 13.6% / 8.7% | 34.3 / 50.4 | 106.5 / 153.5 | 14.53 / 13.28 m |
| 60 s N_right_lane | C1 / R4 | 2376 / 2514 | 86.0% / 91.0% | 7.78 / 8.67 | 33.9 / 38.0 | 29.17 / 34.60 m |
| 120 s E_affected | both | 0 | 0% | - | - | 0.00 m |
| 120 s N_wrong_lane | C1 / R4 | 303 / 172 | 34.7% / 19.7% | 137.5 / 188.7 | 192.5 / 273.7 | 66.80 / 53.93 m |
| 120 s N_right_lane | C1 / R4 | 570 / 701 | 65.3% / 80.3% | 106.98 / 79.88 | 166.3 / 152.1 | **108.56 / 122.13 m** |

So at 120 s: **right-lane, non-entrant HDVs carry 62% (C1) / 69% (R4) of the pooled mean |ds|**,
wrong-lane HDVs 38% / 31%. At 60 s the split is ~2/3 right-lane, ~1/3 wrong-lane, ~2% entrants.

## 4. What is wrong inside the dominant right-lane group (c)

The 120 s survivors are deep-upstream mainline HDVs (t0 s median -1983 m C1 / -2098 m R4; 99.5%
upstream_mainline; free-flowing at t0, median v 26.5 m/s). They run into the back of the jam
within the horizon:

| recorded speed at 120 s | model | n | med \|ds\| | signed ds p10/p50/p90 (m) | mean dv (m/s) |
|---|---|---|---|---|---|
| <2 m/s (stopped) | C1 | 101 | 112.4 | -31.8 / +68.4 / +504.9 | +3.11 |
| <2 m/s (stopped) | R4 | 106 | 72.1 | -83.4 / -31.6 / +344.5 | +2.52 |
| 2-10 m/s | C1 | 156 | 73.1 | -85.3 / +0.5 / +408.7 | +0.73 |
| 2-10 m/s | R4 | 171 | 59.5 | -101.8 / -44.0 / +353.3 | +0.26 |
| >10 m/s (moving) | C1 | 313 | 120.9 | -301.1 / -79.3 / +117.3 | **-11.96** |
| >10 m/s (moving) | R4 | 424 | 119.3 | -359.9 / -39.2 / +154.1 | **-7.24** |

Recorded speed at 120 s: median 15.7 (C1 survivors) / 17.0 (R4) m/s; predicted: **5.1 / 7.7 m/s**.
Moving survivors are predicted ~80 / ~40 m behind their recorded position (model slows them too
early — the jam front is predicted too far upstream / too strong); already-stopped ones are left
ahead (+68 m C1). Signed dispersion (p10..p90 spans ~600 m) means it is a **wave-phase/timing
error**, not a uniform bias: both over- and under-prediction of arrival of the queue front.

## 5. Zones (d)

At 60 s (pooled): on_ramp t0 HDVs n=40, median |ds| 219.0 (C1) / 211.6 (R4) m — the worst
per-vehicle group; merge_zone n=6, median 326.5 / 200.0 m; upstream_mainline n=2717, median
9.6 / 9.6 m. Contributions to the pooled mean: on_ramp 5.6 / 4.7 m (13% / 10%), merge_zone
0.7 / 0.5 m, upstream 38.3 / 43.3 m. At 120 s the matched set is 869/873 upstream_mainline
(median 113.0 / 92.7 m) — zone attribution is degenerate at 120 s because everyone else exited.

## 6. Time profile of pooled median |ds| per 10 s (E120 = affected within 120 s; N120 = rest)

| t (s) | C1 E120 | C1 N120 | C1 all | R4 E120 | R4 N120 | R4 all | wrong-lane C1 / R4 |
|---|---|---|---|---|---|---|---|
| 10 | 9.66 | 1.98 | 2.03 | 10.34 | 2.03 | 2.11 | 2.3% / 2.1% |
| 30 | 44.13 | 4.44 | 4.54 | 30.75 | 4.47 | 4.70 | 5.6% / 4.3% |
| 60 | 118.32 | 10.02 | 10.13 | 81.58 | 9.86 | 9.88 | 13.6% / 8.7% |
| 70 | 475.38 | 15.31 | 15.35 | 328.54 | 15.52 | 15.54 | 19.2% / 11.0% |
| 80 | 198.51 | 24.07 | 24.15 | 193.51 | 26.01 | 26.01 | 25.5% / 13.1% |
| 90 | - | 30.96 | 30.96 | - | 35.90 | 35.90 | 32.5% / 15.9% |
| 100 | - | 46.66 | 46.66 | - | 44.99 | 44.99 | 37.7% / 20.7% |
| 110 | - | 72.75 | 72.75 | - | 70.46 | 70.46 | 42.7% / 19.9% |
| 120 | - | 112.81 | 112.81 | - | 93.50 | 93.50 | 34.7% / 19.7% |

(- : E-group empty after 80 s, all affected HDVs exited.) The non-affected median error grows
smoothly ~10 m @60 s -> ~113 m @120 s (C1); R4 is level with C1 until 80 s and slightly better
at 120 s. This growth is the dominant signal; it is NOT a jump tied to lane flips (wrong-lane
share peaks at 42.7% C1 @110 s while the median error keeps growing smoothly).

## 7. Paired R4 - C1 gap (pooled per-HDV, common matched ids)

60 s: overall mean gap +4.01 m (R4 worse); E-group -0.17 m, N-group +4.18 m (of which the
right-lane subgroup carries R4's higher 60 s error: 34.60 vs 29.17 m contribution).
120 s: overall +0.70 m (E n=0, N +0.70) -> per-HDV, R4 and C1 are statistically indistinguishable
at 120 s; the ROUND4_SCORE +6.6 m state-level paired number is state-median based (CI includes 0).

## 8. Reading (measured vs inference)

Measured:
1. The 120 s error is NOT entries: the recorded truth contains no post-t0 entrants (0 new ids in
   all 67 wall_same + 53 wall_pilot states); affected HDVs are ~2/state, wrong by ~100+ m, but
   gone by 90 s.
2. It is NOT blow-ups (0/66) and NOT exit inconsistency (0 presence mismatches).
3. 62-69% of the pooled mean |ds| at 120 s comes from right-lane, entrant-free HDVs; they are the
   deep-upstream survivors of attrition (13/state), and their error is a stop-and-go wave-phase
   decorrelation: predicted speed at 120 s 5.1/7.7 vs recorded 15.7/17.0 m/s (median), moving
   cars stuck ~40-80 m behind, stopped cars left ahead, signed error spanning ~600 m.
4. Wrong-lane HDVs contribute 38%/31% of the pooled mean at 120 s (share 34.7%/19.7%) — a strong
   amplifier but a minority; R4 halved the share yet its wrong-lane per-vehicle error grew
   (273.7 vs 192.5 m), netting out.
5. At 60 s the on-ramp/merge t0 zone is the worst per-vehicle region (median ~212-219 m over 40
   pooled HDVs) but only ~10-13% of the pooled mean.
6. Paired R4-C1 pooled: +4.0 m @60 s, +0.7 m @120 s — the two models fail in the same way.

Inference (not directly measured):
- The dominant failure is compounding longitudinal dynamics error in congestion — the queue's
  upstream front is misplaced in space-time. The R4 saturating gate removed the late-correction
  explosion (|g*da| 18.0 -> 2.5) without moving this error, which is consistent with the error
  being accumulated small per-step bias/variance in wave propagation, not late-horizon correction
  magnitude.
- Round 5: training with entries will NOT improve this benchmark's 120 s score (the truth has no
  entrants; entries-ON scoring already hurt, WM_CHANGE_LOG #14). Entries only become measurable
  after regenerating evaluation/training data WITH recorded entries — an owner-level data
  decision. The direct 120 s levers are jam-wave fidelity (longer-horizon queue curriculum,
  wave-phase/shock-speed-aware losses or per-driver wave calibration) and, secondarily, lane
  timing (still ~1/3 of the mean). Metric caveat: the reported 120 s number is a median over
  states of ~13 survivors each — a small, selected (deep-upstream) population.

Artifacts: `results.json` (all tables + per-state rows), `scripts/` (rollout_dump.py,
aggregate.py, supplement.py, print_tables.py), `out/rollouts.pkl` (74 MB per-HDV arrays:
C1/R4 x 67 states, predicted + recorded). Ledger/WM_CHANGE_LOG close-out left to the coordinator
(this task was write-restricted to this directory).

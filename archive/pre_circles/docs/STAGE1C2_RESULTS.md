# Stage 1c-2 results — fixed-hold and onset-triggered arms, fair baseline, lead time

Date: 2026-09-05. Status: **complete on all four parts (A-D)**. Seeds are the
Stage 1c development seeds (7401-7520, p = 0.20, plateau-decline profile);
D references are the stored Stage 1c D branches, digest re-verified by the
runner. No go/no-go criterion is attached; p30 remains deferred by order.

## Part A — recording fix and re-record

**Cause.** The generation capture read
`getattr(env, "active_cavs", set())`, but the environment only exposes
`env._active_cavs`; the default silently returned, so `cav_count_band` and
`cav_count_upstream` were constant 0. `ramp_queue_length` used a 15 m/s
threshold that never fired at decision times.

**Fix.** `experiments/idm/rsu_feature_capture.py::step_features_fixed` reads
`env._active_cavs` (with `_prev_obs_ids` fallback) and redefines
ramp queue length as the count of ramp-edge vehicles below **5 m/s** (the
30 s ramp arrival count already existed as `ramp_30s_entries`).

**Variance test.** `tests/test_rsu_feature_capture_variance.py` replays the
frozen damping controller for three development seeds (4101-4103, 255 steps)
and asserts the repaired CAV counts vary and no feature is constant except a
documented allowlist: `lane_*_nearest_to_band_top` (censored to 0 whenever
the band holds a CAV — degenerate by construction), and
`ramp_30s_entries` / `upstream_inflow_30s` / `ramp_count` (seed-invariant
because the route template inserts vehicles with deterministic
`vehsPerHour` headways), and `ramp_queue_length` (see below).

**Re-record.** 240/240 p20 D branches replayed with digest identity
(**0 mismatches** — the capture is metadata and cannot move the digest);
corrected vectors at t = 150/250/350 in
`runs/probe/stage1b_features_v2_p20_s7001_7240/`. The repaired counts carry
real variance (cav_count_band mean ≈ 1.1, pvar ≈ 0.8-1.1; cav_count_upstream
mean ≈ 3.7-3.9). `ramp_queue_length` is 0 in 240/240 at all three t_d —
a corridor property (the ramp never drops below 5 m/s at decision times,
even under merge gridlock), not a capture defect.

**Latent launcher bug found and fixed** (`deterministic_sumo.py`): the
interprocess lock path resolved to `experiments/runs/probe/` (parents[1]),
not the repo root; it only worked while a stray directory existed. Now
`parents[2]` + directory creation. No determinism impact (lock mechanics
unchanged).

## Part B — hold-duration and onset-triggered arms

Arms per seed (cells 2400/2600/2700, n = 30 per cell; 2200 = WON_H300 only):
W250_H300, W250_H450 (fixed holds from t_d = 250); WON_H300, WON_H450
(engagement = treatment-blind onset: first step in the D prefix with
h < 0.15 for 10 consecutive s, capped at t = 900); W250_R1b (corrected
state release: speed gate over 400-800 m ONLY, hard max 900 s of hold
relative to engagement). References (not rerun): 1c W_HOLD (hold 650 from
251) and W_ORACLE (hold 950). Integrity: 360/360 new wall branches
prefix-identical through engagement (first divergence at engagement+1 or
later), 9/9 D repeats exact, 0 digest mismatches vs the stored 1c D.

Onsets are early and universal: median step 141-148 per cell, **0 seeds
without onset by t = 900**.

**Delta incremental delay vs D** (total seconds per branch, mean
[bootstrap CI90], median; s/veh in the doc tables; completions = vehicles
exiting the merge edge within the branch; "dComp" = median completions
delta):

| cell | arm | Δdelay total s | s/veh | frac benef | dComp | Δcollapse dur |
|---|---|---|---|---|---|---|
| 2400 | W250_H300 | **−781** [−1918, +334] | −0.66 | 0.50 | 0 | −117 s |
| 2400 | W250_H450 | −384 [−1447, +657] | −0.33 | 0.47 | 0 | −163 s |
| 2400 | WON_H300 | −715 [−2061, +598] | −0.61 | 0.43 | 0 | −134 s |
| 2400 | WON_H450 | +675 [−878, +2288] | +0.57 | 0.43 | 0 | −162 s |
| 2400 | W250_R1b | +1073 [+27, +2092] | +0.91 | 0.37 | −8 | −238 s |
| 2400 | W_HOLD | +204 [−837, +1215] | +0.17 | 0.43 | 0 | −229 s |
| 2400 | W_ORACLE | +1073 [+27, +2092] | +0.91 | 0.37 | −4 | −241 s |
| 2600 | W250_H300 | −3862 [−5421, −2270] | −3.09 | 0.83 | 0 | −44 s |
| 2600 | W250_H450 | −3849 [−5336, −2313] | −3.08 | 0.83 | 0 | −68 s |
| 2600 | WON_H300 | −3893 [−6592, −1229] | −3.12 | 0.67 | 0 | −116 s |
| 2600 | WON_H450 | −3366 [−5599, −1085] | −2.69 | 0.73 | 0 | −127 s |
| 2600 | W250_R1b | −2781 [−4245, −1210] | −2.23 | 0.77 | −14 | −100 s |
| 2600 | W_HOLD | **−4042** [−5447, −2538] | −3.24 | 0.80 | 0 | −96 s |
| 2600 | W_ORACLE | −2781 [−4245, −1210] | −2.23 | 0.77 | −14 | −107 s |
| 2700 | W250_H300 | −5556 [−7745, −3332] | −4.32 | 0.77 | 0 | −71 s |
| 2700 | W250_H450 | **−6161** [−8172, −4136] | −4.79 | 0.80 | 0 | −91 s |
| 2700 | WON_H300 | −2332 [−4779, +152] | −1.81 | 0.63 | 0 | −81 s |
| 2700 | WON_H450 | −1874 [−3962, +123] | −1.46 | 0.50 | 0 | −95 s |
| 2700 | W250_R1b | −4498 [−6426, −2630] | −3.50 | 0.70 | −14 | −103 s |
| 2700 | W_HOLD | −5793 [−7681, −3936] | −4.51 | 0.77 | 0 | −99 s |
| 2700 | W_ORACLE | −4498 [−6426, −2630] | −3.50 | 0.70 | −14 | −101 s |
| 2200 | WON_H300 | +1105 [+809, +1432] | +1.00 | 0.03 | 0 | −72 s |

**Hold-duration curves (mean Δdelay total s at t_d = 250):**

- 2400: monotone — 300: −781 < 450: −384 < 650: +204 < 950: +1073 (shorter
  is better; only the 300 s arm's mean is negative).
- 2600: interior optimum at the reference hold — 650: −4042 < 300: −3862 ≈
  450: −3849 < 950: −2781 (the 950 s oracle over-holds: +1200 s of delay
  vs W_HOLD and −14 completions).
- 2700: interior optimum at 450 — 450: −6161 < 650: −5793 < 300: −5556 <
  950: −4498.

Onset-triggered curves (from onset, median step ≈ 145): 2400 −715 (300) /
+675 (450); 2600 −3893 / −3366; 2700 −2332 / −1874.

**Best arm per cell:** 2400 → W250_H300 (−781); 2600 → W_HOLD (−4042);
2700 → W250_H450 (−6161). **WON vs best fixed:** WON_H300 is within CI90 of
the best fixed arm at 2400 (+67 [−947, +1117]) and 2600 (+149 [−2123,
+2357]) but significantly worse at 2700 (+3829 [+1454, +6130]); WON_H450 is
within CI at 2600 (+676 [−1251, +2577]), significantly worse at 2400
(+1456 [+350, +2579]) and 2700 (+4287 [+2300, +6223]). Reading: the
onset trigger itself is fine — its fixed 300/450 s holds are too short at
2700, where the collapse needs metering well beyond 450 s from onset
(releasing at ~t = 445-595 forfeits most of the benefit the 650 s hold
captures).

**WON_H300 false-engagement cost at 2200:** +1105 total s [+809, +1432],
+1.00 s/veh, only 3/30 seeds benefit — a real cost but **half the fixed
arms'** (W_HOLD +2301, W_ORACLE +2858 at 2200): engaging at the onset
(~148) and releasing at ~450 does less damage than holding to 901+.
It also prevents 3 collapses while inducing 1 (net −2).

**R1b never fires by rule: 0/90 cells-seeds** (all releases at the hard
max, engagement + 900 = t = 1151, mid-plateau). The corrected speed gate is
reachable in principle (400-800 m excludes the walled band) but the
arrival ≤ 0.9 × discharge condition is not: plateau demand keeps the
approach fed, so the approach never empties relative to discharge. After
t = 1151 the band has already cleared, making R1b outcome-identical to
W_ORACLE in delay terms (per-seed totals equal to the cent). Both state
release rules (1c R1, 1c-2 R1b) are inert on this demand family; only the
density rule (R2) ever fired.

**Safety.** Zero collisions/teleports everywhere; low-TTC excess ≈ 0
(median ≤ 0 per arm-cell). Hard braking remains the dose-linked cost:
median excess +100 to +300 vehicle-steps vs D, at every cell and arm.
Location x phase (new arms): at 2400 the braking concentrates at the merge
edge and near-merge (< 200 m) during hold and post-release; upstream
(> 800 m) braking grows with hold duration.

**Flow.** Merge flow 300-1200 s: unchanged vs D for every arm except the
over-holds (R1b/ORACLE −8 to −14 completions at 2400/2600/2700 — the hold
pushes completions past the branch end, not a throughput loss: D-completion
medians are 756/808/834).

## Part C — fair threshold baseline (L0'), refit (L1'), onset AUCs

Conventions as the v1 gate fit (5-fold GroupKFold by seed, fold seeds
7/17/29, value = −Δ + λ·prevented when engaged, s/veh; CIs are seed-level
bootstrap CI90, 2000 resamples). Rows: 720 (seed × t_d) with the corrected
v2 vectors.

**C1 — L0'** (single feature + threshold by training-fold policy value):
OOF value **1.471** (λ = 0; engagement rate 0.53) and **1.435** (λ = 2).
Chosen features per fold: `upstream_inflow_30s > 21` (3/5 folds at λ = 0),
`near_merge_count > 44-45` (2/5); similar at λ = 2. The fair baseline is
essentially the v1 L0 (−0.012 [−0.158, +0.139]): a raw loading threshold.

**C2 — L1'** (L1 features + repaired cav_count_band, cav_count_upstream,
ramp_queue_length): ridge OOF R² 0.168, Spearman 0.414, AUC 0.757, Brier
0.214 — statistically indistinguishable from L1 on the defective vector
(R² 0.168/0.417/0.760). The repaired features do not move delay prediction.

Policy values (λ = 0): never 0; always-W250 0.705; L0 1.482;
L0' 1.471; L1 ridge 1.608; **L1' ridge 1.631**.
**Headline: L1' does NOT beat L0' at λ = 0 with CI clear of zero**
(+0.160 [−0.014, +0.328]); nor does L1 (+0.137 [−0.023, +0.289]). At λ = 2:
L1' vs L0' +0.146 [−0.023, +0.311] (not clear); L1 vs L0' +0.213
[+0.046, +0.372] (clear). Every learned rule clearly beats never and
always-W250. The honest summary: the delay-channel gate sits at the level
of a loading threshold; the multivariate machinery adds no robust value on
top of it.

**C3 — onset prediction is the strong signal.** Label "D-arm onset within
the next 100/200 s of t_d" (base rate 0.124/0.125 over 720 decisions):

- Univariate top: **merge_edge_mean_speed AUC 0.832**, post_merge_mean_speed
  0.773, h 0.708, inflow_minus_discharge_30s 0.693 — two features exceed
  0.75.
- **Multivariate grouped-CV L2 logistic: OOF AUC 0.944** (both horizons;
  range across fold seeds ≈ ±0.01).

Compare: predicting the W-arm delay delta has OOF R² 0.168 / AUC 0.760.
Predicting *whether the collapse arrives soon* is nearly solved with the
same RSU vector. This is the Part-2 hypothesis confirmed quantitatively.

## Part D — lead-time analysis

lead = D-onset − engagement over all 1680 W decisions (720 p20 + 480 1c +
480 1c-2); "no onset" bin is EMPTY — every D branch eventually sustains a
10 s h < 0.15 run.

| lead bin | n | mean Δdelay (s/veh) | prevention rate |
|---|---:|---:|---:|
| > +100 (early) | 5 | +2.88 | 0.80 |
| 0..+100 | 98 | +0.30 | 0.16 |
| −100..0 | 719 | −0.61 | 0.04 |
| < −100 (late) | 858 | −1.65 | 0.07 |

Prevention concentrates where the wall engages before the onset (0.16 at
0..+100 vs 0.04-0.07 after; the > +100 estimate rests on n = 5). Mean Δdelay
within a demand bin moves only modestly with lead (at 2700: −6.16 at
0..+100 vs −5.27 at < −100; at 2200: +4.07 vs +2.59) — the demand regime,
not lead, flips the sign of the delay effect. By hold (300/450 s, the 1c-2
arms): later engagement bins show larger benefit at 2600/2700 (−3.2 to
−3.3 s/veh at < −100), again demand-confounded.

## Surprises and decisions

1. **Onset prediction is nearly solved (OOF AUC 0.944) while delay-benefit
   prediction is not (R² 0.168).** Decision: the next predictor target
   should be onset-time (or remaining-time-to-onset), with the engagement
   rule built on it — the current results show triggering is not the
   bottleneck; hold-duration selection is.
2. **The onset-triggered arms lose at 2700** (release too early), tie at
   2400/2600. Decision: keep WON as the trigger mechanism but couple it to
   a demand-aware hold (or an R2-style extension condition), not a fixed
   300/450 s.
3. **R1b is inert like R1** — the arrival/discharge gate cannot clear on
   plateau demand (0/90). Decision: drop arrival/discharge release rules
   for this demand family; density-based (R2) or onset-based control are
   the only live options measured.
4. **Holds longer than ~650 s from t = 251 are never the best arm**, and
   the 950 s oracle loses completions (−8 to −14) at every cell above 2400.
5. **The false-engagement cost of onset triggering at 2200 is modest**
   (+1.0 s/veh) and smaller than the fixed reference holds' cost there —
   the gate-relevant asymmetry is favorable.
6. `ramp_queue_length` (5 m/s) is genuinely always-zero at decision times
   on this corridor — recorded honestly; it cannot be a predictor feature
   here. The `lane_*_nearest_to_band_top` feature is censored-by-design
   (0 while the band holds a CAV) and should be recorded unclipped in any
   future schema.

## Artifacts

- `runs/probe/stage1b_features_v2_p20_s7001_7240/` — corrected vectors,
  240/240 digest-verified.
- `runs/probe/stage1c2_arms_m{2400,2600,2700}_p20_s*/`,
  `runs/probe/stage1c2_won_m2200_p20_s7401_7430/` — treatment branches with
  location x phase safety aggregates and R1b rule inputs.
- `runs/probe/stage1c2_arms/summary.json`,
  `runs/probe/stage1c2_partC/partC_summary.json`,
  `runs/probe/stage1c2_partD/lead_time_summary.json`.
- Code: `rsu_feature_capture.py`, `rerecord_stage1b_features.py`,
  `policies/stage1c2_arms.py`, `run_stage1c2_arms.py`,
  `analyze_stage1c2_arms.py`, `fit_stage1c2_partC.py`,
  `analyze_lead_time.py`; tests `test_rsu_feature_capture_variance.py`
  (SUMO smoke), `test_stage1c2_arms.py` (pure logic). Full suite
  1026 passed before compute; `deterministic_sumo.py` lock-path fix
  included.
- All seeds are development seeds already registered under
  stage1b_generation / stage1c_release roles and excluded from prospective
  evaluation; no new seed blocks were touched.

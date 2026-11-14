# CHECK 1 (WM-S-R4-CHK-TYPEID-20261011) — driver TYPE identification from observed history, and physics-only prediction with IDENTIFIED types

Analysis only (CPU rollouts, no training/SUMO/GPU). Rollout harness = the DIAG-ORACLE
physics-variant harness (entries OFF, CAVs teacher-forced, seed 0, dt 0.5 s, R4 checkpoint
`runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt`, residual OFF via zero gate, MOBIL
baseline lanes, per-state scorer `wms.eval.score.car_metrics`). Identifier inputs: ONLY the
pre-t0 observable history (own/leader positions+speeds, lanes, 0.5 s ticks, length 5 m);
truth types are evaluation-only and never an identifier input. All "state-median" numbers are
the ROUND4_SCORE metric (median over states of per-state HDV median |ds|, m).

## 1. Identifier (primary, no learning)

Per-vehicle posterior over the 3 campaign vTypes (DIAG-ORACLE parameters; v0 =
speedFactor x lane limit, mainline 31.29 / on-ramp 22.22 m/s):

- car-following ticks (same-lane leader < 100 m) and near-saturation free-flow ticks
  (v >= 0.85 x lane limit): Student-t(df=3) likelihood of the observed 0.5 s acceleration
  around the candidate-type IDM prediction (port of `wms.physics.idm`, clamp [-decel, accel]
  then [-9, 3], gap floor 0.3 m);
- all free-flow ticks: overspeed evidence -0.5*max(0, v - v0 - 1 m/s)^2 (speed sustained
  above a type's desired speed counts against it; underspeed does not — may be accelerating);
- population prior 70/21/10 (normal/cautious/aggressive, normalised).

Design fixed from **label-free** diagnostics only (`scripts/exp_noise.py`): free-flow
acceleration-phase ticks are heavily biased (mean residual −1.7/−0.9/−0.15 m/s^2 for
aggressive/normal/cautious, std 1.7–4.1) and are excluded; the single noise scale is the
label-free pooled residual std around each vehicle's naive best fit (1.44 h60 / 1.26 real).
The literal textbook variant (Gaussian, std = sigma, all ticks) was measured as sensitivity:
0.913 at 5 s but 0.78 at 60 s (long-window bias). No supervised classifier was needed
(label-free likelihood already ~0.93), so none was fit.

## 2. Identification accuracy vs history length (t0-roster HDVs; truth evaluation-only)

`wall_same_real` (120 states, 60 s native history, truncated from t0 backwards):

| history | 5 s | 10 s | 20 s | 30 s | 60 s |
|---|---:|---:|---:|---:|---:|
| accuracy | 0.929 | **0.931** | 0.930 | 0.914 | 0.877 |
| recall aggressive | 0.76 | 0.81 | 0.81 | 0.80 | 0.71 |
| recall normal | 0.95 | 0.94 | 0.94 | 0.93 | 0.91 |
| recall cautious | 0.93 | 0.95 | 0.94 | 0.91 | 0.83 |

`wall_same` h60 (67 states, 5 s dense history): 5 s accuracy **0.925** (0.72 / 0.96 / 0.89).
Confusion at 10 s (real set), rows true / cols predicted (n = 9313):

| | aggressive | normal | cautious |
|---|---:|---:|---:|
| aggressive | 749 | 145 | 28 |
| normal | 251 | 6103 | 126 |
| cautious | 76 | 69 | 2196 |

By condition (10 s, real set): car-following 0.94 (n 7700), jam (mean v < 5 m/s) 0.94 (n 714),
mixed 0.88 (n 696), free (leader-share <= 0.1) 0.74 (n 203); h60 5 s: CF 0.94, jam 0.89,
free 0.85. By leader-within-100 m share bins (<0.1 / 0.1–0.5 / >=0.5): 0.77 / 0.79 / 0.94.
Mean max-posterior 0.99+ (overconfident — posteriors should not be read as calibrated
probabilities). Dominant confusion aggressive->normal (tau 0.95 vs 1.10 is close under the
~1.2 m/s^2 effective behavioural noise). **More history hurts beyond ~10–20 s**: per-tick
evidence is misspecified (SUMO per-vehicle noise realisations + LC/merge dynamics), so old
ticks add bias faster than information — 5–10 s before t0 is the most informative window.

## 3. Harness verification (this run vs DIAG-ORACLE, wall_same 67)

| variant | 10 s | 30 s | 60 s | 120 s |
|---|---:|---:|---:|---:|
| phys_prior (DIAG-ORACLE) | 3.72 | 10.17 | 15.68 | 53.12 |
| phys_prior (this run) | 3.72 | 10.17 | 15.68 | 53.12 |
| phys_true (DIAG-ORACLE) | 0.46 | 1.34 | 4.31 | 48.65 |
| phys_true (this run) | 0.46 | 1.34 | 4.31 | 48.65 |

Max |diff| 0.004 m (rounding) — the harness reproduces DIAG-ORACLE exactly.

## 4. Physics-only rollouts with IDENTIFIED types (state-median |ds| m; blow-up rule >50 m @60 s AND >100 m @120 s)

wall_same 67 (5 s identified types):

| variant | 10 s | 30 s | 60 s | 120 s | blow-ups |
|---|---:|---:|---:|---:|---:|
| C1 (cache) | 2.04 | 5.07 | 8.71 | 99.84 | 0/66 |
| R4 (cache) | 2.17 | 4.88 | 9.96 | 110.69 | 0/66 |
| phys_prior (default params) | 3.72 | 10.17 | 15.68 | 53.12 | 7/66 |
| phys_true (oracle) | 0.46 | 1.34 | 4.31 | 48.65 | 3/66 |
| **phys_ident5 (identified)** | **0.48** | **1.81** | **5.12** | **46.54** | **0/66** |
| phys_ident5_post (posterior-weighted) | 2.08 | 5.79 | 12.86 | 49.92 | 0/66 |

Paired state-median diffs (variant − ref, world bootstrap 95% CI, 2000 draws):
phys_ident5 − C1 = −2.9 [−5.1, −0.5] @60, **−39.7 [−62.7, −4.8] @120**;
phys_ident5 − R4 = −2.0 [−6.0, −0.9] @60, −23.2 [−46.7, +3.0] @120;
phys_ident5 − phys_prior = −11.3 [−18.2, −5.9] @60;
phys_ident5 − phys_true = +0.4 [+0.1, +0.8] @60, +0.0 @120 — **within 0.8 m of the
TRUE-type oracle floor at <=60 s, and 2.1 m better at 120 s** (misclassifications act as
mild regularisation here; 0 vs 3 blow-ups). Posterior-weighted params are worse (param
averaging blurs the IDM); argmax is the right feed.

real-history set (wallc1_states_h60_real; 60 s identified types; quick-20 = the cached C1/R4 slice):

| variant | 10 s | 30 s | 60 s | 120 s | blow-ups |
|---|---:|---:|---:|---:|---:|
| C1 (cache) | — | — | 16.80 | 66.4 | — |
| R4 (cache) | — | — | 27.73 | 78.5 | — |
| phys_true_real (oracle) | 0.73 | 1.70 | 5.40 | 20.74 | 1/19 |
| **phys_ident60_real** | **1.14** | **5.16** | **9.52** | **19.19** | 1/19 |
| phys_ident60_post_real | 1.14 | 5.23 | 9.60 | 19.19 | 1/19 |
| phys_ident60_real (full 120 states) | 0.66 | 3.41 | 9.22 | 26.94 | 6/114 |

Paired (quick-20): phys_ident60 − C1 = −5.5 [−7.4, −3.5] @60, **−25.2 [−61.4, −6.0] @120**;
− R4 = −12.2 [−18.2, −3.3] @60, −16.4 @120; − phys_true = +0.6 [+0.2, +10.0] @60,
+0.2 [−0.8, +6.2] @120. Note the 60 s-history identification is the *worse* window
(0.877 vs 0.931 at 10 s) and still lands within ~1–4 m of the oracle. Pooled @120:
phys_ident60 14.1 m med |ds| vs oracle 6.8 m; signed ds +105 vs +103 (same fast-drift mode
as all physics-only variants); wrong-lane 10.4% @60 vs oracle 10.3%.

## 5. Reading (measured vs inference)

Measured:
1. **Types ARE identifiable from short observable history**: 92–93% per vehicle from 5–10 s
   (car-following conditions 94%); the 3 vTypes are separated mainly by tau/minGap in
   car-following and by speedFactor in free flow. Free flow is the weak regime (74–85%) and
   aggressive is the confusable class (recall 0.72–0.81, mostly -> normal).
2. **Feeding IDENTIFIED types into the physics nearly achieves the TRUE-type floor**:
   wall_same 5.12 vs 4.31 m @60 s (paired gap +0.4 m, CI [+0.1, +0.8]), 46.5 vs 48.7 @120 s;
   real set 9.52 vs 5.40 @60 s (+0.6 paired, CI up to +10), 19.2 vs 20.7 @120 s. With only
   5 s of history the identification error adds < 1 m @60 s to physics-only prediction.
3. **Identified-type physics beats both trained models at every horizon** (wall_same
   −2 to −3 m vs C1/R4 @10–60 s, −40 m vs C1 / −23 m vs R4 @120 s; real set −5.5 / −12.2 m
   @60 s, −25 / −16 m @120 s) and beats default params by 11.3 m @60 s. Blow-ups drop to
   0/66 (wall_same).
4. The posterior-weighted blend is worse than argmax — feed discrete identified settings.
5. The 120 s bottleneck is unchanged by identification (wave-phase drift: signed ds +204 m
   vs +211 m oracle, survivor-speed overshoot identical to TRUE-type physics) — consistent
   with DIAG-ORACLE: the remaining error is lane/wave structure, not parameter knowledge.

Inference (not directly measured): since identification is ~free (5 s history suffices) and
closes most of the prior->true parameter gap, **option A (feed identified driver settings
into the physics core) is validated as cheap and effective at <=60 s**; the residual ~4–9 m
@60 s floor and the 120 s drift require the lane/jam-phase corrections DIAG-ORACLE
identified, not better type identification. An identified-types physics core + a
wave/lane-aware residual is the natural combination (the residual should be re-trained
against identified-type physics, not prior physics).

**Answer.** Identification accuracy vs history: 0.93 (5 s) / 0.93 (10 s) / 0.93 (20 s) /
0.91 (30 s) / 0.88 (60 s). Physics-only with identified types: wall_same 5.12 m @60 s,
46.5 m @120 s (oracle 4.31 / 48.65; C1 8.71 / 99.84; R4 9.96 / 110.69); real set 9.52 /
19.2 m (oracle 5.40 / 20.74; C1 16.80 / 66.4; R4 27.73 / 78.5). Option A is worth it:
identification needs only ~5–10 s of observable history, is free of truth labels, and
converts physics-only prediction to near-oracle quality at <=60 s while halving the 120 s
error of C1/R4 — but it does not fix the 120 s wave-phase/lane drift.

Artifacts: `results.json` (all tables, paired CIs, pooled/per-type splits, identification
tables), `out/` (truth_full.json, ident_eval.json, ident_rows.pkl, noise_study.json,
variant_*.pkl, logs), `scripts/` (typeid.py, exp_noise.py, identify_eval.py, run_rollouts.py,
aggregate.py, build_truth_full.py). True types stayed evaluation-only.

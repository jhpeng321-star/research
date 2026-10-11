# REPORT — DIAG-HIST: why R3/R4 fail on real-60 s-history states while C1 does not

Run: `E:/research-dense/runs/active/WM-S-R4-DIAG-HIST-20261011` (analysis only; CPU, <=5
worker processes, no training/SUMO/GPU; reads only under `E:/research-dense`, writes only
here and in `E:/agent-work/tmp`). Python `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`,
`PYTHONPATH=E:/research-dense/src`, `TEMP/TMP=E:/agent-work/tmp`, `PYTHONDONTWRITEBYTECODE=1`,
1 torch thread per worker. Machine numbers all come from `out/*.json`; this file is
regenerable from `scripts/`.

## 0. Setup, provenance, integrity

* Scorer: the round-4 clean scorer contract (`wms.eval.score.evaluate_state`, teacher-forced
  CAVs from the recorded `wall` arm, entries OFF, 120 s, seed 0, horizons 10/30/60/120 s).
* Models: R4 `WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt`
  (sha256 `3d73f4ac7c49abe643d63d8df5998da91c1eceed7f5759a312cc80e7c2d3f49f`),
  C1 `data_cache/ckpt_c1fast/wm_s.pt` (sha256 `87ed02e0707e5e4eefdd2dbb0fd282f834c1288776f7f395b41ab4d13826aaef`).
  C1 per-state values are the round-4 cache (`score/cache/C1__wall_same_real.json`).
* R3 numbers are quoted from `WM-S-R3-DIAG-20261010` (same quick-20 states, legacy loader);
  R3 was NOT rerun (its loader workspace is outside this task's read scope).
* **Integrity**: my unmodified-R4 rollouts reproduce the round-4 cache bit-exactly
  (max |diff| 0.0 over 79 comparable per-state horizon values; single-state smoke test
  identical to 8 digits). All 20 quick-20 states scored for every variant.
* Commands (from this directory): `python -B scripts/q1_properties.py`, `... q2_encoder.py`,
  `... q5_history_format.py`, `... q5_train_gaps.py`, `... q4_interventions.py` (5 workers),
  `... q3_blowups.py`, `... make_results.py`.

## 1. (Q1) Are the real-history states intrinsically harder? — YES, moderately

* The two files contain **no identical (world, t0) pairs**: every real-history state is the
  SAME run sampled **exactly +60.5 s later** (paired t0 diff 60.5 s, IQR 0; 15 of 20 quick-20
  worlds shared). "Pairing" is therefore by world.
* State properties at t0 (medians, real vs std quick-20; full-set in `out/q1_properties.json`):
  HDVs 92 vs 78.5; mean HDV speed 18.4 vs 21.8 m/s; share<5 m/s 0.18 vs 0.14; ramp HDVs 6 vs 4;
  recorded 2->1 merges in 120 s 3.5 vs 2.5; **entrants after t0: 0 in both** (entries are not
  the story); density ~equal (32.3 vs 32.9 veh/km span); junction-lane share 0 vs 0.011.
* Paired by world (n=15, real - std): C1 60 s error **+5.94 m** (p25 -0.9, p75 +16.9);
  R4 **+25.15 m** (p25 +7.8, p75 +30.3). So the states are genuinely harder (+6 m for C1,
  consistent with 16.8 vs 11.96 on the quick-20 std set), but that explains only ~1/4 of
  R4's degradation. R4/C1 per-state ratio is ~1.0 on the easiest and the most-jammed states
  and 1.6-5.0 on mid-jam states (top offenders: w8663 51.5 vs 10.4, w8687 34.6 vs 7.0,
  w8680 107.0 vs 32.5). No property correlates strongly (best Spearman: n_hdv0 +0.49).

## 2. (Q2) Driver-encoder outputs: real vs standard vs training — mildly shifted, not pathological

Training reference: 220 windows from the R4 train list `lpwo_v3/windows_train_noPI_noWall.json`
(163 shifted / 57 not; 21,550 present HDV slots), featurised with the trainer's own
`prior_history`/`long_history_features` (history_s 60, stride 1 s).

* **Valid history length**: real states median 61 ticks (55% of slots full 60 s, 3% <5 ticks);
  training median 39 (38.5% full, 27% <5); std states all 5 ticks. Real histories are well
  inside the training length distribution.
* **The 9 IDM/MOBIL parameters are nearly constant everywhere**: e.g. v0 train q5-q95
  30.91-31.20 vs real 30.90-31.23 vs std 30.94-31.05; **zero slots at any clip bound** in
  train/real/std (shares 0.0000). The shrinkage (median 0.83) plus small tanh deltas leave
  them within a few % of the vType defaults. Real-vs-train outside-5-95% shares are 6.6-18.2%
  per parameter (std: ~0%) — i.e. the real histories move the inferred parameters slightly,
  but the parameters carry almost no signal in the first place.
* **Style embedding (102-d)**: real mean |z| vs training 0.76 (p95 1.92, max 7.9), std 0.69
  (p95 1.68); mean fraction of dims outside train 5-95% = 10.7% real vs 7.5% std (training
  baseline 10% by construction). Mildly more spread, no extremes.
* **No correlation with failure**: per-state embedding OOD-ness (frac dims out 0.076-0.148)
  does not track R4 error (blow-up states 0.108-0.129 sit mid-pack; the most OOD states
  w8682/w8686 have R4 errors 9.5/5.7 m).

## 3. (Q3) What is special about R4's 3/19 blow-up states? — congestion-wave bulk failure

Blow-ups (rule |ds|>50 @60 s AND >100 @120 s): **w8663m2700p10t480** (51.5/658.9 vs C1
10.4/183.7), **w8680m2600p20t300** (107.0/1003.8 vs C1 32.5/64.4), **w8685m2700p20t480**
(53.1/563.3 vs C1 32.3/61.0). It is NOT a few vehicles: top-10 vehicles carry only ~50% of
the error mass (vs 74-84% in easy states), i.e. the median HDV is wrong.

* **w8663 / w8680 (jam-discharge failure)**: worst vehicles stand in a queue at t0 (v0 1.2-4.8
  m/s, clustered at s0 -2600..-2500 and -270..-230). Recorded: jam dissolves, they reach
  ~25 m/s by 60 s. R4: they crawl (v at 60 s 4-18 m/s, dipping to 0), ending 270-515 m
  BEHIND (ds = p - r < 0).
* **w8685 (phantom-jam failure)**: worst vehicles approach at 28 m/s; recorded they slow
  moderately (16 m/s at 60 s); R4 stops them dead (v 0.0-0.7 at 60 s), 160-210 m behind.
* Wrong-lane share at 60 s in blow-ups is unremarkable (0.07-0.31 vs C1 0.27-0.33);
  embedding |z| of the worst vehicles (0.4-1.6) equals the state average. Predicted lane
  changes over 120 s are 330-854 vs 15-28 recorded (11-30x) — but easy states also over-predict
  (134-432 vs 5-18), so lane excess alone does not separate them.
* Common denominator: strong stop-and-go waves mid-development (queue discharge in one
  direction, wave arrival in the other) — exactly the dynamics 60 s after a wall forms, and
  absent from the standard states taken ~5 s after wall assembly.

## 4. (Q4) Clamp / prior intervention test — encoder outputs are NOT the dominant cause

All variants on the real-history quick-20, R4, paired vs unmodified R4 and vs C1
(negative favours the variant; CI over worlds, 2000 resamples):

| variant | 60 s median | 120 s median | blow-ups | paired vs base @60 s | paired vs C1 @60 s |
|---|---:|---:|---:|---|---|
| C1 (cache) | 16.80 | 66.38 | 0/19 | — | — |
| R4 base | 27.73 | 78.47 | 3/19 | — | +2.674 [+0.068, +13.101] |
| R4 paramclamp (9 params -> train 5-95%) | 27.73 | 78.47 | 3/19 | **0.000 exact no-op** | +2.674 (same) |
| R4 paramprior (9 params -> vType defaults) | 27.73 | 78.47 | 3/19 | **0.000 exact no-op** | +2.674 (same) |
| R4 embclamp (embedding -> train 5-95%) | 23.27 | 84.94 | 2/19 | -0.535 [-2.184, 0.000] | +1.540 [-0.382, +9.152] |
| R4 embprior (embedding -> train median) | 30.65 | 107.22 | 3/19 | +1.430 [+0.124, +3.291] | +5.461 [+2.098, +14.919] |

* **Architectural fact, verified numerically**: in R4 (rollout_mode=residual) the 9 inferred
  parameters never feed the rollout — physics uses the vType defaults and the residual is
  conditioned only on the fused embedding. Clamping or replacing the parameters changes
  nothing (max per-state |diff| 0.0 across all horizons). The same holds for R3.
* Clamping the embedding to the training 5-95% band recovers only a little (60 s medians
  27.7 -> 23.3; paired -0.5 [-2.2, 0.0]; blow-ups 3 -> 2) and is slightly WORSE at 120 s
  (+3.0 [+0.3, +10.1]). Replacing the embedding with the population prior is clearly harmful
  (+1.4 at 60 s, +14.2 at 120 s) — consistent with R3-DIAG's zero-embedding (34.2) and
  empty-history (39.1) results. The encoder is used and roughly helps; its outputs are not
  the failure channel.

## 5. (Q5) Is the history input malformed? — NO

All 20 states pass every structural check: metadata (history_s 60, stride 1.0 s, dt 0.5,
span 60.0 = 121 ticks), kin [121,192,7] finite, lane one-hot valid, speeds 0-33.8, accels
-4.5..+3.0, lengths 5 m, ids unique, t0 kin row equals `initial_state` for all 101 t0
vehicles, all initial ids covered by history ids, ordering correct. Featurised channel
quantiles match the training sample almost exactly (v/30 q50 0.887 vs 0.888; a/3 0.000 vs
0.000; gap/100 0.510 vs 0.505; dv/30 0.000 vs 0.000; leader_seen 1.0; disp/250 real -5.82..
-0.05 vs train -5.75..-0.01 — the STD states, truncated to 5 s, are the odd ones out).
Junction-lane share in observed history rows 0.66% (train t0 rows 0.0% but training history
rows also contain junctions via the same pack machinery). Interior mask gaps inside a
vehicle's span exist in every state (83-616 cells, ~1%) **and in every training window**
(median 0.89% of cells, 100% of windows have some) — same semantics, handled identically by
the shared continuity chain. Raw dynamics of the observed history: share v<5 = 17.2% real vs
18.0% training; |a|>3 = 0.77% vs 0.86%. The input is well formed and in distribution.

## 6. Verdict (measured vs inference)

**Most likely cause, measured**: the real-history states are the same runs 60.5 s later,
with developed wall-induced congestion (more HDVs, slower, more merges; C1 +5.9 m paired).
On those states R3/R4's conditioned residual core mishandles congestion waves — queue
discharge too slow (vehicles held at 4-18 m/s where the recording reaches ~25 m/s) and
phantom stops of approaching platoons — producing bulk (median-HDV) errors and the 3
blow-ups. Wrong-lane share, entries, merge calibration, history length, and input format
are all ruled out (Q1, Q3, Q5; R3-DIAG D3/D1).

**Encoder specifically, measured**: not the cause. The 9 inferred parameters are
near-constant, unbounded-free, and (in residual mode) architecturally inert — clamp/prior
is a bit-exact no-op. The style embedding is only mildly shifted (mean |z| 0.76, p95 1.9),
does not correlate with per-state error, clamping it recovers ~0.5 m paired (medians -4.5)
at 60 s while hurting at 120 s, and neutralising it (prior/zero/empty) hurts. R3-DIAG
already showed cutting the history to 5 s does not rescue R3 (27.2 vs 28.3).

**Inference (beyond measurement)**: the failure is the R3/R4 *residual-on-congested-states*
regime, not the history pathway. Candidate contributors that C1 lacks and that cannot be
separated without retraining: (i) lpwo_v3 dropped the eval worlds entirely, while C1 trained
on these worlds' other arms — the traffic compositions are in-distribution for C1 only;
(ii) the R3/R4 guards (soft residual bound 4 + residual-size penalty) plausibly under-correct
exactly where large corrections are needed (jam discharge); (iii) the conditioning adapter
amplifies embedding variation in these states (embclamp helps a little).

**Implications for C2 (the history-encoder track)**: judge and train C2 on real-history
(60 s) states — the standard ~5 s states cannot see this failure. Do not spend effort on
parameter clamping/priors (inert) or input-format fixes (nothing to fix). Keep the encoder
(it beats its ablations), but pair any C2 model with a congestion-rich training mix that
includes wall/PI-exposed arms (or the eval worlds' other arms) and check the jam-discharge
fingerprint (queue discharge rate, phantom-brake rate on fast platoons) at 60/120 s.

# DIAG 4 — how much of the ~100 m / 120 s error is avoidable with the TRUE driver settings?

WM-S-R4-DIAG-ORACLE-20261011. Analysis only (CPU rollouts, no training/SUMO/GPU). Set: `wall_same`
h60, 67 states, entries OFF, CAVs teacher-forced, seed 0, dt 0.5 s, per-state scorer
`wms.eval.score.car_metrics` — identical to the R4 scoring harness. Physics core / prior from the
R4 checkpoint `runs/active/WM-S-R4-20261011/box/gpuA/r4_main_a1/wm_s.pt` (rollout_mode=residual,
saturating gate, `sumo_fidelity` IDM prior (a,b,T,s0,delta) = (2.49, 4.2, 1.135, 2.6, 4),
v0 = 31.29*0.992 = 31.04 m/s, MOBIL baseline lanes, no projection).

**Verification:** this harness re-running R4 unmodified reproduces the R4 score cache exactly
(267/267 per-state `ds_med` checks at 10/30/60/120 s, max |diff| = 0.0). C1/R4 comparators come
from the same cache (C1 8.71 / 99.84 m, R4 9.96 / 110.69 m state-median at 60/120 s; the cache was
itself reproduced bit-exactly by WM-S-R4-DIAG-ERR-20261011). All "state-median" numbers below are
the ROUND4_SCORE metric (median over states of per-state HDV median |ds|); "pooled" is per-HDV.

## 1. True driver settings of the wall-campaign worlds (Step 1 — measured)

Generator chain: `traffic_shepherd/scenarios.py::build_scenario` writes per-world
`run/scenario/requests.rou.xml` with **three explicit IDM vTypes** copied from the scenario
template (vTypeDistribution, with `speedDev` frozen to 0) plus **one `<vehicle type=...>` row per
scheduled trip**; the per-vehicle type assignment is drawn once per world
(`damping_type_v1:{seed}:{id}`) and never changes at runtime. There is **no per-vehicle hetdata
sidecar** for these campaign runs. All 29 worlds present in `wall_same` carry **byte-identical
vType definitions** (asserted; `out/truth.json`), and every vehicle id of every state maps to a
type (6,967 ids: 69.7% normal, 20.7% cautious, 9.6% aggressive). Verified from
`runs/campaign_stage1_chunk4/{world}__base/run/scenario/requests.rou.xml` (scenario copies in the
arm runs are sha-identical).

| vType | carFollow | tau | minGap | accel | decel | speedFactor | sigma | lcCoop | lcAssertive | impatience |
|---|---|---|---|---|---|---|---|---|---|---|
| aggressive_car | IDM | 0.95 | 1.5 | 3.0 | 4.5 | 1.08 | 0.6 | 0.3 | 0.6 | 0.7 |
| normal_car | IDM | 1.10 | 2.5 | 2.6 | 4.5 | 1.00 | 0.4 | 0.6 | — (SUMO default) | 0.3 |
| cautious_car | IDM | 1.50 | 3.5 | 2.0 | 3.0 | 0.92 | 0.2 | 1.0 | — (SUMO default) | 0.0 |

Known **per vehicle**: the type (= all its parameters) and the exact speedFactor (`speedDev=0`,
so no per-vehicle speed-factor draws exist). Known **per type only**: everything above — there are
no per-vehicle parameter draws anywhere in these worlds. **Not deterministically recoverable**:
the per-action-step `sigma` noise draws (0.6/0.4/0.2 driver imperfection), LC2013 internal state
(impatience accumulation, strategic/speed-gain lookahead), and lateral positions during lane
changes. `delta` is not set → SUMO default 4 (same as the model). Network: mainline lanes
31.29 m/s, on-ramp 22.22 m/s (SUMO applies per-lane limits; the model has one v0). SUMO ran with
`--step-length 0.5`; `actionStepLength` not overridden in the launcher.

Mapping into the model's 9 driver params (documented choice): IDM five set to TRUE —
v0 = 31.29 x speedFactor (33.79 / 31.29 / 28.79), time_headway = tau, min_gap = minGap,
accel, decel. Lane params (politeness 0.2, mobil_threshold 0.1, merge_gap 6.0, merge_urgency 1.0)
left at the R4-core defaults because LC2013 has no MOBIL counterpart — flagged; variants (a)-(c)
then differ only in the longitudinal parameters.

## 2. Variants (Step 2)

| variant | 10 s | 30 s | 60 s | 120 s | pooled med 60/120 | blow-ups | wrong-lane 60/120 | mean dv @120 (m/s) | signed ds @120 (m) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 (cache) | 2.04 | 5.07 | 8.71 | 99.84 | 10.13 / 112.81 | 0/66 | 14.7% / 36.4% | -7.2 | -16.9 |
| R4 (cache = r4_vanilla) | 2.17 | 4.88 | 9.96 | 110.69 | 9.88 / 93.50 | 0/66 | 9.7% / 21.4% | -5.0 | -42.0 |
| (c) phys_prior | 3.72 | 10.17 | 15.68 | 53.12 | 18.14 / 50.53 | 7/66 | 11.0% / 24.7% | +3.3 | +255.9 |
| (a) phys_true | **0.46** | **1.34** | **4.31** | **48.65** | 3.09 / 39.39 | 3/66 | 10.1% / 24.5% | +2.9 | +211.0 |
| (b) phys_true_tfl | 0.44 | 0.71 | **1.00** | **3.91** | 0.84 / 3.16 | 0/66 | 1.0% / 1.7% | +2.4 | +158.1 |
| (d) r4_true | 3.32 | 15.55 | 41.13 | 164.95 | 42.43 / 135.35 | 23/66 | 11.7% / 25.4% | -4.3 | -148.1 |
| (e) phys_true_sl | 0.46 | 1.34 | 4.31 | 48.65 | 3.10 / 39.39 | 3/66 | 10.2% / 24.5% | +2.9 | +211.0 |
| (e2) phys_true_sl_act | 0.51 | 1.46 | 4.97 | 65.38 | 3.31 / 57.66 | 3/66 | 10.1% / 26.5% | +3.5 | +229.1 |

State-median |ds| (m); pooled median over per-HDV values; blow-up rule >50 m @60 s AND >100 m
@120 s; dv/ds signed at 120 s over matched HDVs (n = 873 pooled, 66 states). At 120 s
phys_true has 19/66 states >100 m (13 >300 m); phys_true_tfl 12/66 (7 >300 m); r4_true 45/66.
Median speed of the 120 s survivors: recorded 17.9 m/s; phys_true 21.3, phys_true_tfl 21.2,
R4 7.7 (C1 5.1, DIAG-ERR). 120 s state p90: phys_true 621.6, phys_true_tfl 552.2, R4 282.6.

Paired state-median differences (variant − reference, world bootstrap 95% CI, 2000 draws):

| variant | − C1 @60 | − C1 @120 | − R4 @60 | − R4 @120 |
|---|---|---|---|---|
| phys_true | -3.9 [-6.3, -1.7] | -24.2 [-72.0, 21.1] | -3.1 [-9.4, -1.7] | -3.7 [-45.9, 11.9] |
| phys_true_tfl | -6.5 [-8.4, -4.2] | **-68.0 [-95.7, -51.7]** | -7.7 [-11.9, -3.0] | **-51.8 [-87.3, -33.6]** |
| phys_prior | +7.2 [2.8, 13.3] | -31.5 [-62.0, 11.1] | +6.4 [3.5, 12.9] | -19.1 [-43.5, 29.5] |
| r4_true | +34.0 [20.3, 48.9] | +36.5 [24.8, 79.8] | +32.1 [16.1, 48.6] | +42.3 [2.8, 97.8] |

Pooled per-10 s median |ds| (m): phys_true 0.5 → 1.0 → 3.1 → 8.5 (80 s) → 39.4 (120 s);
phys_true_tfl 0.4 → 0.6 → 0.8 → 2.2 → 3.2; phys_prior 3.5 → 11.2 → 18.1 → 25.2 → 50.5;
r4_true 3.3 → 15.3 → 42.4 → 72.4 → 135.3; C1 2.0 → 4.5 → 10.1 → 24.2 → 112.8;
R4 2.1 → 4.7 → 9.9 → 26.0 → 93.5.

## 3. IDM/MOBIL implementation vs SUMO (differences list)

1. **sigma** (0.6/0.4/0.2): SUMO perturbs the driver each action step; the model is deterministic — irreducible without replicating SUMO's RNG.
2. **Stepping**: SUMO ballistic at 0.5 s (`--step-length 0.5`); model ballistic at 0.5 s (same form). A 1 s action-step hold (e2) makes things *worse* (65.4 vs 48.7 at 120 s) — per-tick decisions match the data better; SUMO's default actionStepLength was not overridden by the campaign launcher.
3. **v0 / speed limits**: SUMO uses speedFactor x per-lane limit (ramp 22.22 vs mainline 31.29); model single v0. Variant (e): identical medians (Δ < 0.1 m) — immaterial here.
4. **delta**: vType-unset → SUMO default 4 = model 4. **minGap/s0**: same standing-gap role in s*; model floors the leader gap at 0.3 m.
5. **Accel bounds**: model clamps IDM to [-decel, accel] per driver then globally [-9, +3] (R4 physics.json); SUMO bounds by vType accel/decel with emergency braking beyond — close but not identical.
6. **Lane changes**: LC2013 (strategic/speed-gain/cooperative, impatience, continuous lateral motion, multi-second occupancy checks) vs MOBIL-style instant lane jumps with 6 m safe gaps and 0.1 m/s² threshold — the dominant structural difference (see (b)).
7. Vehicle length 5.0 m in both (vTypes set none). Junction internal lanes: model carries the slot on its last lane; exits/presence teacher-forced identically in all variants.

## 4. Reading (measured vs inference)

Measured:
1. **The floor given true settings: 4.3 m @60 s and 48.7 m @120 s** (physics-only, TRUE params, MOBIL lanes), and **1.0 / 3.9 m** when lanes are also taken from the recording. TRUE params alone beat both models at <=60 s (paired vs R4 -3.1 m, CI excludes 0; vs C1 -3.9).
2. **Driver-parameter error is NOT the 120 s bottleneck**: physics-only prior->TRUE improves 60 s 15.7 -> 4.3 m but 120 s only 53.1 -> 48.7 m; inside the trained R4, TRUE params make everything *worse* (+32 m @60, +42 m @120, 23 blow-ups) — the residual has learned to correct the prior physics, so R4's ~110 m is not an encoder/parameter-identification failure.
3. **Lane handling is the big avoidable slice**: TRUE params + recorded lanes cut 120 s from 48.7 to 3.9 m (paired vs C1/R4 -68/-52 m, CIs exclude 0) and wrong-lane share from 24.5% to 1.7%.
4. **What remains is stop-and-go wave phase**: physics-only drifts the fleet *fast* (signed ds +211 m, survivor speed 21.3 vs 17.9 recorded) while C1/R4 drift *slow* (-17/-42 m; 5.1/7.7 vs 17.9 recorded) — the queue front is misplaced in opposite directions; even the (b) floor is bimodal (12/66 states >100 m, all jam states; median state 3.9 m).
5. (e) per-lane speed limits: no effect; (e2) 1 s action stepping: worse — neither SUMO detail explains the gap.
6. TRUE params at 10 s/30 s: 0.46/1.34 m vs models' ~2/5 m — short-horizon model error is mostly prior-parameter mismatch; at 120 s it is structure (wave phase) + lanes.

Inference (not directly measured): the residual 120 s advantage of R4 over physics-only (93 vs 39 pooled |ds|... state-median 111 vs 49) comes from damping the *fast* drift; a physics core with TRUE params plus a wave/lane-aware correction, or residual training *conditioned on* true-type physics, should be able to combine both (this run shows the ingredients are individually available). Sigma-noise and LC2013 internal state remain unmodellable per-step, but (b)'s 3.9 m median shows they are small outside jam-phase states.

**Answer.** Floor with TRUE settings: 60 s: 4.31 m (1.00 with true lanes); 120 s: 48.65 m (3.91 with true lanes). Of R4's 110.7 m at 120 s: ~5-10 m is driver-parameter error, ~45 m is lane-decision error, and the rest is wave-phase/structural error concentrated in jam states; none of it is recoverable by feeding TRUE params to the current architecture (that makes it worse) — the avoidable parts are lane timing and jam-wave phase, not parameter identification.

Artifacts: `results.json` (all tables + paired CIs + time profiles + per-type splits),
`out/truth.json` (per-world vTypes + vehicle->type), `out/variant_*.pkl` (per-HDV arrays,
7 variants x 67 states), `scripts/` (build_truth.py, run_variants.py, aggregate.py).
True driver settings are evaluation-only and were not written into any training config or data.

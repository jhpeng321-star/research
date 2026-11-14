# REPLICATE_DESIGN_DRAFT — replicated branch campaign for G-2 and Part A validation

**Pilot finalised 2026-09-28 as REGISTRATION_REPLICATE_PILOT.md (owner-approved). Main stage still a draft.**

Draft for the owner's decision (via main discussion), 2026-09-27. World model session (WD-01).
Design text and a cost estimate only. Nothing is dispatched. Chunk 2 is NOT changed.

## 0. Why

A2 step 2 (ZCode e5265607) ran on chunk-1 TRAIN: 49 worlds, 7 arms, H = 30 s commit window.
- The learned Part A's **paired** delay delta (arm − native) does not match the single-realisation truth: sign agreement 0.58–0.67, Spearman −0.01 to 0.37, Pearson ≈ 0.
- Its MAE (11–16 veh·s) is worse than a "CAVs as native" predictor (6–11).
- True 30 s deltas: mean −0.15, sd 15, p95 |Δ| 34 veh·s.
- Paired TTC<3 deltas: truth ≈ 0 to −0.5, predicted −0.4 to −1.2. Naive (0) is closer.

Reading:
1. At 30 s, the effect of per-CAV action changes is below SUMO's single-realisation stochastic noise (dawdling, lane changes, insertion). One truth branch per arm can't resolve it.
2. Therefore G-2 (validating the paired TTC and hard-brake deltas) and Part A action validation need **expected-value truth**: several replicates per (branch point, arm), with arm and native paired on the same random stream.
3. The planner's ΔQ signal must come mostly from Part B (V over the remaining horizon), which supports DESIGN_PARTB_V2. Replicates at longer horizons also give the noise floor Part B must beat.

## 1. Design

**Pairing mechanism (to verify first; planner-owned perfect simulator, WM-41).**
- Chunk 1 restores the saved state with its RNG (`--save-state.rng`), which gives R = 1 with common random numbers (CRN) across arms.
- For replicate r (r = 1..R), restore the same saved state and then reseed. Either:
  - start SUMO with `--seed s_r` and load a state saved WITHOUT RNG state (so the RNGs come from `s_r`); or
  - restore with RNG, then advance/reseed deterministically by `s_r`, if TraCI allows it.
- Native and every arm of replicate r use the SAME `s_r`, so each replicate is a CRN pair.
- Verification (pilot, before anything else):
  - (a) the same (state, arm, `s_r`) is byte-identical when run twice;
  - (b) different `s_r` give different trajectories;
  - (c) the arrival schedule (pre-generated routes) is identical across r;
  - (d) the per-vehicle RNGs (dawdle, lane change) are covered by `s_r`, including under `--thread-rngs` if used.
- If SUMO cannot reseed after restore, the fallback is: **re-run the base prefix with seed `s_r` up to t0 and save a state per r**. The t0 states then differ slightly. The paired design still holds within r.

**Data regime.**
- Use chunk-2 worlds (post-fix, CAV speedMode 27) from its TRAIN split. Chunk 1 is pre-fix.
- Logging as in the chunk-1 data contract (0.5 s physics, dispatcher rows, suffix commands, outcome), plus `replicate_id` and `seed`.

**Arms.** native, pi, fixed1 (a random per-CAV fixed assignment), and one wall arm if chunk 2 has walls. Switching arms are headroom-only, so they are excluded.

**Branch points.**
- W = 12 worlds, stratified: demand {2400, 2600, 2700} × penetration {0.05, 0.10, 0.20, 0.40}, one world per cell where available.
- Onset timing is mixed.
- One branch point per world, at the world's t0.

**Horizons scored.** 30 s (commit window), 60, 120 and 600 s (Part B remainder). Each replicate branch runs the full 600 s, so all horizons come from the same runs.

## 2. Power

Notation: σ is the sd of the single-realisation paired delta; δ is the smallest expected effect we need to resolve.

- **Minimum replicates.** Using the one-sided 95% test with 80% power:

  R ≥ ((1.645 + 0.84) · σ / δ)².

- **At H = 30 s,** using σ ≈ 15 veh·s from chunk 1 as an upper bound (it also contains between-world variation, so the within-pair noise is likely smaller):

  | δ (veh·s) | min R |
  | --- | --- |
  | 5 | 56 |
  | 10 | 14 |
  | 15 | 7 |
  | 20 | 4 |

- **Separating pairs for G-2.** G-2 must report agreement separately for pairs whose expected delta is clearly nonzero. With R = 8, the SE of the replicate mean is ≈ 15/√8 ≈ 5.3 veh·s, so "clearly nonzero" means |mean Δ| > 2 SE ≈ 11 veh·s. About 30% of chunk-1 30 s deltas exceed that in absolute value (p95 = 34). This gives a usable, if modest, subset.
- **At H = 600 s,** the noise sd is UNKNOWN. The chunk-1 planner offline analysis (PL-02a, planner delay units, one realisation) gives a pi-vs-native saving with median |Δ| ≈ 2.6k and sd ≈ 5.4k, and a best safe saving mean of 3.1k. So effects are large, but the replicate noise could be comparable. **Pilot measures σ_rep(H) at 30/60/120/600 s first,** and R is then re-sized with the formula above.
- **Recommendation:** pilot R = 4, then main R = 8 (resolves δ ≈ 13 veh·s at H = 30, and gives σ_rep for all H). Increase R only if the pilot's σ_rep × 2.5 / δ_target calls for it.

## 3. Budget (at about 30k sim-s/h aggregate throughput)

| stage | worlds | arms | R | branches | sim-s (600 s each) | wall time |
| --- | --- | --- | --- | --- | --- | --- |
| pilot (RNG verification + σ_rep) | 4 | 3 | 4 | 48 | 28,800 | ≈ 1.0 h |
| main | 12 | 3 | 8 | 288 | 172,800 | ≈ 5.8 h |
| optional wall arm | 12 | +1 | 8 | +96 | +57,600 | +1.9 h |
| **total without wall** | | | | **336** | **201,600** | **≈ 6.7 h** |

- Saved states are reused from chunk 2, so there is no warm-up cost.
- If the no-reseed fallback is needed, each replicate adds a prefix re-run to t0 (≈ 240–540 s): up to about +50%.
- **Scheduling:** this can run as a separate small campaign after chunk 2 finishes (about 13:30), under the reconstructor's campaign harness, with its own registration.
- **The pilot's RNG verification gates the main stage.**

## 4. What it unlocks

- **G-2 validation.** Sign agreement and Spearman of predicted vs replicate-mean (plan − native) TTC<3 and hard-brake deltas. Reported separately for clearly-nonzero pairs and near-zero pairs.
- **Part A action validation.** Expected-value targets at H = 30/60/120, instead of noisy single realisations.
- **Part B.**
  - The replicate spread gives the irreducible noise floor of the 600 s ΔQ (the target V must be calibrated to).
  - The native replicates add Monte-Carlo anchors for V0.

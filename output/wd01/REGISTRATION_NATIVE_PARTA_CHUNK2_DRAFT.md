# DRAFT registration: native Part A validation on chunk-2 VALIDATION roots

**SUPERSEDED 2026-09-28 by REGISTRATION_NATIVE_PARTA_CHUNK2.md (frozen, owner decision (B)).**

Status: DRAFT for the owner, via main discussion. NOT frozen and NOT run.
World model session (WD-01), 2026-09-27.

## Why

- The 4 chunk-1-era validation roots are spent: 8 looks (DEV-2). Look #8 failed at H=30 on (c) delay coverage.
- Their truth also comes from pre-fix runs.
- The next validation of the native Part A emulator uses fresh post-fix worlds from chunk 2 (CAV speedMode 27).

## Data

- **Worlds:** chunk-2 worlds with `split == VALIDATION`, or `CALIBRATION` if chunk 2 has no VALIDATION split; the reconstructor to confirm. Use every world with a complete native-branch receipt. HOLDOUT worlds are never opened.
- **Truth:** the `__native` branch of each world over (t0, t0 + 600]. All CAVs run native: SUMO car-following, speedMode 27. Record `cav_speed_mode` per world and refuse any world that isn't 27.
- **Windows:** start every 30 s in [t0, t0 + 540], H ∈ {30, 60} s. A window counts only if t0' + H ≤ t0 + 600.
- **Access:** the loader refuses these worlds without this registration's frozen hash, the same mechanism as the holdout block.

## Model under test

- Frozen before the run: the emulator parameters and the spread factors, each with sha256 and UTC.
- These come from the TRAIN-only diagnosis now running (merge fix plus k-fold spread refit).
- Chunk-2 TRAIN worlds may be used for fitting BEFORE the freeze. Chunk-2 validation worlds are never used for fitting.

## Criteria

Gate v2 (a)–(d) exactly as in VALIDATION_GATE_V2.md + AMENDMENT_1:
- primary H = 30; H = 60 reported;
- (d) binding in its per-horizon form: member-average speed RMSE ≤ 1.05 × deterministic IDM-normal (L0) RMSE, with L0 run in the same pass.

The (c) TTC-coverage item follows the owner's pending (A)/(B) decision:
- **(A):** kept as is;
- **(B):** moved to G-2 (then reported as a diagnostic only).

This registration is frozen only AFTER that decision, with the chosen variant written in.

Always reported:
- merges vs discretionary lane-change counts;
- CRPS vs a TRAIN climatology for delay and TTC;
- the bias vs width breakdown of every coverage miss;
- per-demand and per-penetration breakdowns.

## Protocol

- ONE evaluation pass. Raw rows are saved before aggregation.
- A FAIL means diagnosis on TRAIN only, with no second look on these worlds.
- A PASS makes native Part A's gate met. The chunk-1-era holdout second use (DEV-1) then becomes eligible under its own frozen, baseline-relative registration, subject to owner confirmation.
- The look is logged in DEV-2 (new counter: chunk-2 look #1).

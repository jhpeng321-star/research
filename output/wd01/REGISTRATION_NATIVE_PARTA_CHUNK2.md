# REGISTRATION (FROZEN): native Part A v1 — chunk-2 look #1

ID: WD-01-C2L1. Frozen 2026-09-28 by the world model session (WD-01).
This supersedes REGISTRATION_NATIVE_PARTA_CHUNK2_DRAFT.md. The frozen hash of
this file is reported to main discussion BEFORE the run. The run starts only
after main discussion acknowledges that hash.

Authority: owner decisions 2026-09-28 items 1 and 2
(tmp/zcode_review/OWNER_DECISIONS_20260928.md).

## 1. Question

Does the frozen native Part A v1 emulator pass gate v2 (amendments 1 and 2),
at primary H = 30 s, on fresh post-fix worlds (CAV speedMode 27)?

Closest prior: VAL look #8 (val8/REPORT_VAL8.md). Same model and same
thresholds. The difference is the worlds (chunk-2 CALIBRATION, post-fix,
campaign native branches instead of the chunk-1-era native roots) and
amendment 2 (TTC<3 moves out of (c)). The chunk-1-era validation roots are
retired (8 looks, WD-01-DEV-2).

## 2. Model under test (no change, no refit)

- Native Part A v1, engine kind `idm_merge` (NATIVE_PARTA_V1_STATUS.md).
- Parameters: output/wd01/part3o/results/candidate_p3o_iter2.json
  sha256 458890d9c4b0ee6ee03de145be50498d9bd0873505a4e86f2bf8a1e224ebe45d
- Spread factors: output/wd01/part3o/results/calibration_p3o.json
  sha256 c7ffa7540768fa67e75e99881eec6f801201990dea4ace90af93c202f789c22a
- Both hashes are over LF-normalised text, as frozen 2026-09-27T05:09:19Z.
- Chunk-2 data is NOT used to refit anything. The model is evaluated exactly
  as frozen. The model under test is L6c (the frozen candidate plus the frozen
  spread factors), M = 5 members.
- Known shift (disclosed, not corrected): v1 was fitted on pre-fix
  (speedMode 31) native roots. In chunk 2, native CAVs run speedMode 27.

## 3. Criteria (frozen gate files; LF-normalised sha256)

- part3f/VALIDATION_GATE_V2.md: 84b8a23df1ff654f8d602e1fd75119517c4bd7cb458dfb2191d2d088452da47f
- part3f/VALIDATION_GATE_V2_AMENDMENT_1.md: 27aeddab51a899dd1e7a7e24f073fbfe4d29efbe6c2cd3db7e992663d39d79d4
- part3f/VALIDATION_GATE_V2_AMENDMENT_2.md: 2200678280a1d236ab649d4bf44014df3b05d836fb9325e442e1453cdf5993df

**Verdict rule:** PASS if and only if L6c meets all of (a), (b), (c) and (d)
at H = 30 s:

- **(a)** Predicted collisions ≤ 1 per 1000 vehicle-steps.
- **(b)** Total lane changes per window within ±20 % of the truth (mean over
  windows, as in val8).
- **(c)** Physics-delay 10–90 % band coverage ≥ 0.80, AND the physics-delay
  ensemble CRPS is below the L0 physics-delay MAE on the same windows.
- **(d)** Member-average speed RMSE ≤ 1.05 × L0 speed RMSE at H = 30. L0 is
  deterministic IDM-normal, rolled in the same pass. The fixed 5.22 m/s form
  is reported, not binding.

**Always reported (not gate items):**

- H = 60 with the same criteria;
- raw L6 (without spread factors);
- the old gate;
- TTC<3 coverage and CRPS vs climatology, labelled "diagnostic (G-2)";
- exposure-proxy delay;
- hard-brake seconds;
- lane changes by type (merges vs discretionary);
- the bias vs width breakdown of every coverage miss;
- breakdowns by demand and by penetration.

Climatology is the chunk-1-era TRAIN climatology used in val8. Its file hash
is recorded in the run.

## 4. Data

- **Plan:** output/campaign_stage1_v2/results_plan_chunk2/WORLD_PLAN.json
  (WM-47, chunk 2), sha256 75fefc75aecc99b121556c67f3b4e248f3c3aa3b45160b736febab00d71649bf
  (raw bytes).
- **Runs:** runs/campaign_stage1_chunk2 (INDEX.json, cav_speed_mode 27).
- **Worlds.** Chunk 2 has no VALIDATION split, so the look uses all 7
  `CALIBRATION` worlds. Each has a base and a native branch recorded as
  complete in INDEX.json.

  | world | demand (veh/h) | penetration | t0 (s) |
  | --- | --- | --- | --- |
  | w8480m2600p02t300 | 2600 | 0.02 | 300 |
  | w8494m2600p05t540 | 2600 | 0.05 | 540 |
  | w8508m2700p10t300 | 2700 | 0.10 | 300 |
  | w8522m2000p20t540 | 2000 | 0.20 | 540 |
  | w8523m2600p20t540 | 2600 | 0.20 | 540 |
  | w8535m2400p40t480 | 2400 | 0.40 | 480 |
  | w8536m2400p40t360 | 2400 | 0.40 | 360 |

- **Truth:** the `__native` branch over (t0, t0 + 600]. The run refuses a world
  if its recorded CAV speed mode is not 27, or if the branch is incomplete. A
  refused world is reported and is not replaced.
- **Windows:**
  - window starts s = t0 + 30k for k = 0..18, and H ∈ {30, 60};
  - a window counts only if s + H ≤ t0 + 600;
  - this gives 7 × 19 = 133 windows per horizon.
  - The initial state and the arrivals (including entrance queues, contract
    §8.14) come from the native branch's logs, exactly as the p3/val8 loaders
    take them from native roots.
- **HOLDOUT worlds are never opened.** Chunk-2 TRAIN worlds may be used ONLY
  for a pipeline dry run before the look. That dry run checks that the loader
  and the conversion are correct (e.g. that the truth vehicle-step counts
  reconcile with the logs). It fits nothing and changes no threshold.
- **Access control:** the loader refuses the 7 CALIBRATION worlds unless it
  is given this file and its LF-normalised sha256 matches the frozen value.
  This is the same mechanism as holdout_block.py.

## 5. Protocol

- Before the look:
  - the evaluation code is committed;
  - its commit id and the sha256 of each evaluation script are written to a
    run sidecar.
- ONE evaluation pass. Raw rows (L6, L0) are saved before aggregation, and
  aggregation re-rolls nothing.
- No re-run after any number is seen. A crash before any output row is
  written may be restarted once, and that restart is disclosed. A crash
  after rows exist is reported as is.
- **FAIL:** diagnosis continues on TRAIN only. There is no second look on
  these 7 worlds. A structural v2 (DESIGN_PARTA_STRUCTURAL.md) would need
  fresh worlds and a new registration.
- **PASS:** native Part A gate v2 is met. The DEV-1 holdout second use then
  becomes eligible, under its own frozen, baseline-relative registration and
  only after main discussion relays the owner's confirmation.
- **Records:** the look is logged in WD-01-DEV-2 as chunk-2 look #1.

## 6. Interaction with other uses of these worlds

- The chunk-2 CALIBRATION worlds are also Part B V0's CALIBRATION split
  (early stopping and model selection for V). That use does not touch Part A,
  so it does not contaminate this look.
- Planner-side analyses of these worlds do not feed Part A.
- Part A has never been fitted or tuned on any chunk-2 world.

## 7. Budget

Zero simulation, CPU torch rollout only: 7 worlds × 19 starts × 2 horizons ×
(5 members + L0). Expected under 1 h on CPU, run as one ZCode task.

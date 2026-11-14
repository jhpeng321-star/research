# REGISTRATION (analysis, zero simulation): Part A v2 general, evaluation (a), model error from the TRUE initial state

- **ID:** WD-01-PA2G-A.
- **Status:** registered 2026-09-29 before any run. Authority: main discussion
  under the owner's delegation (Q5).
- **Type:** ANALYSIS on existing logs. No SUMO/TraCI. TRAIN only.

## Question

How large is the MODEL error of the generic textbook rules (Part A v2
general: Treiber IDM + MOBIL + the merge rule, one parameter set, no vType)
against the campaign SUMO truth, when started from the TRUE logged state
restricted to the recon zone? And at which horizon do untracked arrivals start
to dominate?

## Data

- Existing logged native branches of chunk-2 and chunk-3 TRAIN worlds:
  `runs/campaign_stage1_chunk2|chunk3/<world>__native/run/physics_steps.jsonl`.
- Splits come from their WORLD_PLANs.
- CAL, VALIDATION and HOLDOUT worlds are never opened (asserted in code).

## Protocol

- **Windows:** start every 30 s over (t0, t0 + 540]; H ∈ {10, 30, 60, 120} s.
- **Initial state:** the logged vehicles inside the recon zone at the window
  start, with the zone extent from the reconstructor's schema (default: the
  RSU merge + ramp zone as defined there).
- **CAVs:** native in these branches. The native command path is used.
- **Metrics per H**, over the tracked vehicles:
  - trajectory RMSE (s, v);
  - lane-change counts by type (merge / discretionary) vs truth;
  - window delay error;
  - TTC<3 exposure error;
  - hard-brake error;
  - predicted collisions per 1000 vehicle-steps.
- **Missing-arrivals curve:** the truth metric over (i) the tracked vehicles
  only vs (ii) all vehicles, per H.
- **Counts:** `n_free_road_assumed` per window.
- **Comparators:** the v1 p3o engine (from the same true state and with the
  logged arrivals; reference only) and constant-speed persistence.
- **Breakdowns:** per penetration and per demand.

## Rules

- **No tuning.** Parameters are fixed at the Treiber defaults. Errors are
  reported as model error.
- **Sensitivity.** A T/a sensitivity is reported as a separate, labelled
  sweep: T ∈ {1.0, 1.5, 2.0}, a ∈ {0.7, 1.0, 1.5}. It is descriptive only,
  and no parameter is chosen from it.
- **Records.** Raw rows are saved before aggregation. The outcome is appended
  to the ledger draft as an analysis.

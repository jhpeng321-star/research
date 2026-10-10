# REGISTRATION: WD-01-PA2G-CNT1: Part A v2g plus entrance-counter arrivals (SIDE RESULT; sensitivity, "requires counter hardware")

**Registered:** 2026-09-30 by the world model session, BEFORE execution.

**Authority:** owner decision, 2026-09-30 about 11:30 (via main discussion).

**Type:** TRAIN-only, zero-SUMO SENSITIVITY analysis.

- The CAV-strict / tracked-scope Part A v2 general remains the HEADLINE.
- This result does not change it, the frozen C4L1, or the nominal
  parameters.

## 1. Question

Does injecting arrivals at the network entrances substantially reduce Part
A's long-horizon delay error? The arrivals come at the rate an entrance loop
counter observed over the last W s. The comparison is against the TRUE
all-vehicle delay (tracked plus arriving vehicles).

**Motivation:** PA2G-A2 shows a long-horizon negative bias from the tracked
scope. Untracked arrivals are about 58 % of the true delay at H120.

**Closest prior:** WD-01-PA2G-A2 and LC1. The engine, truth, worlds, windows
and metrics code are the same. The difference is the added arrivals injector
and the all-vehicle delay target.

## 2. Input (observable-only)

- **Counts:**
  - one entrance counter at the mainline entry boundary (edge 1191554636,
    network start);
  - one at the on-ramp entry boundary (the ramp route's first edge; the
    worker determines it from the network and records it).
- **What is counted:** vehicles entering the network. Truth proxy: the
  `departed_ids` of the physics rows, classified by entrance from the route
  or edge of the vehicle's first physical row. This is the same counter
  channel as recon COUNTER and the Part B detector variant.
- **Window:** only counts in (w − W, w] are used, where w is the window start
  (prediction time). NO counts after w; a test enforces this.
- **Nothing else:** no vehicle type, no CAV flag, no penetration, no future
  schedule. Injected vehicles are generic HDVs with the nominal parameters.

## 3. Mechanism (all values a priori; NOT tuned)

- **Rate:** λ_e = count_e(w − W, w] / W per entrance e, with **W = 60 s**. It
  is held constant over (w, w + H] ("carried forward").
- **Insertion schedule:** deterministic and evenly spaced. For entrance e the
  k-th injected vehicle is due at w + (k − 0.5)/λ_e for k = 1, 2, …, while its
  due time ≤ w + H.
- **Placement:**
  - at s = 0 of the entrance edge;
  - mainline arrivals alternate lanes 0/1 round-robin, starting with lane 1
    (right);
  - ramp arrivals use its single lane.
- **Insertion speed:** v_ins = min(lane speed limit, speed of the nearest
  leader in the target lane).
- **Insertion condition:** the IDM safe gap
  gap ≥ s0 + v_ins · T (s0 = 2 m, T = 1.5 s, the nominal IDM values).
  - Otherwise the vehicle waits in a virtual entrance queue and is retried
    each step (FIFO).
  - Delay accrues while queued, at the free-flow reference, exactly as the
    truth accounts entrance queues. If the truth delay excludes pre-insertion
    waiting, the worker must use the same definition on both sides and state
    which.
- **Dynamics:** injected vehicles follow the nominal IDM + MOBIL + merge
  engine. They are included in collisions and safety metrics, and they are
  labelled separately in raw rows.

## 4. Data, comparators and metrics

- **Worlds and windows:** the 32 PA2G-A2 TRAIN worlds of chunk 4
  (`lc1_a2_worlds32.json`), with the same windows and H ∈ {10, 30, 60, 120}
  s. The post-LC1 counting convention applies.
  - TRAIN only; the split is checked before every open.
  - NO chunk-4 CALIBRATION or HOLDOUT file is opened (C4L1 uses the CAL).
- **Models, paired on the SAME windows:**
  - (i) **v2g-CNT**, this variant;
  - (ii) **v2g** (tracked only, no arrivals; the A2 nominal);
  - (iii) const-speed;
  - (iv) the p3o v1 replica. It receives the LOGGED true arrivals, so it is
    labelled "oracle-arrival reference, not deployable".
- **Metrics per model and H:**
  - **PRIMARY:** the all-vehicle window delay error (pred − truth), signed
    and MAE. The truth is `delay_true_all`: every vehicle in the modelled
    road during the window, including arrivals.
  - the tracked-only delay error, for continuity with A2;
  - s_RMSE and v_RMSE on the tracked vehicles, which are unchanged by the
    definition. Also a separate injected-vehicle s_RMSE vs the matched truth
    arrivals, reported descriptively by order of entry per entrance;
  - collisions per 1000 veh-steps;
  - TTC<3 and hard-brake errors;
  - the injected vs true arrival counts per window.
- **Uncertainty:** a world-bootstrap 90 % CI on each v2g-CNT − v2g
  difference.

## 5. Decision rule (fixed BEFORE running)

v2g-CNT is **WORTH IT** as a counter-hardware side option if ALL of:

- (1) the all-vehicle delay MAE at H60 AND at H120 is each ≥ 25 % lower than
  v2g (no arrivals), with the world-bootstrap 90 % CI of the relative change
  excluding 0;
- (2) the all-vehicle delay MAE at H10 and at H30 is not worse than v2g by
  more than 5 %;
- (3) the tracked-vehicle s_RMSE at H10 and at H30 is not worse than v2g by
  more than 5 %;
- (4) collisions ≤ v2g + 0.1 per 1000 veh-steps.

**Otherwise:** PARTIAL if (1) holds for only one of H60/H120 and (2)–(4)
hold; NOT WORTH IT otherwise.

The report always states the label **"sensitivity; requires counter
hardware; does not change the CAV-strict headline"**.

## 6. Budget

- Zero SUMO.
- CPU: 1 v2g-CNT run over 32 worlds. The comparators come from A2 / post-LC1
  rows where they are bit-identical; otherwise they are re-rolled once. A2
  took 1377 s at 4 workers, so this is ≤ 1 h at ≤ 3 workers.
- One ZCode task (Flash).
- Priority: after C4L1, LC2 and the wall-B path.

## 7. Outputs

- partA_v2g/REPORT_PA2G_CNT1.md: the mechanism spec, the tables with CIs,
  the verdict under §5, and the label.
- Raw rows are saved before aggregation.
- LEDGER_DRAFT_PA2G_CNT1.json in the experiment_history ENTRY schema.
- Tests:
  - no count after w is read;
  - the insertion schedule and the safe-gap rule;
  - the nominal path is byte-identical with the injector OFF;
  - TRAIN-only guard.

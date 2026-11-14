# WM-N1: a fully neural world model (design, v1 draft)

World model session, 2026-09-30.

**Owner decisions:**

- build a fully neural world model (WM-N1) on a rented GPU;
- ADD-ON: report pure-neural, tree Part B and a COMBINED model side by side.

**Scope:**

- PREPARATION ONLY: design, registration draft, data pipeline, code and tests.
- No training, no SUMO.
- The current hybrid (Part A v2g plus Part B v0.4a/b) stays the working
  version and is the BASELINE TO BEAT. Nothing here touches v0.4a, CNT1 or
  v0.4b.

## 0. What the planner needs, and what WM-N1 replaces

The planner asks: "if I apply action plan P (per-CAV native / damping / PI /
wall member) from now, what happens over the next τ s?" The quantity it
decides on is the CONTRAST between plans:

ΔV(P) = V(s, P) − V(s, all-native)

for delay, with safety constraints alongside.

- **Today:** Part A (rules) predicts the next ≤ 30 s per vehicle, and Part B
  (trees) predicts the remaining outcome per τ.
- **WM-N1:** one network does both, learned from the campaign logs, and is
  trained explicitly on paired branches so that ΔV is accurate.

## 1. Inputs (deployment-honest)

| group | content | source at deployment |
| --- | --- | --- |
| vehicle tokens | per vehicle at t0: s_down (m), lane_phys (0 main_left, 1 main_right, 2 accel; one-hot), v, a, nominal length, **is_cav** | recon sample (training: true state) |
| CAV plan tokens | per CAV, the action over the horizon in 30 s segments: {native, damping, PI, wall-member, unknown}; wall id, wall speed, planned hold | planner (it chooses them) |
| wall tokens | wall id, phase, speed, time in phase, members, planned hold remaining (as in partB_v0data/wall_tokens_v04.py) | planner |
| counter history | entrance counts per 10 s bin over the last 300 s, per entrance (mainline, ramp) | loop counters (COUNTER channel; owner decision) |
| static | lane geometry masks, speed limits, merge-zone extent | map |
| context | τ-query set; time since plan start | planner |

**Excluded:**

- hidden driver type or vType;
- true penetration;
- pre_fix, speed_mode, world strings, seed, t0.

The one exception is that is_cav IS allowed: CAVs identify themselves to the
RSU.

**Training uses the TRUE state.** Evaluation on recon samples is a later,
separately registered step (a C4L2-like look).

## 2. Architecture (one network; about 2–5 M parameters per member)

1. **Encoder.**
   - Each vehicle token is embedded by an MLP.
   - L = 4 layers of lane-aware relative attention. Each vehicle attends to
     neighbours within ±150 m on its own and adjacent lanes, with
     relative-geometry features (Δs, Δlane, Δv) as attention biases, like a
     GNN over a radius graph.
   - Plan tokens attend into their CAV's token, and wall tokens attend to
     their members.
   - A counter-history encoder (1-D conv over 30 bins × 2 entrances) feeds a
     global token.
2. **Stochastic step decoder** (dt = 0.5 s, matching the physics logs and
   Part A):
   - per vehicle, a K = 3 Gaussian mixture over the longitudinal
     acceleration;
   - a 3-way categorical {stay, left, right}, where "right" from lane 2 is
     the merge;
   - autoregressive rollout by re-encoding every step. At inference the
     encoder can be re-run every 2 steps for speed (an ablation).
3. **Arrival head:**
   - per entrance, a negative-binomial count for the next dt from the counter
     history and the rollout state;
   - an insertion speed distribution;
   - inserted vehicles enter as new tokens (generic, is_cav = 0 unless the
     planner roster says otherwise).
4. **Outcome heads** (they replace Part B):
   - from the pooled encoder state plus the plan and wall tokens, for τ ∈ {30,
     60, 120, 300, 570};
   - quantiles (19 levels) of the total delay (all vehicles, including
     arrivals), TTC<3 exposure and hard-brake seconds;
   - a collision logit.
   - They are evaluated in ONE forward pass, with no rollout needed. That
     makes them fast for the planner, and the rollout is used for
     short-horizon checks.
5. **Physical-validity layer** (inside the rollout):
   - the acceleration is clipped to [−9, +3] m/s²;
   - a lane change is accepted only if the target gap ≥ 0 after the move;
   - a speed projection keeps the rear bumper gap ≥ 0.
   - Every projection is COUNTED ("interventions"), and the count is
     reported. A collision is still possible and counted.

## 3. Training

- **Data units:**
  - (a) STEP samples: every 0.5 s tick of every run, for teacher-forced
    one-step NLL of acceleration, lane change and arrivals;
  - (b) ROLLOUT samples: 20-step (10 s) windows with scheduled sampling, the
    pathwise loss on positions and speeds;
  - (c) OUTCOME samples: decision times every 30 s, with the Part B targets
    per τ;
  - (d) PAIRED samples: the same world and t0, with different arms (base
    damping vs restored native / PI / fixed / switching / wall).
- **Losses:**
  - NLL for the step heads;
  - pinball over 19 quantiles for the outcomes;
  - BCE for collisions;
  - **CONTRAST loss** on paired samples: the pinball loss of the paired
    quantile difference, with shared quantile levels (comonotone pairing) for
    ΔV = V(P_a) − V(P_b) against the realised ΔY. The weight is λ_c = 1.
- **Uncertainty:**
  - aleatoric from the sampling and quantile heads;
  - epistemic from a deep ensemble of **5** members (matching the recon
    choice), trained with different seeds and world-bootstrap resampling.
  - Calibration is per-τ conformal shifts fitted on CAL ONLY.
- **Selection:** early stopping on CAL loss, world-grouped. CAL is never used
  for gradients.

## 4. Data, splits and the fresh test set

- **Sources:**
  - chunks 1–4 base and branch runs (about 2,700 runs, about 120 GB raw
    physics JSONL);
  - J3 truearms (280);
  - J4 wall branches (176, when available).
  - Actions come ONLY via partB_v0data/actions_v04.py (switch_schedule plus
    suffix_commands with kind == "command"). Never dispatcher_rows or
    insertions.jsonl.
  - Chunks 1–3 "native"/"pi" arms are labelled switching (WM-72).
- **Splits:**
  - TRAIN = every chunk's TRAIN worlds;
  - CAL = the chunk 2–4 CAL worlds (early stopping, calibration, blend
    weights, variant choice);
  - HOLDOUT is SEALED (guard plus exclusions).
  - Disclosed: the chunk-4 CAL is shared with C4L1 and Part B v0.4a. The
    chunk-2/3 CAL was read for reports.
- **FRESH TEST SET (needed; the chunk-4 CAL is spent):** I propose a new
  registered **chunk 5 TEST**.
  - 30 worlds, fresh seeds from the seeds registry with an exposure audit.
    Stratified 3 demand × 5 penetration × 2 replicates.
  - Per world: base (damping) + native_base + restored native, PI, 2 fixed
    and 2 switching arms, plus 4 wall arms (Option-A/B member rule).
  - Budget: about 30 × (900 + 10 × 600) ≈ 207 k sim-s.
  - It runs on the rented box, like J1. Outcomes are sealed until the final
    registered evaluation.
  - The alternative (not recommended) is to reuse the chunk-2/3 CAL as test.
    It is weaker, because reports have been read on it.
- **Format:**
  - laptop-side featurization (like the recon) into memmap tensors;
  - vehicle-step table float16 [N_steps, K_max = 128 vehicles, F = 16] with
    masks, plus plan, wall and counter tensors, plus outcome targets.
  - Estimate: about 4.9 M ticks × 128 × 16 × 2 B ≈ 20 GB uncompressed,
    about 6–8 GB zstd. It is uploaded compressed, with SHA256SUMS.
  - Featurization is CPU-bound, about 4–6 h at 4 workers on the laptop. It
    can also run on the rented CPU box.

## 5. Combination layer (owner add-on): three variants side by side

- **N: pure neural** (the headline "fully neural").
- **T: tree Part B** (v0.4a, then v0.4b when walls exist), as registered.
- **C: combined.** Its components:
  - **(a) STACKING (recommended):**
    - per τ and target, a linear pool of the two predictive CDFs:
      F_C = w_τ·F_N + (1 − w_τ)·F_T, with w_τ ∈ [0, 1];
    - w_τ is fitted by minimising CRPS on CAL with world-grouped 2-fold
      cross-fitting (fit on half A, evaluate on B, and vice versa);
    - then per-τ conformal recalibration on CAL;
    - NEVER fitted on TEST.
  - **(c) DISAGREEMENT SIGNAL (recommended, with (a)):**
    D_τ = CRPS-distance(F_N, F_T) / scale_τ.
    - It is exposed to the planner as an uncertainty/abstain input.
    - Reported: the Spearman correlation of D with the realised |error| on
      CAL and TEST, and its abstain curve.
  - **(b) RESIDUAL (ablation):** the neural outcome head gets the tree
    prediction as an input feature (a prior) and learns a correction.
    - Leakage rule: the trees are fitted on TRAIN only. TRAIN rows receive
      world-grouped **5-fold OUT-OF-FOLD** tree predictions; CAL and TEST
      rows receive the full-TRAIN tree prediction.
  - **Step-dynamics prior (ablation):** the IDM/MOBIL acceleration from
    partA_v2g as an input to the step decoder, which learns a residual.
- **Which variant the planner uses, chosen on CAL BEFORE the test:**
  - among {N, T, C(a)}, the lowest CAL mean CRPS of ΔV over the gating
    contrasts and τ ∈ {60, 120, 300};
  - subject to overall 10–90 % coverage in [0.75, 0.85];
  - a tie within 2 % goes to the simpler model (T < C < N).
  - The choice is frozen in a sidecar before TEST is opened.

## 6. Evaluation vs the hybrid (TEST = chunk 5; the same windows for all)

- **Short horizon** (H10/30/60/120, true-state start): s_RMSE and v_RMSE as a
  % of the true distance and speed, and the delay % error. The neural model
  uses its rollout; the hybrid uses Part A v2g. Also reported: the
  all-vehicle delay with arrivals, and CNT1 for context.
- **Outcomes** per τ: CRPS, 10–90 % coverage and PIT, for each of N, T and C
  vs climatology.
- **Contrasts:** damping−native, PI−native and wall−native (wall only if the
  J4 and chunk-5 wall arms exist):
  - sign accuracy (|ΔY| > the TRAIN median) with a world-bootstrap CI;
  - ΔV CRPS vs the zero-contrast and TRAIN-cell-mean baselines.
- **Validity:** collisions per 1000 veh-steps, interventions per 1000
  veh-steps, and the arrival-count error.
- **Planner relevance:** the PL-NOISE1 error budget (registered by the
  planner, NOT STARTED yet) is the target. When it exists, report whether each
  variant's ΔV error is within the budget.
- **Decision rule** (in the registration, fixed before training): "WM-N1
  (planner variant) beats the hybrid" iff, on TEST, ALL of:
  - the ΔV CRPS is lower for ≥ 2 of the gating contrasts at ≥ 2 of the τ
    {60, 120, 300}, with the world-bootstrap CI of the difference excluding
    0;
  - the outcome CRPS is not worse than the hybrid by more than 5 % at any τ;
  - the H10/H30 s_RMSE is not worse than Part A v2g by more than 10 %;
  - collisions ≤ hybrid + 0.1/1000.
  - Otherwise the hybrid stays the working version.

## 7. Compute and bundle

- **GPU:** one 24–48 GB card (L4 / A10G / RTX 4090 / A6000) is enough. One
  member has 2–5 M parameters with batches of scenes of 128 vehicles.
- **Estimate** (to be replaced by the timing gate):
  - one-step training at about 20–40 min per epoch per member on an A10G;
  - 15 epochs × 5 members ≈ 25–50 GPU-h;
  - rollout fine-tuning +30 %;
  - evaluation about 2 GPU-h.
- **Timing gate first:** a 1-h gate on the rented GPU (throughput, memory, a
  loss decrease on a 2 % TRAIN subset). It extrapolates the full cost, and
  the owner approves before the full run.
- **Shared bundle:** the same rented GPU box as the recon. Coordinate with
  the reconstructor's GPU bundle:
  - a shared conda env spec (torch/CUDA version pinned to the recon's);
  - a shared data-upload path;
  - runs are sequential by default, or concurrent if VRAM allows (two
    processes and MPS off).

## 8. Readiness plan (preparation only)

1. **Now (WM):** this design, the registration draft and the plain-language
   summary.
2. **Sonnet:**
   - the data pipeline (featurizer to memmap, split guard, action labels,
     OOF-tree export hook, integrity sums);
   - the model;
   - the losses (including the contrast loss);
   - the rollout with the validity layer;
   - the combination layer;
   - the eval harness;
   - tests on synthetic and TRAIN micro-subsets;
   - the timing-gate script.
   - No training beyond CPU smoke tests (≤ 1 minute, a tiny model).
3. **The chunk-5 TEST registration** is a separate registration for the
   owner and a reconstructor allocation.
4. **Laptop-side featurization** of TRAIN + CAL: a separate allocation, when
   CPU allows.

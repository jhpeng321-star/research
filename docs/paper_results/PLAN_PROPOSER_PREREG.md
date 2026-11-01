# PRE-REGISTRATION: learned search (plan proposer) for arm B — activation rule

Status: DRAFT for MAIN registration. Register it before any outcome related to the
proposer. **Activation is NEVER automatic.** The current launch is unchanged
(design freeze 2026-11-02 01:38). Until activation, B runs exactly the qualified
warm fast-path MPPI search: no proposer seeds, no warm start, no learned-search code
path. The only change allowed before activation is logging-only receipts (8).

1. **Object.** `tools/coopmwm/plan_proposer.py`, branch c4/plan-proposer.
   - **Model.** A permutation-equivariant set network.
     - Inputs: the observable member/neighbour rows (`vid, s_m, v_mps, lane,
       is_cav[, length_m]`), the regime and the group size. It never sees
       world/seed/demand/share labels, clock, driver parameters or SUMO truth.
     - Output: a defensive mixture, 0.9 x 4 Gaussian joint perturbations plus
       0.1 x the search's own initial N(0, Sigma_0) with uniform durations. It is
       expressed in mppi_search's canonical space (members x 10 knots over 300 s,
       relative to the policy trajectory) and carries a duration categorical.
   - **Two flag-only uses** of `mppi_search.search`. With both off (the default)
     the output is byte-identical, which is tested.
     - (i) `extra_seeds`: 4 proposer seeds. They take population slots.
     - (ii) `warm_start`: the moment-matched proposer distribution replaces the
       initial distribution. Every iteration's sampled population keeps:
       - a **generic share of 0.2** drawn from the plain policy-centred
         N(0, Sigma_0) with uniform durations (`WARM_GENERIC_SHARE`);
       - a **novelty share of 0.1** (`WARM_NOVELTY_SHARE`). From 4x as many
         generic draws, keep those with the lowest proposer/generic density ratio.
         This is rank-based, with no threshold, so every search deliberately
         explores moves the proposer considers unlikely.
     - In every mode the WM scores every candidate. Screen K4/H300 and final
       K8/H_final are never shortened.
2. **Arms.** The proposer is for arm B only.
   - Arm R never uses it. R keeps generic unscored perturbations, matched in
     count, regime, size and duration.
   - Arm A has no search.
   - Training data is same-arm and same-seed, like the library and covariance.
3. **Training data.**
   - Sources:
     - arm-B TRAIN-world (986000..986071) searches from updates before the
       boundary;
     - **all** searches with a full screened pool, accepted or not;
     - top-k-only service receipts, used only when no pool exists.
   - Never used: dev, check, final, sealed or branch-panel outcomes.
   - Weights:
     - Full pool: each search contributes one unit of weight over its own top-E
       screened elites, with E = ceil(N/4). Elites are ranked by utility against
       the search's own comparators: min(saving vs 'policy continues', saving vs
       MAPPO). The weight is the utility above the next-ranked candidate, so it is
       >= 0 and still rank-informative when nothing beat the comparators.
     - Top-k only: weight is proportional to positive saving vs the policy.
     - A SUMO-positive executed plan adds +1.
     - SUMO-negative executed plans are never targets.
4. **Frozen configuration** (`plan_proposer.REGISTERED` at the registered commit).
   Any change needs a new amendment before training.
   - Model: M=4, d=64, 2 layers, defensive 0.1.
   - Mixture-entropy floor 0.5 log M, weight 1.
   - Training: at most 1500 AdamW steps, lr 3e-3, wd 1e-4, seed 20261102, 1 thread,
     deterministic.
   - Early stopping on an inner world split of the TRAIN worlds only.
   - Seed and warm-start settings: 4 seeds, generic share 0.2, novelty share 0.1,
     novelty oversample 4.
5. **Offline check (required, not sufficient).** Held-out searches only, using a
   world-grouped split (25% of worlds by SHA256 rank of (seed, world)). The held-out
   set needs >= 8 worlds and >= 40 searches with elites. PASS requires all three,
   each judged on a world-cluster bootstrap (2000) 95% CI against the search's own
   initial distribution:
   - (a) elite log-likelihood gain: CI lower bound > 0;
   - (b) nearest-elite RMS distance change (16 samples): CI upper bound < 0;
   - (c) the same as (b) against the zero 'centre' seed: CI upper bound < 0.
6. **Evaluations-saved estimate (reported, not a gate).**
   - Run `learned_search_eval` on held-out full-pool searches as a surrogate replay
     of each search's own WM screen pool.
   - Modes: plain (3 iterations) vs +seeds vs warm vs both vs warm with 1 and 2
     iterations. The horizons are the same in every mode.
   - Reported: evaluations to reach the plain search's final elite quality,
     quality vs evaluations, and elite coverage, which checks that the learned
     search still finds the recorded elites.
   - Any proposal to cut iterations or population requires its own owner-approved
     amendment that cites these numbers.
7. **Schedule (no gate shopping).**
   - The check runs only at TRAIN-check boundaries (every 4 updates).
   - The first attempt needs **>= 160 full-pool B searches (>= 40 held out) from
     >= 32 TRAIN worlds (>= 8 held out)**. This is based on the synthetic curve,
     where full-pool receipts with 30% unaccepted searches pass from 96 training
     searches (`PLAN_PROPOSER_SYNTH_CURVE.json`, `LEARNED_SEARCH_SYNTH.json`).
   - At most one attempt per boundary, last attempt at u40, frozen configuration.
   - Every attempt is written to `PROPOSER_CHECK_u<k>.json` with checkpoint and
     dataset SHA256s, split worlds and metrics.
   - `learned_search_report` may run at any time as information. It does not
     count as an attempt.
8. **Logging before activation (optional, needs MAIN approval).**
   - Mechanism: wrap B's `evaluate` in `RecordingEvaluator` and call
     `log_receipt(root, record, result, rec)`. This writes
     `<root>/proposer_receipts/<state_id>.json`.
   - The search output is byte-identical (tested). There is no extra WM, GPU or
     SUMO cost. Disk use is about 0.1-0.3 MB per search.
   - Without this logging, real full-pool data does not exist.
9. **Activation requires BOTH** (a) a PASS in (5) under (7) **and** (b) **explicit
   owner approval recorded by MAIN** (owner statement, time, approved mode and
   checkpoint SHA).
   - Activation is an engineering amendment at an update boundary:
     `ACTIVATE_PLAN_PROPOSER_u<k>.json`, with code commit, checkpoint SHA, the
     approved mode (seeds | warm | both) and the owner-approval reference.
   - No script, check or agent may activate it on its own.
   - Without (b), a PASS is only reported.
10. **Refresh.** A retrained proposer replaces the active one only if it passes
    (5) on a fresh split and the owner approval covers refreshes. Every attempt is
    recorded.
11. **Not a selection.** The final checkpoint rule (A/B u48, R u24), the sealed
    tests and the pause rules are unchanged.
    - Diagnoses and per-update reports break executed plans and SUMO advantage
      down by origin, and include the novelty metrics (`plan_novelty`,
      `novelty_report`):
      - distance to the nearest proposer mode;
      - distance to the nearest library entry;
      - the post-hoc family label, including 'new'.
12. **Disclosure.**
    - After activation, B is reported as "B + learned search (<mode>) from u<k+1>".
    - No SUMO-benefit claim is made without a separately registered comparison.
    - Compute: CPU only (about 5 ms per group of inference; training takes
      minutes). There is no sim-s charge.

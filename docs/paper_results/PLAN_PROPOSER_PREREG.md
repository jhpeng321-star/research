# PRE-REGISTRATION: activation of the learned plan proposer (arm B only)

Status: prospective registration before any B/R outcome; MAIN02:53 owner-approval gate applies.
It does not change the current launch (design freeze 2026-11-02 01:38): until
activation, B runs exactly the qualified warm fast-path MPPI search.

1. Object. `tools/coopmwm/plan_proposer.py` (branch c4/plan-proposer). It is a
   permutation-equivariant set network over the observable member/neighbour rows
   (`vid, s_m, v_mps, lane, is_cav[, length_m]`), the regime label and the group
   size. It never sees world/seed/demand/share labels, clock time, driver
   parameters or SUMO truth. Its output is a defensive mixture: 0.9 x 4 Gaussian
   joint perturbations plus 0.1 x the search's own initial N(0, Sigma_0), uniform
   over durations. It lives in the mppi_search canonical knot space (members x 10
   knots over [0, 300] s, relative to the policy trajectory), with a categorical
   over the registered durations.
2. Arms. Only arm B may use it. Arm R never uses it: R keeps generic unscored
   perturbations around the same centre, matched in count, regime, size and
   duration. Arm A has no search. Both the proposer and its training data are
   same-arm (B) and same-seed only, like the plan library and covariance.
3. Training data. Training uses only arm-B TRAIN-world search receipts from updates
   before the activation boundary. A receipt is the plan-service request rows plus
    the search result, with the full screened pool when `RecordingEvaluator`
    logging is on, and the `search_learning` exact paired SUMO grades of executed
   plans. Training never reads dev, check, sealed or branch-panel outcomes.
   - Weights: each search's candidates with a positive WM-predicted saving vs
     'policy continues' share one unit of weight, proportional to the saving.
   - A SUMO-positive executed plan adds +1 (`sumo_bonus`).
   - A SUMO-losing executed plan is never a target.
4. Frozen configuration. `plan_proposer.REGISTERED` at the registered commit:
   - Model: M=4, d=64, 2 layers, defensive 0.1.
   - Training: <=1500 AdamW steps, lr 3e-3, wd 1e-4, seed 20261102, single thread,
     deterministic algorithms.
   - Early stopping: on an inner world-grouped split of the TRAIN worlds only.
   - Changing any of these needs a new amendment before training.
5. Offline check. PASS plus explicit OWNER approval recorded by MAIN in REGISTRATION.md are BOTH required for activation. It uses held-out searches only,
   with a world-grouped split: 25% of worlds chosen by the SHA256 rank of
   (seed, world id). The held-out set must have >= 8 worlds and >= 40 searches with
   >= 1 elite. Elites are the search's own top-k candidates with a positive final
   (K8) predicted saving vs the policy, or else the top quarter of the screened
   pool. PASS requires both of the following, against the search's own initial
   distribution:
   - (a) Mean elite log-likelihood gain (proposer minus initial; duration term
     included): the world-cluster bootstrap (2000) 95% CI lower bound is > 0.
   - (b) Mean nearest-elite RMS distance change, with 16 samples from each: the
     95% CI upper bound is < 0.
   - (c) The same distance change against the search's existing zero 'centre'
     seed: the 95% CI upper bound is < 0.
6. Schedule (no gate shopping):
   - The check runs only at TRAIN-check boundaries (every 4 updates). The first
     attempt is at the first such boundary where the data minimum (5) holds AND
     >= 200 B searches with elites from >= 32 distinct TRAIN worlds exist.
   - That minimum comes from the synthetic curve
     (`docs/paper_results/PLAN_PROPOSER_SYNTH_CURVE.json`). It passes from about
     96 searches using top-k receipts only, and from about 48 with full pools. The
     log-likelihood gain is stable from 192 searches on.
   - At most one attempt per boundary. The last attempt is at u40. There is no
     configuration change between attempts, and the data only grows.
   - Every attempt (PASS or FAIL) is written as `PROPOSER_CHECK_u<k>.json`, with
     the checkpoint .pt/.json SHA256, the dataset SHA, the split worlds and the
     metrics.
7. Activation is NEVER automatic; explicit OWNER approval by MAIN is required in addition to PASS. Activation is an engineering amendment at an update boundary
   (`ACTIVATE_PLAN_PROPOSER_u<k>.json`, with code commit and checkpoint SHAs).
   From update k+1, each B search receives `proposal_seeds(record, group,
   policy_traj, n=4, ...)` as `extra_seeds`. These are iteration-0 seeds that take
   population slots. The following are all unchanged: population, iterations,
   screen and final K/H, top-k, Thompson choice, trust beta, the library and
   covariance learning, the PPO-own-action / AWR update, and the WM cost per
   search. The WM scores every seed, and SUMO alone grades executed plans.
8. Refresh. At later TRAIN-check boundaries the proposer may be retrained on the
   accumulated B receipts with the frozen configuration. A retrained proposer
   replaces the active one only if it passes the same check (5) on a fresh
   world-grouped split. Otherwise the active version stays. Every refresh attempt
   is recorded.
9. Not a selection. Activation and refresh never use dev, sealed or branch
   outcomes. The final checkpoint rule (A/B u48, R u24) and the sealed tests are
   unchanged. The pause rules (01:38) are unchanged. On a B pause, the diagnosis
   reports executed plans and SUMO advantage by origin, with 'proposer' shown
   separately.
10. Disclosure. After activation, B is reported as "B + learned proposer seeds
    from u<k+1>". The report also gives:
    - the activation update;
    - the share of top-k and executed plans by origin;
    - the per-update offline-check metrics.

    No claim is made about the proposer's effect on SUMO outcomes without a
    separately registered comparison.
11. Compute. Proposer inference is about 5 ms on CPU per group (4 seeds, 6
    members, 1 thread). Training is CPU-only at update boundaries, takes minutes,
    and uses no GPU and no SUMO. Neither touches the sim-s cap.

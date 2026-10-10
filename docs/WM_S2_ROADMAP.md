# WM-S2 roadmap: weekly plan (owner-approved framing, 2026-10-06)

## Thesis
A world model that separates what we control (CAV controller code, known exactly) from what we must learn
(how each human driver behaves, inferred from sparse RSU/CAV observations) can (1) evaluate controllers it has
never seen and (2) reveal which combinations of ordinary human drivers make a control action fragile.

## Contributions
- C1 (headline) Zero-shot evaluation of unseen controllers. Controller = known intervention run as code; only the
  HDV mechanism is learned. Decisive test: leave-one-controller-out, structured WM vs black-box WM (controller as an
  input feature, e.g. A11-style effect head / neural dynamics with controller id).
- C2 (enabling) Per-driver identification from sparse observations: attention encoder over each HDV's accumulated
  trajectory -> per-driver IDM/MOBIL/merge parameters + uncertainty, used inside per-step physics (WM-S2).
  Validated against SUMO ground-truth driver parameters vs observation time and CAV penetration; calibration.
- C3 (distinctive) Fragility of a control action: particle rollouts with common random numbers across controllers,
  per-step hazard h_t = P(breakdown at t | none before), P(breakdown by T) = 1 - prod(1 - h_t), splitting for rare
  events; jointly-critical driver pairs via the Shapley interaction index
  I_ij = P(b|i,j vary) - P(b|i) - P(b|j) + P(b|none). Decision: best expected delay subject to a fragility limit.
  Validated against true breakdown probabilities from repeated SUMO runs from the same snapshot, and causally by
  re-running SUMO with the flagged driver pair's parameters set.
Kept: per-step open-loop model, traffic-level evaluation, decision test, closed loop. Dropped/deferred: delay head,
paired-difference loss, ensemble, macro hand-off, standalone FORM.

## Key numbers to report every week (same table each week)
car-level |ds| median at 10/30/120/300 s; traffic-level section speed / queue / delay error; paired PI-native
difference sign agreement + rank correlation; (from W4) driver-parameter recovery error + calibration;
(from W5) seen vs unseen controller error ratio, structured vs black-box; (from W7) predicted vs true P(breakdown).

## Timeline (weeks start Tuesday 2026-10-06)
| Week | Dates | Work | Deliverable for professor | Gate |
|---|---|---|---|---|
| W1 | Oct 6-12 | Score WM-S run 1 (stage 10/30/60/120 checkpoints) vs keep-speed / pure physics, 5-300 s, car + traffic; audit SUMO HDV heterogeneity (vTypes, speedFactor/dev, IDM params, randomness) read-only; close remaining review blockers; WM-S2 prototype tests + dry run; systematic Scholar check (cites of Liu 2026, Montanino & Punzo 2021, survey 2604.12857) | Framing + thesis slide; run-1 accuracy table; data-audit result | G1: does existing data have per-driver variation? (decides how much the new dataset must add) |
| W2 | Oct 13-19 | Design + register heterogeneous-driver dataset: per-HDV IDM/LC parameters drawn from literature distributions (with correlations), truth logged per vehicle; controller split (train: native, damping, wall v2.1; held out: PI, wall v2.2 - rotate later); world/seed splits; pilot mechanics runs; batched (vectorised) controller code for many-particle rollouts | Dataset spec + pilot traces; held-out controller protocol | Registration acked |
| W3 | Oct 20-26 | Generate dataset (laptop + rented CPU box), QA; classical per-driver fitting baseline (per-vehicle gradient/RLS IDM fit); black-box WM baseline defined | Dataset stats; baseline identification numbers | Dataset QA pass |
| W4 | Oct 27-Nov 2 | Train WM-S2 v1 (GPU); C2 evaluation: parameter recovery vs observation length and CAV penetration, calibration of uncertainty; car/traffic accuracy vs run 1 | Identification results (C2 figure draft) | WM-S2 beats run 1 and pure physics on car/traffic |
| W5 | Nov 3-9 | C1 headline: leave-one-controller-out for WM-S2 and black-box baseline; paired effect + traffic metrics on unseen controller | C1 headline figure draft | G2: structured >> black-box on unseen controller |
| W6 | Nov 10-16 | Model iteration (long horizon 300 s, traffic-level losses, stability aids); decision test (controller choice regret) on fresh worlds | Updated accuracy + decision table | Beats always-native on fresh worlds (or documented why not) |
| W7 | Nov 17-23 | C3 risk layer: particles + common random numbers, per-step hazard, splitting; ground-truth P(breakdown) from repeated SUMO runs per snapshot (registered budget); calibration plot | Predicted vs true breakdown probability | Calibrated within agreed tolerance |
| W8 | Nov 24-30 | Shapley interaction pairs; causal validation by re-running SUMO with flagged pairs' parameters; attribution examples | "Normal alone, critical together" figure | Flagged pairs reproduce breakdown in SUMO |
| W9 | Dec 1-7 | Decision with fragility limit; closed-loop SUMO test with WM-S2 planner on fresh worlds, incl. an unseen controller in the action set | Closed-loop results | Closed loop safe; delay vs native reported |
| W10 | Dec 8-14 | Ablations (penetration levels, history length, no-encoder, black-box), robustness, figures | Full results set | - |
| W11-12 | Dec 15-28 | Paper draft (IEEE IV / T-ITS target), buffer for slips | Draft | - |

## Fallbacks
- G1 negative (little driver variation): expected; the W2-W3 dataset supplies it (and ground truth).
- G2 negative: analyse where the structured model fails (driver model too simple? controller-induced HDV
  behaviour outside training?); C2 and C3 still stand; narrow the C1 claim.
- WM accuracy stays short-horizon: C3 restricted to the horizon where it is calibrated; report the horizon.
- Breakdown probability not calibrated: present C3 as relative ranking of actions/pairs, validated causally (W8).

## Rules
Every simulation/training registered before running (runs/active/WM-S2-*), fresh output dirs, ledger close-out;
CAL/HOLDOUT/sealed stay sealed; owner approves GPU runs; executor routing per memory (GLM / DeepSeek flash / luna 6).

## C1-FAST (owner 2026-10-06 ~02:00: verify contribution 1 as fast as possible; max agents; rented GPU)
Replaces W2-W5 ordering for C1 only (C2/C3 follow afterwards).
- Day 0 (Oct 6, by ~noon): code in parallel - trainer --exclude-window-list/--window-list (ws-p2), weak black box
  = controller-id embedding (ws-bb), strong black box = learned HDV dynamics without physics, CAVs teacher-forced
  (ws-bb2), C1 scoring CLI with seen/unseen sets + bootstrap CIs (ws-c1eval); integrator merges; CPU dry runs.
  Data ready: E:/rd-wt/ws-p5/_out/lpo (1307 no-PI train windows, 189 val, 1494 PI eval states, 161 primary fleet-PI).
- Day 1 (Oct 6 afternoon/evening): register WM-S-C1FAST; gpuA (optionally + gpuB) trains structured WM-S,
  BB-id, BB-traj on no-PI windows, same budget each (3 processes share one 48 GB GPU, or split across 2 GPUs).
- Day 2 (Oct 7 morning): C1 scoring (seen vs unseen error ratio, structured vs black-box CI) -> preliminary verdict.
- Definitive C1 on heterogeneous drivers: if G1 shows little driver variation, register WM-S2-HETDATA Oct 6-7,
  simulate on a rented many-core CPU box Oct 7-8, train Oct 8, score Oct 9-10 (instead of W5).

## UPDATE 2026-10-09 (owner): world model first, direct output as main training data, then learned cooperative control (C4)

### Status at end of W1 (Oct 6-9)
- **C1 SUPPORTED** (CAVs teacher-forced; closed loop still to test). PI arm, u_same @60 s: structured 16.9 m vs BB-traj 23.8 / BB-id 28.2; paired CI excludes 0.
- **Dataset:** heterogeneous-driver dataset generated and packed (330 runs).
- **Trained:**
  - WM-S2 (driver-recovery scoring pending);
  - run 3 (better at 10-30 s, worse at 120 s);
  - OLDHIST (parity to 60 s with real history, blows up at 120 s).
- **Switching + direct-output dataset:** generated and packed (300 runs; packs on gpuB).
- **Diagnoses:**
  - DIAG120: long-horizon blow-ups = missed jams / under-braking, error carried backward from upstream.
  - EFFH-02: controller benefits need ~260-470 s to settle.
  - RANKING: C1 struct picks the better of native/PI in 80% of cases at 300 s, regret 19 vs 128/194 for fixed choices.
  - CTM alone: stable but weak at the merge.
- **Docs:** design discussion docs/DESIGN_DISCUSSION_20261008.md; lit checks docs/LITCHECK_MICRO_MACRO_20261008.md and docs/LITCHECK_MULTICAV_WM_RL_20261009.md.

### Status update 2026-10-09 (owner)
- **C1 extends to the unseen WALL.** `E:/rd-wt/ws-wallc1/_out/WALL_C1_RESULT.md`: structured 8.71 m vs bb_traj 25.41 / bb_id 40.07 m HDV median |ds| at 60 s on `wall_same`; verdict SUPPORTED with caveats. The PI-arm C1 result above stands; closed loop is still to test.
- **Braking-fix and no-braking models trained 2026-10-09** ([REGISTRATION.md](../runs/active/WM-S-MAIN2-20261009/REGISTRATION.md)); scoring in progress (`E:/rd-wt/ws-main2score`) — not yet a result. The braking fix targets DIAG120's under-braking / missed-jam blow-ups.
- **Data finding (in progress):** pack_v3 contributed **0** training windows; fix in progress (`E:/rd-wt/ws-packv3fix`).
- **entries-ON rescoring in progress** (`E:/rd-wt/ws-entrieseval`).
- **Black-box retraining dropped** (owner rule): new models are compared only with the previous C1 models.
- **Both GPUs shut down.**
- **This week = WM readiness + code setup;** direct output / per-CAV switching data is the main training data; RL (C4) comes later. Walls are one action; **regret** is the primary planner score.

### New contribution C4: learned cooperative multi-CAV control inside the world model
- **Claim:** because the world model executes CAV actions/controller code and is shown to generalise to unseen controllers (C1), it is a valid training ground for NEW controllers. A shared decentralised multi-CAV policy (own sensing + V2V from nearby CAVs, optionally RSU) learned mostly in imagination discovers new interactive moves - possibly wall/metering-like coordination - that transfer to SUMO.
- **Method:** Dreamer-style imagination; short imagined horizons (where the WM is accurate) plus a learned value function bridging the ~5 min benefit horizon (EFFH-02); per-CAV credit via analytic gradients through the differentiable gated physics, compared with PPO-in-model. Iterative SUMO validation rounds feed policy data back into the WM (each round registered with a small budget).
- **Closest prior work:** Liu, Xie & Xiong 2026 (arXiv 2608.17739, offline MARL in a physics-informed WM, SUMO on-ramp). Differentiate by: proven unseen-controller generalisation, online iterative loop, controller execution, analytic gradients + value bridging, emergent coordination validated in SUMO.
- **Required evidence:**
  - SUMO-measured vs WM-predicted gains;
  - uncertainty penalty / truncation against model exploitation;
  - full service metrics (incl. ramp/source waiting; guard against lane-blocking reward hacking);
  - baselines: model-free MAPPO in SUMO at an equal SUMO budget, PI, wall law, native.

### Revised sequence (owner priority: the world model first)
1. **Phase A - WM long-horizon fix** (W2, Oct 13-19).
   - Braking-response fix from DIAG120: decel-weighted / asymmetric "too fast" loss, follower propagation after leader braking; a better macro anchor is optional/longer-term.
   - Closed-loop ranking check (CPU).
   - One registered retrain. Gate: 120 s blow-up rate and 300 s ranking regret improve vs C1 struct.
2. **Phase B - direct output as the MAIN training data** (W3-W4, Oct 20-Nov 2).
   - Train the WM on the switching + direct-output packs (per-CAV 1 s action timeline; direct acceleration commands), with the Phase-A fix.
   - Extend direct-output data generation if needed (richer, longer profiles; more CAVs per world).
   - Zero-shot tests: held-out direct style and held-out switch pattern.
   - Gate: post-switch / direct-output HDV response accuracy, structured beats black box.
3. **Phase C - C2 scoring + C1 closed loop** (W5, Nov 3-9).
   - WM-S2 driver recovery vs dense truth (on a box).
   - C1 re-scored in closed loop.
4. **Phase D - C4 RL** (W6-W8, Nov 10-30).
   - Imagination training of the shared multi-CAV policy.
   - SUMO validation rounds (registered budgets); WM refresh with policy data.
   - Gate: SUMO-measured improvement over native / PI / wall law with all service metrics.
5. **Phase E** (W9-W10, Dec 1-14): C3 fragility (if time), ablations, figures. W11-12: paper draft.

### Status of earlier items
- C3 moves after C4 unless the owner re-prioritises.
- Training resumes only after the owner approves the Phase-A design.
- Self-recovery training stays rejected; the learned-120 s-anchor idea was discussion only.

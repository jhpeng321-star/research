# Causal Observation/Value-Screen Protocol for the Sequential P30 Timing Decision

**Status:** frozen 2026-08-30 before fitting anything. Development-only
(seeds 4121–4132); Path A untouched; no seeds ≥4201; **no MAPPO/RL** — this
protocol authorizes only the screen and its analysis.
**Library under study:** `runs/eval/idm_timing_pilot_v1/` (308 branches,
69 states) and `runs/eval/idm_timing_schedules_v1/` (216 single-trajectory
discovery records) — existing development artifacts only; no new SUMO run.
**Reference oracle:** constrained timing oracle +0.04784
(`runs/eval/idm_timing_oracle_v1.json`).

## 1. Exact prediction targets (frozen)

Model the decision sequentially, one scalar advantage target per stage:

```text
stage 3 (at t3):   Q_act(3) = realized reward of defer_r3   (penalties retained)
                   V_wait(3) = 0                            (abstain)
stage 2 (at t2):   Q_act(2) = realized reward of defer_r2   (penalties retained)
                   V_wait(2) = max(0, Q_act(3))             if t3 exists and
                              defer_r3 is constrained-admissible, else 0
stage 1 (at t1):   Q_act(1) = realized reward of activate_now (penalties retained)
                   V_wait(1) = max(0,
                                    V_wait(2)               if t2 exists,
                                    Q_act(2))               if t2 exists and
                                       defer_r2 is constrained-admissible, else 0
advantage(stage s) = Q_act(s) − V_wait(s)
```

- Realized activation rewards use attempted semantics: a post-activation dose
  failure or safety/service failure keeps its frozen −2 penalty and is
  never retrospectively replaced by abstention.
- `V_wait` is the best subsequently available **constrained** option
  (constraints pass and dose ≥ 16 of 20), matching the frozen oracle
  definition; it is a target used for the model, not evidence that the
  deployable policy may use realized outcomes.
- Missing future landmarks: a missing `t2` removes the `Q_act(2)`/`V_wait(2)`
  options (`V_wait(1) = 0`); a missing `t3` sets `V_wait(2) = 0`.
- Inadmissible actions (safety/service/dose): excluded from `V_wait`
  (constrained), but present in `Q_act` with their realized penalties.

## 2. Feature provenance (machine-readable audit)

Every feature is causally available at the current decision instant. The
existing artifacts store, per landmark of all 69 states: `runway_m`,
`predicted_dose_s`, `landmark` kind, `persistence_steps`, `lane_id`,
`step` (= current instant), `admissible`. Frozen families:

- **A. recurrence/phase + static local**: stage index (1/2/3), landmark kind
  (wave_onset / severe_onset), persistence steps, runway, predicted dose,
  current instant, lane one-hot (two mainline lanes);
- **B. local temporal history**: **not present in existing artifacts**
  (no per-step local sequences stored);
- **C. merge-zone RSU aggregates**: **not present** (no 134-field observation
  or classic RSU block stored for these states);
- **D. corridor-wide RSU aggregates**: **not present**;
- **E. interactions / compact recurrent**: only if A shows stable signal and
  the sample support (69 states, ≈120 stage samples) justifies it.

Consequently the screen evaluates family **A** now; families B–E are
declared **not-evaluable from existing artifacts** and would require a
separately authorized artifact-generation phase (new native-IDM schedule
capture on the same development states) — not part of this execution.

Forbidden as features: seed, demand, penetration (deployment/admission
variables, reported only), future landmark steps or gaps (`t2−t1`, `t3−t2`,
`L`, `H`), realized dose, outcomes/rewards, release reasons, hashes,
artifact identifiers, oracle-selected action, post-decision metrics.

## 3. Validation (frozen)

- Group every split by seed; one fold per seed (12 folds, 4121–4132); all
  states and stages of a seed stay in its fold.
- Leave-one-seed-out evaluation: each held-out seed is scored by models
  fitted exclusively on the other 11 seeds.
- Feature normalization, ridge alpha, and the per-stage activation
  threshold are selected strictly inside each training fold using
  seed-grouped out-of-fold predictions on the training seeds; the held-out
  seed is only scored once.
- Report by penetration, demand, recurrence stage and seed.

## 4. Models (frozen, data-appropriate)

1. always abstain;
2. fixed timing rules (activate_now / defer_r2 / defer_r3, pre-action
   infeasibility abstains, penalties retained);
3. simple per-stage threshold/rule (activate stage s iff training-OOF mean
   advantage for that stage exceeds a frozen positive grid value);
4. regularized linear (ridge) per-stage advantage model with a frozen
   threshold grid;
5. shallow tree / constrained gradient boosting: **not justified** — 69
   states / ≈120 stage samples do not support it (frozen decision);
6. no neural network (explicitly not justified by sample support).

Complex models may only be considered if the simple models show stable
signal; the protocol freezes the simple baseline as the deployable candidate.

## 5. Policy evaluation (frozen)

Construct an entirely cross-fitted sequential policy. For a held-out seed:
at `t1` the stage-1 policy decides activate-now vs wait from the frozen
feature vector; if it waits and `t2` materializes, the stage-2 policy
decides defer-r2 vs wait at `t2`; if it waits and `t3` materializes, the
stage-3 policy decides defer-r3 vs abstain. The chosen path is replayed
against the existing paired library and the **realized** outcome is taken:
activation branches keep their realized reward (including −2 penalties);
waiting paths that reach no further landmark end at abstain (reward 0).
Never replace a mistaken or failed predicted action retrospectively with
abstention. Pre-action feasibility (landmark exists and candidate admitted)
is the only deployment mask; realized dose/constraint status is never an
input.

## 6. Frozen pass/fail gates

The screen passes only if one completely cross-fitted deployable policy, on
the leave-one-seed-out aggregate:

1. mean paired policy value ≥ +0.02 over abstention **and** over the best
   fixed timing policy;
2. recovers ≥ 40% of the +0.04784 oracle headroom (≥ +0.0191);
3. aggregate value remains positive when each seed is left out
   (per-seed-complement value > 0);
4. meaningful benefit (> +0.02) in at least three independent held-out
   seeds;
5. not derived from one penetration, one demand, one recurrence stage or one
   outlier seed (value positive — ≥ 0 — in both penetrations, ≥ 2 of 3
   demands, ≥ 2 of 3 recurrence stages; most-contributing state < 50% of the
   total positive value);
6. selects both activation and wait (≥ 5 activations and ≥ 5 waits in the
   aggregate);
7. retains all action failures and negative outcomes (attempted, dose-failed,
   safety/service-failed, harmful/neutral counts reported; no silent
   abstention);
8. introduces no collision, teleport or frozen safety/service violation
   among selected outcomes;
9. performs above the shuffled-label control (policy value ≥ shuffled +
   0.01) and above the majority-action control (best fixed timing rule,
   always-wait = 0);
10. exceeds replay/numerical noise (abstain replay exact at 1e-12; realized
    effects beyond the 0.02 deadband).

Uncertainty: state-level bootstrap (10k resamples) and per-seed values are
reported, but the predeclared deterministic gates are not replaced by any
confidence interval.

## 7. Execution order

1. feature/schema inspection (done: only family-A numerics exist);
2. this frozen protocol;
3. leakage, grouped-split, recursive-target and policy-evaluation tests;
4. focused tests, then `python -m pytest -q`;
5. execute the screen on existing development artifacts only;
6. stop and report the cross-fitted results; no RL training before review.
   Recommendation rules: if the gate passes → recommend a frozen binary
   sequential option environment and reward audit (not automatically MAPPO);
   if it fails → diagnose whether the failure is observation insufficiency
   (thin family A could not be collected? B–D unavailable), sample scarcity,
   unsafe/dose prediction, or genuinely unpredictable metastability; do not
   add prospective seeds or expand model complexity automatically.

## 8. Distinctions (explicit)

- retrospective oracle value (+0.04784, unattainable);
- cross-fitted deployable policy value (this screen);
- training-set fit (reported, not evidence);
- prospective evidence (later, untouched seeds — not started).

Nothing in this protocol — and none of its results — is a claim that a
trained controller works.

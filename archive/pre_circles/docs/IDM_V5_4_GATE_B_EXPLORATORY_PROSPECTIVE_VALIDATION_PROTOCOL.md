# IDM v5.4 Gate-B exploratory prospective-validation protocol

Status: **frozen before any seed 4601--4660 feasibility or treatment artifact is
generated.**

Date: 2026-08-31

## Scientific role

The confirmatory development gate failed: no tested model achieved the frozen
90% safety / 50% joint-benefit requirement. This protocol does not reinterpret
that result. It authorizes one separately labelled, metastability-tolerant
prospective test of the development-selected **exploratory selective-enrichment
policy**. A pass supports an exploratory escalation gate; it does not establish
a calibrated safety probability or a universal guarantee.

The validation partition is seeds 4601--4660. These seeds were not used for
wall design, feature selection, model fitting or threshold selection. Final
paper seeds 5001--5060 remain sealed unless this validation passes.

## Frozen controller, predictor and policy

- Control arm: the frozen analytical IDM decentralized damping controller.
- Treatment arm: identical damping plus the frozen v5.4 compact moving-wall
  overlay, with identical assemble--hold--release and emergency-handoff logic.
- Gate F: the unchanged deterministic predecision geometry, lane-coverage,
  authority, activation-safety and projected-dose supervisor.
- Gate B candidate: `primary_shallow_boosting`, fitted once to all authorized
  development cases.
- Inputs: the 56-field primary decision-time feature contract, hash
  `3FB9061E85073EAF65A788132ECAFDED4839CCE4D1F9A4DF1F0CC28E74D1A729`.
- Safety threshold: `0.7000000000000002`.
- Conditional-benefit threshold: `0.7499999999999999`.
- Frozen bundle:
  `runs/eval/idm_wall_v5_4_gate_b_observation_model_repair_v1/model_bundle.joblib`,
  SHA-256
  `9cea403b96815dae0c856de88771cc49d03b1f8057d7fb0d9454afd593418662`.

The deployable policy first applies Gate F. A Gate-F refusal always abstains.
The bundle, thresholds and command function are frozen before treatment
outcomes exist. For efficient paired evaluation, the command is recomputed
offline after both arms finish, but the scorer can read only Gate F and the
common predecision feature vector. Tests require command IDs to remain exactly
unchanged if every dose, outcome and oracle field is mutated. The wall is
selected only when both Gate-B thresholds pass; counterfactual outcomes cannot
alter that frozen command.

## Frozen validation grid

- IDM, mainline demand 2700 veh/h, ramp demand 400 veh/h;
- CAV penetration 30%;
- seeds 4601--4660 exactly;
- decision steps 620, 660, 680 and 700;
- 20-step requested treatment and 260-step outcome horizon;
- four exact-repeat starts selected deterministically from the first four
  independent seeds with at least one Gate-F admission.

The feasibility scan is treatment blind. Paired outcomes may be generated only
after its manifest and analysis pass their frozen availability checks.
The scan must contain at least 40 admitted windows across at least 30 seeds and
at least three of the four decision steps, with projected duration at least 80%
of the request. These are availability checks, not treatment-effect outcomes.

## Operational denominators

The online population is **all Gate-F admissions**, not only cases that later
achieve the requested dose. A selected dose failure remains a selected action,
remains in every rate denominator and is always nonbeneficial. It may retain
its observed safety label, but is also reported separately as a failed-dose
command. This prevents post-treatment dose information from improving the
apparent predictor.

The feasible-always-wall baseline uses the same all-admitted population:

- baseline safety is the fraction of all admissions labelled safety-admissible;
- baseline joint benefit is the fraction of all admissions that are both
  dose-qualified and labelled `wall_selected`.

Gate-B enrichment is the selected rate minus this operational baseline rate.
As a pre-validation grouped-OOF sanity check, this denominator gives the
development always-wall baseline 116/199 = 58.29% safety and 53/199 = 26.63%
joint benefit; the threshold-selection predictions' 8/10 and 5/10 are
therefore +21.71 and +23.37 percentage points, respectively. These are
development cross-fit diagnostics, not prospective results. Dose-qualified-
only rates are diagnostics, not the enrichment baseline.

## Integrity and treatment gates

All of the following must pass before predictor efficacy is interpreted:

1. all 240 declared seed/window rows exist and case IDs are unique;
2. damping-versus-damping and treatment-versus-treatment repeats are exact;
3. predecision branch identity and every causal damping-preservation check pass;
4. no partial-lane hold or silently unreported compactness violation occurs;
5. realized effects exceed exact replay noise;
6. at least 80% of Gate-F admissions are dose-qualified;
7. selected dose success is at least 80%;
8. no selected action adds a collision or teleport;
9. every selected dose failure and every selected harmful/nonbeneficial case is
   retained and listed by seed, decision step and reason.

## Metastability-tolerant prospective gate

The validation cohort passes only if the frozen policy, without refitting or
threshold adjustment, satisfies all of the following:

- at least five selected actions across at least four independent seeds;
- selected safety-admissibility at least 80%;
- selected joint-benefit precision at least 45%;
- safety enrichment at least 15 percentage points over feasible-always-wall;
- joint-benefit enrichment at least 15 percentage points over
  feasible-always-wall;
- zero selected excess-collision or excess-teleport cases.

The reduced five-case/four-seed coverage requirement reflects that this
60-seed partition is half the size of development and the development policy
selected only 10 of 199 admissions. It was frozen before validation and does
not relax the outcome-rate or enrichment requirements. Wilson score intervals,
probability metrics, decision-time coverage, service/delay effects, dose
failures and all negative cases are mandatory reports but are not silently
converted into new thresholds.

## Stop and paper-readiness rules

- No threshold, feature, label, treatment or model change is allowed after a
  validation outcome is opened.
- If validation fails any integrity or prospective gate, final seeds remain
  sealed and the learned Gate-B claim ends for this treatment/data contract.
  The honest fallback is deterministic Gate F plus the demonstrated wall
  treatment boundary.
- If validation passes, the identical bundle and thresholds may be tested once
  on final seeds 5001--5060. The final cohort must independently meet the same
  five-case/four-seed and rate gates.
- A paper-ready exploratory predictor additionally requires the pooled 120
  untouched seeds to contain at least 10 selected actions across eight seeds
  while retaining the 80% safety, 45% joint-benefit, +15-point enrichment and
  zero-catastrophic-excess requirements. Development rows are never pooled into
  this held-out paper result.

No pass permits the phrases "calibrated safety probability", "guaranteed
safe", "generalized across IDM settings" or "collapse prevention" without a
separate prospective test of that claim.

## Frozen execution order

```powershell
python experiments/idm/scan_idm_wall_v5_4_predictor_validation.py --workers 4
python tools/analyze_idm_wall_v5_4_predictor_validation_feasibility.py `
  --scan-root runs/probe/idm_wall_v5_4_predictor_validation_feasibility_p30_s4601_4660 `
  --output runs/probe/idm_wall_v5_4_predictor_validation_feasibility_p30_s4601_4660/analysis.json
python experiments/idm/run_idm_wall_v5_4_predictor_validation.py --workers 4
python tools/analyze_idm_wall_v5_4_gate_b_prospective_validation.py
```

The scan and paired runner are resumable: existing per-seed artifacts are
skipped unless `--force` is explicitly supplied. Expected wall-clock time is
approximately 0.5--1.5 hours for the scan and 2--4 hours for paired replay on
four workers, depending on SUMO throughput.

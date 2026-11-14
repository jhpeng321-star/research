# Counterfactual Escalation Predictor — Diagnosis and Recovery Handoff

## Project and proposed paper

Workspace: `D:\research-dense`

Proposed IEEE IV paper title:

> Counterfactual Escalation from Decentralized CAV Damping to Coordinated Moving-Wall Control

The intended control policy is frozen decentralized CAV damping. The treatment
policy must use the identical damping controller plus a temporary coordinated
moving-wall override. CAVs must return to damping when the wall is inactive or
released; they must not be released to uncontrolled SUMO behavior.

The scientific goal is not to predict the exact deterministic time at which a
metastable breakdown occurs. It is to estimate the conditional expected value,
or probability of benefit, of escalating from damping to a moving wall using
traffic patterns observable before the decision.

## Bottom-line diagnosis

A probabilistic escalation predictor is plausible, but the current experiments
do not establish that it works. The primary limitation is not the choice of
classifier and not metastability itself. The current data-generating process
does not deliver a consistent wall treatment or enough treatment-effect
headroom.

The existing all-case prediction signal is dominated by whether a wall can be
formed from the current CAV placement. Once analysis is restricted to cases in
which the wall actually engages, benefit prediction is approximately chance.
More seeds or a more complex model will not fix this until the treatment,
outcome, and sampling protocol are corrected.

## Evidence from the current v2 artifacts

Primary artifact directory:

`runs/probe/counterfactual_escalation_predictor_v2/`

### Dataset composition

- 300 eligible decision cases from 100 seed groups across the current five
  primary cells.
- 98 cases engaged the wall, across 67 seeds.
- 202 cases were vetoed by formation or runtime feasibility.
- 65 engaged cases were safety-admissible.
- 33 engaged cases were unsafe, across 29 seeds.
- 20 cases passed the final safety/service selection oracle, across 19 seeds.
- The current primary set omits the originally planned dense/p10 cell.
- The small IDM pilot has 24 cases from 8 seeds and zero selected cases, so it
  provides no transfer evidence.

### The planned wall treatment is almost never delivered

Of the 98 engaged cases:

- Only 1 received the full requested 60 simulation steps (30 seconds).
- 97 stopped early.
- 13 received one override step or less.
- Mean override duration was 8.11 steps, approximately 4.05 seconds.
- The typical case received approximately 5 steps, or 2.5 seconds.
- 92 stopped because of `formation_abort:uncovered_occupied_lane`.
- 4 stopped because of `formation_abort:insufficient_inband_cavs`.
- 1 stopped because of a controlled-follower hard-brake safety abort.

The controller recomputes formation feasibility during treatment. Requiring an
in-band CAV for every currently occupied lane makes the treatment terminate as
soon as lane occupancy or CAV coverage changes. Consequently, the nominal
30-second moving wall is usually a short and highly variable impulse.

Relevant implementation:

- `policies/counterfactual_escalation.py`, especially `_formation_status()` and
  the runtime formation-abort logic in `act()`.
- `experiments/counterfactual/generate_counterfactual_escalation_headroom.py`, especially the treatment
  duration and `require_all_occupied_lanes` configuration.

### There is no collapse-prevention headroom in the present data

Across all 300 cases:

- Sustained congestion under damping: 130 cases.
- Sustained congestion under the wall: 129 cases.
- Collapse prevented by the wall: 1 case.
- Collapse induced by the wall: 0 cases.

Among the 98 engaged cases:

- Damping collapsed in 54 cases.
- Wall collapsed in 53 cases.
- Both arms collapsed in 53 cases.
- Neither arm collapsed in 44 cases.
- The wall prevented collapse in only 1 case.

Therefore, a predictor of collapse prevention has only one positive treatment-
effect example. No statistical or machine-learning model can learn a credible
conditional collapse-prevention probability from this dataset.

### The reported all-case AUC is misleading for treatment benefit

From `primary_5cells_model_signal_v1.json` and
`primary_5cells_multihead_signal_v1.json`:

- The best all-eligible model reaches AUROC of approximately 0.81, average
  precision of approximately 0.19, and Brier score of approximately 0.09.
- Placement/local features predict wall engagement at approximately AUROC 0.99
  and average precision 0.99.
- Among the 98 engaged cases, the best meaningful-benefit discrimination is
  only approximately AUROC 0.56.
- Engaged-only prediction of final selection is approximately chance.
- Most continuous treatment-effect targets have negative out-of-fold R-squared.
- Low-TTC exposure reduction has only a small positive signal, with very few
  positive examples, and is not yet robust.

The current all-case model is principally learning whether CAV placement permits
wall formation. It is not demonstrating that traffic state predicts whether a
successfully delivered wall will help.

### The current benefit label mixes different mechanisms

The safety/service oracle defines meaningful benefit as any of:

- At least two fewer hard-brake onsets.
- At least three fewer low-TTC exposure steps.
- At least three additional discharged vehicles.

It then requires safety and service noninferiority. Among engaged cases:

- 26 show a hard-brake-onset benefit.
- 9 show a low-TTC benefit.
- 21 show a discharge benefit.
- 44 show at least one of the three benefits.
- No case improves all three mechanisms.
- 17 nominal benefit cases are unsafe.
- 11 nominal benefit cases fail service constraints.

These outcomes can arise from different physical mechanisms. A single composite
binary target obscures those mechanisms and makes the learned relationship less
stable. The current oracle is implemented in
`counterfactual/safety_service_oracle.py`.

### The sampled cells do not isolate the metastable boundary

- Dense cells are largely stable and show essentially no collapse.
- Hyper/p10 engaged cases are saturated: essentially all collapse under both
  arms.
- Hyper/p20 and hyper/p30 contain more variation, but the current wall rarely
  changes the collapse result.
- Some decisions occur after congestion has already started, mixing prevention
  with recovery.
- Fixed decision times mix stable, pre-breakdown, and already-congested states.

A conditional probability model needs observations around the transition band,
where the damping-only collapse probability is neither near zero nor near one
and where the treatment can change the result.

### Sample size and dimensionality are mismatched

- The richest feature set has approximately 58 variables.
- The engaged cohort contains only 98 rows from 67 independent seed groups.
- Only 20 cases pass the final composite oracle.

The effective independent sample size is closer to the number of seeds than the
number of decision rows. Rich nonlinear models are therefore highly vulnerable
to overfitting. This is secondary to the treatment-dose problem but will remain
important after treatment repair.

## Required recovery sequence

### Phase A — Make the moving wall a consistent intervention

Do not generate a new predictor-scale dataset yet.

Implement and validate a deliberate assemble-hold-release protocol:

1. Check formation feasibility before activation.
2. Select and latch wall members at activation.
3. Maintain members for a prespecified time or distance using longitudinal and
   lane-coverage tolerances.
4. Do not terminate merely because a newly occupied lane lacks a new in-band
   CAV on one simulation step.
5. Where needed, recruit a replacement CAV or perform an explicit controlled
   handoff.
6. Retain safety aborts for genuinely dangerous conditions.
7. Release smoothly back to the identical frozen damping controller.
8. Preserve identical damping for all out-of-band CAVs throughout treatment.

Record treatment dose explicitly:

- Requested and realized override duration.
- Controlled CAV-seconds.
- Lane coverage over time.
- Wall-member handoffs and abort reasons.
- Speed and acceleration deltas relative to damping replay.
- Merge inflow/discharge effect during and after treatment.

Formation should be a deterministic eligibility gate. It should not be mixed
into the treatment-benefit label or credited as predictive success.

### Phase B — Run a corrected 12–20-seed action-headroom pilot

Before model fitting, require evidence that the corrected wall produces a real
and sufficiently varied intervention.

Suggested pass conditions:

- Damping-versus-damping replay identity passes.
- Treatment branches remain identical before the declared decision.
- In-band CAVs receive the wall override only while treatment is active.
- Out-of-band CAVs retain identical damping actions.
- At least 80% of feasible activations receive at least 80% of the planned wall
  dose, unless a documented safety abort occurs.
- Unsafe engagement remains below a prespecified acceptable threshold.
- Completed treatments include beneficial and nonbeneficial outcomes across
  multiple independent seeds.
- The wall changes the outcome more than replay or numerical variability.
- More than one collapse-prevention example appears if collapse prevention is
  to remain the primary scientific target.

If these conditions fail, redesign the wall action before collecting more data.

### Phase C — Locate the metastable transition band under damping

Stay within the dense/hyper workspace; do not resume medium optimization.

Run a damping-only risk map over a narrow grid of mainline demand, ramp demand,
CAV penetration, and relevant driver-model parameters. Identify cells/windows
where the future sustained-collapse probability under damping is approximately
20–80%.

Use event-aligned candidate decision landmarks based only on predecision data,
such as:

- Merge speed decline and its recent slope.
- Growth of speed variance or headway variance.
- Queue growth rate rather than queue area alone.
- Upstream/downstream density or speed imbalance.
- Ramp pressure and merge discharge deterioration.
- CAV lane coverage and longitudinal placement.

Exclude already-collapsed cases from a prevention model. Analyze recovery as a
separate task and separate outcome.

### Phase D — Define separate causal outcome heads

The primary collapse target should be based on paired counterfactual arms:

`tau_C(X) = P(C_damping = 1 | X) - P(C_wall = 1 | X)`

where `C` is future sustained congestion or a defensible capacity-drop event.

Estimate separate quantities for:

1. Baseline collapse risk under damping.
2. Collapse risk under a completed wall treatment.
3. Safety effects, including hard braking and low TTC.
4. Service effects, including discharge and possibly delay as a secondary cost.
5. Treatment feasibility, handled primarily as a deterministic gate.

Escalation should be based on expected utility, for example:

`collapse cost avoided + expected service gain - expected safety/service cost`

rather than a single heterogeneous OR label.

### Phase E — Start with a small, calibrated model

After treatment and headroom pass:

- Begin with 6–12 mechanism-based predecision features.
- Prefer regularized logistic regression, a generalized additive model, or
  shallow constrained boosting before a large neural model.
- Split and bootstrap by seed, not by decision row.
- Prevent all information from later decisions in the same seed from leaking
  across folds.
- Report calibration, Brier score, average precision, confidence intervals,
  safety violations, policy value, and regret—not AUROC alone.
- Increase model complexity only if repeated seed-held evaluation demonstrates
  a stable improvement.

The current features use mixed-traffic state from `snapshot.vehicles`, including
HDVs. The paper must either assume RSU/infrastructure sensing with V2X
aggregation or restrict the feature set to quantities realistically obtainable
from CAV sensing and communication.

## Falsification and pivot criteria

More seeds should be collected only after the corrected pilot demonstrates both
consistent treatment dose and causal headroom.

If a stable, safely delivered wall still prevents collapse in fewer than roughly
5% of appropriate at-risk cases, or if it produces no reproducible service or
safety effect, then a collapse-prevention predictor is not supported by this
scenario. More samples cannot compensate for a treatment that has essentially
no effect.

At that point, possible honest pivots are:

- Predicting short-horizon safety stabilization.
- Predicting discharge recovery or deterioration.
- Treating the wall as a constrained recovery action after breakdown rather
  than a preventive action.
- Reporting the conditions under which decentralized damping cannot be
  usefully escalated with the tested wall design.

## Recommended paper claim if the corrected pipeline succeeds

Avoid claiming deterministic breakdown prediction. A defensible framing is:

> We estimate the conditional expected value of escalating from decentralized
> CAV damping to coordinated moving-wall control under partially observed
> metastable merge traffic. Paired counterfactual replay isolates the effect of
> escalation, while deterministic feasibility and learned safety/service heads
> constrain activation.

The central novelty would be the counterfactual escalation decision—not merely
a health-factor activation threshold and not merely the moving-wall controller.

## Immediate instruction to the implementation agent

First inspect the current dirty worktree and preserve every existing user
change. Read `AGENTS.md`, `docs/COUNTERFACTUAL_ESCALATION_HANDOFF.md`,
`docs/COUNTERFACTUAL_ESCALATION_V2_PLAN.md`, and this document. Then audit the
runtime wall termination path and propose the smallest damping-preserving
assemble-hold-release implementation slice. Do not generate another paper-scale
dataset and do not claim that the predictor works until the action-dose and
headroom gates above pass.

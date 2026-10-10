# IDM Phase-5 action-local residual-pulse protocol

Status: frozen development diagnostic. This protocol does not authorize PPO
training or an independent controller claim.

## Scientific comparison

The nominal controller is analytical IDM damping. At a deterministic causal
landmark, one gate-selected lane-role receives either `-0.10` or `+0.10 m/s²`
for 10 one-second controller steps through the existing five-second filter,
bounds, and safety projection. The paired control requests exact zero residual.
The treatment returns to zero after the pulse. If the frozen controller
performs a deterministic same-lane handoff during the pulse, the residual
follows that selected lane-role. It never moves to another lane or a
nonselected CAV. At least 8 of 10 requested role steps are required for an
admissible diagnostic treatment.

Safety and wave outcomes use the first 20 seconds after the decision. Service
outcomes use 60 seconds because existing decision observations place the median
selected absorber about 41 seconds from the merge and the upper quartile about
51 seconds away.

## Landmarks

After the 300-step warmup, a landmark requires exactly one gate-selected
absorber, at least 150 m disturbance runway, no already-sustained collapse, and
three consecutive steps satisfying one condition:

- wave onset: normalized wave level at least 1.00 and short-minus-long trend at
  least +0.10;
- severe-braking onset: normalized severe level at least 1.00 and
  short-minus-long trend at least +0.10.

Accepted landmarks are separated by at least 60 controller steps. At most one
of each type is retained per demand/seed. Wave onset wins a same-step tie.
Sustained collapse retains the frozen definition: at least five pre-merge
vehicles below 15 m/s while mean pre-merge speed is at most 15 m/s for 60
consecutive controller steps.

## Effects and deadband

The two residual directions remain separate action-state pairs. The safety
composite is the mean of reference-normalized severe-braking and integrated
acceleration-squared improvements relative to zero. Helpful means greater than
the symmetric deadband with all constraints passing; harmful means less than
the negative deadband or a failed hard constraint; values inside the deadband
are neutral.

The base deadband is ±0.02. The duplicate-zero smoke freezes it at the larger
of 0.02 and twice the p99 absolute replay effect before treatment results are
used.

## Resume contract

The runner writes:

- `manifest.json`: immutable protocol and hash;
- `landmarks/*.json`: atomic zero-residual landmark discoveries;
- `branches/*.json`: atomic paired branch results;
- `progress.json`: current completed/pending counts;
- `summary.json`: completed paired analysis.

To resume after interruption or power loss, rerun the identical command. Valid
completed artifacts are skipped. A changed protocol is rejected for the same
output directory. At most the currently running branch is lost.

## Seed discipline

Only development seeds 4107--4120 are allowed. The CLI rejects other seeds.
All decisions from one seed remain grouped in statistical evaluation.

## Execution order

First run the two-seed duplicate-zero smoke. Only if its identity, execution,
horizon, safety, and numerical gates pass may the 14-seed development
diagnostic run. Neither run opens controller-validation or paper-test seeds.
The full run receives the passing smoke `summary.json` through
`--replay-noise-source`; its hash and frozen deadband are stored in the full
manifest.

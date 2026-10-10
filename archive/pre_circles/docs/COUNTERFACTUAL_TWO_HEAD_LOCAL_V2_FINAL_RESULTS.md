# Counterfactual Two-Head Local-v2 Gate: Frozen Development and Locked-Test Results

Date: 2026-08-25

## Bottom line

The paired counterfactual wall intervention has reproducible selective headroom,
but the current pre-treatment two-head predictor does not generalize to unseen
traffic seeds. It must not be described as a working deployable gate.

More independent seeds improved the evidence that beneficial intervention cases
exist. They did not make the instantaneous treatment effect identifiable from
the current local-v2 state representation with grouped ridge models.

## Frozen experimental contract

- Development seeds: 101--130.
- Locked test seeds: 201--210.
- Decision steps per seed: 120, 180, and 240.
- Development cases: 90 paired off/soft cases.
- Locked cases: 30 paired off/soft cases.
- Soft-wall extra tau: 0.0 seconds.
- Treatment horizon: 120 steps.
- Outcome horizon: 300 steps.
- Frozen congestion definition: queue speed at most 10 m/s, queue count at
  least 10, pre-merge mean speed at most 10 m/s, sustained for 30 steps.
- Frozen safety budgets:
  - hard-brake rate at most 0.2006667 vehicle-steps/s;
  - low-TTC exposure fraction at most 0.0466667;
  - zero collisions;
  - zero starting teleports.
- Model: one standardized L2 ridge model for each continuous target.
- Development evaluation: nested leave-one-seed-out.
- Locked evaluation: one use after development model and rule were frozen.

The exact preregistered protocol is in
runs/probe/counterfactual_two_head_protocol_frozen.json.

## Implementation and data verification

- Each generated record contains 240 causal-at-decision features.
- The frozen model uses 32 local-v2 features.
- Added inputs cover local lane and spatial-bin traffic, speed variance,
  hard-braking and TTC history, downstream/upstream speed gradients, CAV
  formation gaps, and CAV/neighbor speed mismatch.
- All 90 development and 30 locked cases are unique.
- Every case contains matched off and safe-soft arms.
- No development arm terminated early.
- No development arm contained a collision or starting teleport.
- Focused implementation tests: 17 passed.

## Development results

The frozen readiness audit found 58 off labels and 32 deployable-soft labels.

### Continuous held-out-seed prediction

| Target | RMSE | R2 | Correlation |
|---|---:|---:|---:|
| Sustained-congestion reduction | 0.507 | -0.107 | -0.159 |
| Queue-area reduction | 1272.219 veh-s | -0.129 | -0.132 |
| Delay reduction | 1215.385 s | -0.128 | -0.134 |
| Additional discharge | 7.654 vehicles | -0.079 | -0.056 |
| Soft hard-brake rate | 0.035 | -0.123 | -0.190 |
| Soft low-TTC fraction | 0.013 | -0.053 | 0.021 |

Every target performed worse in mean-square error than predicting the
development mean for a held-out seed.

### Development policy comparison

| Policy | Activations | Unsafe activations | Congestion cases | Queue area | Delay | Discharge |
|---|---:|---:|---:|---:|---:|---:|
| Always off | 0 | 0 | 22 | 12458.44 | 17507.47 | 315.44 |
| Always soft | 90 | 12 | 19 | 12242.91 | 17363.42 | 316.82 |
| Deployable oracle | 32 | 0 | 13 | 12000.48 | 17083.16 | 318.38 |
| Cross-fitted two-head gate | 47 | 7 | 21 | 12354.26 | 17401.29 | 316.52 |

The development gate produced small average traffic improvements over always
off but violated safety seven times and was worse than always soft on average.
It therefore failed the prespecified development status check.

## Locked-test results

The locked audit contained 22 off labels and 8 deployable-soft labels.

### Locked continuous prediction

| Target | RMSE | R2 | Correlation |
|---|---:|---:|---:|
| Sustained-congestion reduction | 0.320 | -0.034 | 0.152 |
| Queue-area reduction | 1590.919 veh-s | -0.124 | -0.120 |
| Delay reduction | 1493.193 s | -0.142 | -0.002 |
| Additional discharge | 8.431 vehicles | -0.280 | -0.160 |
| Soft hard-brake rate | 0.026 | -0.200 | -0.383 |
| Soft low-TTC fraction | 0.013 | -0.283 | -0.096 |

### Locked policy comparison

| Policy | Activations | Unsafe activations | Congestion cases | Queue area | Delay | Discharge |
|---|---:|---:|---:|---:|---:|---:|
| Always off | 0 | 0 | 4 | 11869.27 | 16757.74 | 320.73 |
| Always soft | 30 | 1 | 5 | 12059.60 | 17122.52 | 318.90 |
| Deployable oracle | 8 | 0 | 3 | 11427.40 | 16413.89 | 322.40 |
| Frozen two-head gate | 19 | 1 | 3 | 12110.00 | 17079.83 | 319.43 |

The frozen gate selected only 3 of the 8 deployable cases while activating in
19 cases total:

- deployable activation precision: 3/19 = 15.8%;
- deployable activation recall: 3/8 = 37.5%;
- unsafe activations: 1;
- queue area versus always off: +240.73 veh-s;
- delay versus always off: +322.08 s;
- discharge versus always off: -1.30 vehicles.

It failed the locked pass criteria. All policies remained collision- and
teleport-free in this dataset, but the learned gate exceeded a calibrated
safety budget in one activation.

## What the experiment establishes

1. The scalar health signal is unusable in the hyper-density cases because it
   saturates at zero.
2. The safe-soft wall has real but state-dependent treatment headroom.
3. Always-on intervention is not robust: it helped on average in development
   but harmed all three mean traffic metrics on the locked seeds.
4. A counterfactual oracle is consistently valuable:
   - development: 9 fewer congestion cases, 457.97 veh-s lower queue area,
     424.31 s lower delay, and 2.93 more discharged vehicles than off;
   - locked: 1 fewer congestion case, 441.87 veh-s lower queue area,
     343.85 s lower delay, and 1.67 more discharged vehicles than off.
5. The present instantaneous observation does not support a transferable
   point treatment-effect estimate with the frozen ridge formulation.

## Scientifically defensible interpretation

The road's metastability does not make conditional decision-making impossible,
but it makes exact pre-intervention treatment-effect prediction highly
uncertain. The negative result across 30 development seed groups and a locked
10-seed test means that simply adding more seeds to the same model and feature
set is not a supported next step.

The counterfactual framework remains valuable as:

- an oracle upper bound;
- a data-generation and evaluation protocol;
- a way to identify safe versus harmful interventions;
- a supervisor for learning a response-adaptive controller.

It is not currently valuable as evidence that a pre-treatment predictor works.

## Recommended next pivot

Move from a one-shot prediction gate to a safety-shielded
**probe-and-commit response gate**:

1. Apply a short, low-intensity safe-soft pulse only when deterministic CAV
   formation constraints are satisfied.
2. Observe early causal response signals during the pulse: queue slope,
   downstream discharge, shock-front speed, braking onset, and TTC.
3. Continue the wall only when response improves and safety stays within
   budget; otherwise release immediately.
4. Evaluate pulse/continue/release branches with the same matched replay
   framework.

This pivot asks an easier and more physically grounded question:

> After a safe diagnostic intervention, is the traffic responding favorably
> enough to continue?

It avoids claiming that the metastable future can be inferred reliably from a
single pre-treatment snapshot. It also retains the strongest novel parts of
the work: moving-wall control, safety correction, matched counterfactual
evaluation, and selective activation.

If another offline model is explored, seeds 201--210 can no longer serve as a
locked test because their outcomes have now been consumed. A new locked split,
for example seeds 301--310, must be reserved before model changes.

## Claim-safe paper language

> Paired replay reveals substantial selective headroom for a safe CAV
> moving-wall intervention, while always-on control is not robust across
> independent traffic realizations. A preregistered 30-seed development and
> 10-seed locked evaluation shows that neither a saturated scalar health
> signal nor a 32-feature instantaneous local-state ridge model can reliably
> predict treatment effect and safety out of sample. These findings motivate
> a response-adaptive probe-and-commit gate rather than one-shot activation.

Do not state that the current predictor works or generalizes.

# Frozen Rule-Based P30 Arm — Prospective Acceptance Results

**Date:** 2026-08-30  
**Status:** complete; **acceptance fails**. The rule remains development-only
and is not promoted as a prospectively accepted IDM damping controller.

## Frozen prospective cohort

- Seeds 4201--4220, opened once for this acceptance only.
- Six cells: CAV penetration 5% and 10% by mainline demand 1900, 2000 and
  2100 veh/h.
- One frozen opportunity per seed-cell: third persistent `risk_onset`, runway
  rank-1 candidate, fixed P30 for 20 one-second steps, 60-second paired
  horizon.
- 120 total opportunities: 58 activations and 62 explicit no-decision
  abstentions.
- All replay, discovery, branch identity, horizon, target-only command, dose,
  collision, teleport, low-TTC and immediate-follower braking checks pass.

Artifacts:

- `runs/eval/idm_p30_rule_prospective_acceptance_v1/`
- `runs/eval/idm_p30_rule_prospective_acceptance_analysis_v1/summary.json`

## Outcome

- Helpful: 12/58 activations.
- Harmful: 13/58 activations.
- Neutral: 33/58 activations.
- Harmful fraction: 22.41% (passes the <=25% bound).
- Helpful-or-neutral: 77.59% (passes the >=70% bound).
- Opportunity-level mean physical effect: +0.00382 reference units.
- Opportunity-level Path-A acceptance reward: -0.04325.
- Activation-conditional mean physical effect: +0.00790.
- Activation-conditional acceptance reward: -0.08949.
- Seed-cluster bootstrap 95% interval for opportunity-level reward:
  [-0.1243, +0.0157].
- Every leave-one-seed-out reward remains negative.

Three activations failed the frozen merge-discharge noninferiority budget;
no activation failed collision, teleport, low-TTC, immediate-follower braking,
delay, arrival, dose, horizon, command-integrity or treatment-abort checks.

## Frozen gate verdict

Passes:

- positive mean physical effect;
- harmful fraction <=25%;
- helpful-or-neutral fraction >=70%;
- dose/command integrity;
- safety noninferiority;
- zero added collisions/teleports;
- exact replay and branch identity;
- qualified effects exceed the fixed 0.02 deadband and zero replay noise.

Fails:

- positive mean acceptance reward;
- helpful activations more frequent than harmful activations (12 versus 13);
- service noninferiority (three discharge failures);
- every leave-one-seed-out mean positive;
- all exact constraints pass.

Therefore **`acceptance_passes = false`**. No threshold, treatment, seed,
metric or classification rule was changed after prospective outcomes opened.

## Scientific interpretation

The fixed P30 rule shows small positive average physical attenuation and a
bounded harmful fraction, but it does not prospectively satisfy the complete
safety/service/value contract. It may be reported as a conservative
development arm and as evidence of a low-authority boundary; it must not be
presented as a validated or trained IDM controller.

The escalation project can continue only with an explicit control-arm
interpretation frozen before wall experiments. Outside the 5--10% admission
domain the rule abstains, so a higher-penetration compact-wall pilot would
compare `frozen damping policy + wall` against the same policy's native-IDM
abstention, not against active P30. Changing to another active damping baseline
would be a new scientific control definition requiring separate validation.

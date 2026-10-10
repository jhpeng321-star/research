# Prospective acceptance of the standalone trained damping controller

Status: criteria specified before reserved outcomes; analysis and candidate
export checks implemented. No acceptance scenario has been generated. The full
training campaign is still active and no damping checkpoint is accepted.

This protocol supplements `NATIVE_DAMPING_FULL_CAMPAIGN_2026-09-05.md`.
It does not change the running learner or development selection criteria.

## Entry and identity

Complete all three optimizer lineages through 100 optimized episodes and retain
all nine prespecified checkpoint evaluations. Select using the original eight
development cases and their frozen eligibility/ranking rule. If none qualifies,
there is no candidate and acceptance remains sealed. A promising training return
or an early eligible checkpoint cannot bypass this requirement.

`tools/freeze_damping_candidate.py` checks completion, source snapshots, scenario
and checkpoint identities, raw trajectory cost accounting and recomputed
development eligibility. The selected actor must exactly reproduce saved
deterministic actions with the original per-tick batch shapes. It exports an
immutable candidate bundle with actor weights, controller/map, source snapshot,
evidence hashes and these acceptance criteria. Its status is candidate only;
it cannot authorize a wall or label itself accepted.

## Fixed prospective comparison

Use the 32 reserved cases, seeds 912001–912032: eight each at mainline loadings
2000, 2200, 2600 and 2700 veh/h, with the existing fixed-request, p20 CAV,
2400-second fixture. Compare native IDM, the already specified five-second
velocity-matching comparator, and the one frozen learned candidate. This is
96 full rollouts, separate from the training/development budget. Each triplet
uses identical requests, types, CAV assignment, network and simulator settings.
The learned arm runs deterministically without its training critic.

The current scenario API rejects acceptance generation. A provenance-checked
acceptance runner and generation authorization still need implementation after
the candidate exists. No data are opened by the analysis or export tools.
Do not tune weights, thresholds, checkpoint choice or features on this cohort.
If a candidate fails, a revised controller needs a newly registered prospective
cohort; these outcomes become development evidence.

## Outcomes and thresholds

For each paired seed let B be severe-braking vehicle-seconds and E be integrated
squared acceleration. Compute braking improvement `b = 1 - B_learned/B_native`,
energy improvement `e = 1 - E_learned/E_native`, and composite `c = (b+e)/2`.
These are physical comparison metrics, not the PPO reward or its fixed scales.
Service uses the entire requested population, including waiting at sources.

Every one of the 32 cases must satisfy the frozen development constraints:

- No added reported collision involvements or starting teleports.
- System time and source waiting increases each at most 2 seconds per requested
  vehicle; at most one fewer arrival and one additional unfinished request;
  no additional uninserted source backlog at the horizon.
- TTC below 3 seconds exposure increase at most the greater of 5 vehicle-seconds
  and 5% of native exposure.
- Composite worsening no greater than 10% on any individual case.

Across the 16 lower-loading cases, mean composite improvement must be at least
5%, with neither mean braking nor mean energy worsening. Across the 16 overload
cases, mean braking and mean energy may each worsen by at most 2%.

Uncertainty adds two requirements on the lower-loading group:

1. The lower endpoint of the nominal 95% paired bootstrap interval for composite
   improvement must be strictly above zero.
2. Compute the analytical comparator's composite against the same native
   denominator. The lower interval endpoint for learned-minus-analytical
   improvement must be at least -0.02. This allows at most a two-percentage-point
   noninferiority margin in units of native composite burden; it does not mean
   that learning must or will beat analytical feedback.

The second criterion is added prospectively here before acceptance data exist.
It prevents a native-only success claim from ignoring a substantially better
fixed comparator. The two-point margin is an engineering choice, not an
empirically calibrated universal threshold. Analytical superiority remains a
separate result requiring its own supporting interval. Passing noninferiority
alone does not demonstrate that learning is necessary.

Use 20,000 resamples with bootstrap RNG seed 37. Resample paired seeds within
each loading, preserving the three-arm pairing, and weight the two loadings
equally within each regime. Report 2.5th/97.5th percentiles. Show both individual
components, composite, service and analytical differences for both regimes.
Do not bootstrap individual vehicles as if they were independent experiments.
The intervals are nominal uncertainty estimates for this scenario family;
eight seeds per loading give limited precision. Component overload limits are
mean constraints, not confidence bounds on generalization. TTC and event
checks are observed simulation constraints, not proof of collision avoidance.

## Acceptance and paper claims

`idm/damping_acceptance.py` requires the complete correctly labeled cohort,
computes the above checks and returns analysis only. Even a passing analysis
keeps `accepted_checkpoint=False` until trajectory, runtime and frozen-bundle
provenance are verified by the acceptance runner. The accepted D* package must
retain the candidate hash, all paired results and the final verification record.

Only then proceed to damping-preserving wall mechanics and CAV-reported RSU
escalation. This gate establishes a useful controller on the registered fixed
scenario family. Sensor noise/occlusion/delay, changed penetration, unseen demand
families, controller ablations and external validity remain separate paper
questions. No damping acceptance result establishes an escalation predictor.

## Verification and registration record

The combined acceptance/campaign checks pass 13 tests. They include complete
synthetic nine-checkpoint evidence, rejection of a mislabeled raw trajectory,
an incomplete campaign, uncertain positive means, individual harm and analytical
inferiority. Four previously collected exposed deterministic cases reproduce
all 108,843 saved actor actions exactly; this replay required zero new SUMO
runs and is an identity check, not fresh efficacy evidence.

The dated registration and source snapshot are stored under
`runs/probe/native_damping_acceptance_preregistration_v1/`. Candidate export
verifies that registration against the campaign identity, acceptance cases,
criteria and archived/current source hashes. It copies the registration and
its sources into the prospective candidate bundle. Acceptance execution is
still pending; no artifact in this registration contains reserved outcomes.

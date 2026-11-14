# IDM Decentralized p30 Fresh Validation Report

Status: fixed-p30 benefit prediction and the ranked-neighbor upgrade are
rejected. MAPPO training remains unauthorized.

## Evidence completed

All experiments used native IDM as abstain, the unchanged 20-second p30
terminal-speed ceiling as treatment, 60-second paired outcomes, hard SUMO/TTC/
secure-gap/comfort/jerk projection, 10% scattered CAV penetration and exact
abstain replay.

Initial lower-loading development (seeds 4107--4120) produced 105 decisions at
1900, 2000 and 2100 veh/h. The current 134 fields narrowly failed observation
sufficiency (LOSO AUROC 0.676, balanced accuracy 0.641). A post-hoc ranked-three-
neighbor family appeared promising and was frozen before opening new seeds.

Fresh feature validation used seeds 4121--4132 over 1900, 2000, 2100 and 2200
veh/h. It found 114 decisions and generated 342 paired branches. Integrity held:

- exact discovery, branch and abstain-replay identity;
- all horizons complete;
- zero collision and teleport observations;
- 93.0% adequate treatment dose;
- 94.7% safety/service admissibility excluding dose;
- 59 qualified nonneutral outcomes.

## Rejected prediction hypothesis

On fresh seeds:

- current full observation: AUROC 0.507, balanced accuracy 0.465;
- frozen three-nearest-neighbor fields: AUROC 0.504, balanced accuracy 0.479;
- within 2000: ranked-neighbor AUROC 0.29, balanced accuracy 0.41.

The earlier neighbor signal did not replicate. It was development-sample
structure, not a reliable action-benefit predictor. No ranked-neighbor encoder
or recurrence field should be implemented from these results.

## Rejected fixed-p30 operating-domain claim

At 2000 on the fresh seeds, 30 opportunities produced only 24 fully qualified
treatments. Mean qualified effect was +0.0192 reference units, below the frozen
+0.02 gate. The harmful fraction among qualified cases was 16.7% and the tenth
percentile was -0.107, but two treatments lost dose through lane changes and
four full-dose treatments lost arrival or merge-discharge service. The original
clean 2000 cohort therefore did not replicate as an all-treatment operating
domain.

Reward/effect correlation remains high, but individual-agent reward accumulation
does not exactly equal the 60-second paired effect because treatment changes how
long the selected vehicle remains an active RL agent. This produces wrong-sign
credit only for small helpful effects in the inspected 2000 cohort; all effects
with magnitude above 0.06 had the correct sign. This must be handled explicitly
before long RL training, not by reward-weight searching.

## Placement diagnosis

A seven-state, two-candidate development screen compared the closest and second
safe runway candidates under identical p30 control:

- second candidate better in four states, tied in two and worse in one;
- closest candidate: one helpful, two neutral and four harmful;
- second candidate: one helpful, four neutral and two harmful;
- every branch delivered 20/20 steps;
- one second-candidate branch had a positive safety effect but lost three
  arrivals while retaining merge discharge and near-zero delay change.

Candidate-two predicted dose was usually 21--29 seconds versus 16--25 for the
closest candidate. A full-duration runway margin can reduce short-dose risk,
but it does not eliminate harmful full-dose p30 outcomes. Placement alone is
not a complete fix.

## p50 projection feasibility diagnosis

A diagnostic-only preview on the already opened 2000 veh/h feature-validation
seeds 4121--4132 evaluated p30, p50 and p70 through the unchanged analytical
controller and hard projection without applying p50 to SUMO. All 12 decisions
were rediscovered and the percentile ordering was exact.

- p50 differed from p30 by a median 0.312 m/s in terminal reference and by at
  least 0.10 m/s in 66.7% of decisions;
- the unprojected desired acceleration differed by a median 0.111 m/s2 and by
  at least 0.05 m/s2 in 66.7% of decisions;
- the first projected acceleration differed by at least 0.05 m/s2 in only
  16.7% of decisions, with a zero median difference;
- p50 was frequently identical or close to p70 because the disturbance speed
  distribution itself was compressed in several states.

The p50 hypothesis is therefore not a redundant reference definition, but its
first command is commonly hidden by the existing jerk, comfort and IDM-follow-
speed constraints. This preview neither passes nor rejects p50 as a treatment:
the targets can diverge over a 20-second option, which requires a paired SUMO
outcome screen. The artifacts are
`runs/eval/idm_p50_projection_preview_m2000_v1` and
`runs/eval/idm_p50_projection_preview_m2000_analysis_v1/summary.json`.

A subsequent full-option shadow diagnostic replayed the unchanged p30 arm on
the same 12 first-risk-onset decisions and evaluated p50 on every observed p30
state without sending the shadow command to SUMO. Coverage was 221/221 command
steps. Desired p50 acceleration differed from p30 by at least 0.05 m/s2 on
74.7% of steps; the hard-projected command differed on 33.0%. Six of twelve
cases had projected differences on at least 30% of their command steps. Thus
p50 is mechanically distinct over a complete option even though many individual
steps remain shield-limited. Because all states still come from the p30
trajectory, this is not a p50 outcome estimate. The complete artifact is
`runs/eval/idm_p50_full_option_shadow_m2000_analysis_v2/summary.json`.

## Predicted-dose threshold sensitivity

A post-hoc sensitivity analysis reused all 90 fresh paired opportunities at
1900--2100 veh/h. Raising the predicted-dose admission threshold from 16 to 20
seconds did not repair fixed p30:

- opportunity coverage fell from 100% to 72.2%;
- contract-qualified fraction rose from 87.8% to 90.8%;
- qualified harmful fraction changed only from 20.3% to 18.6%;
- mean qualified effect changed from +0.0409 to +0.0425 reference units;
- the qualified tenth percentile changed from -0.0766 to -0.0727;
- five helpful, ten neutral and ten harmful opportunities were excluded.

The result was not consistent by configured demand. At 1900, the higher
threshold removed four helpful cases and reduced mean benefit. At 2000 and
2100 it removed several service failures, but substantial qualified harm
remained. At 2200 it was actively counterproductive: coverage fell to 58.3%,
qualified harmful fraction rose from 38.1% to 50.0%, and mean effect fell from
-0.0159 to -0.0564. This further supports treating the highest-loading cell as
residual-risk evidence for later wall escalation rather than strengthening or
selectively extending decentralized damping.

Predicted duration is useful as a feasibility diagnostic, but 20 seconds is not
a universal benefit gate and should not be confounded with the first p50
treatment comparison. The reproducible artifacts are
`runs/eval/idm_decentralized_p30_lower_dose_threshold_diagnosis_v1` and the
four demand-specific `idm_decentralized_p30_m*_dose_threshold_diagnosis_v1`
directories.

## Proposed next scientific decision

Do not strengthen braking and do not train the chance gate. The remaining
physically motivated action hypothesis is a p50 terminal-speed ceiling:

- p70 was previously too weak;
- p30 can create excessive speed differential and delayed safety/service loss;
- p50 changes the terminal target without increasing braking-rate authority;
- every hard projection and smooth-release rule remains unchanged.

The smallest informative next experiment is now a development-only matched
p30-versus-p50 screen at the unchanged 16-second predicted-dose admission,
using already-inspected seeds first. Holding admission fixed isolates terminal-
target strength; the 20-second threshold remains a reported sensitivity, not a
second treatment factor. Only a passing p50 arm may proceed to broader
development analysis. Changing the terminal percentile alters the scientific
treatment definition and requires explicit approval before implementation.

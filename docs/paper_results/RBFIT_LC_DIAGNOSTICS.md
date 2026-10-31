# LC fit diagnostics — raw physical decisions versus discrepancy emissions

All numbers are conditional later native FIT969400–969431,300..360s;
626 recorded changes/258,691at-risk intervals. No labels, new simulation or
controller-effect outcomes select the fitted settings. Immutable per-world
counts/independent likelihood checks:artifacts/rbfit/*/VERIFIED.json.

## Completed old-source candidates (LC811cd7e8/snapshotv1)
- Source-default physical LC proposes12,095changes,239correct at the recorded
  destination/time:38.2%event recall,2.0%precision, eventBrier.047327.
- Directional v3 physical driver mixture:7,811expected changes,193expected
  exact matches. Directional scoring fixes Boolean wrong-destination matches.
- Source-domain boundary v4:5,444.5expected changes,178.2expected exact matches;
  raw eventBrier.007423. Its emission Brier.002374 is NOT a runtime LC score.
- Raw-physical least-squares v5:population516.8expected changes,54.7expected
  matches, eventBrier.002274; driver posterior410.2changes/52.8matches,
  eventBrier.002264 versus source-lane-rate baseline.002412.
- Hence v5 underpredicts the total626events; expected exact recall8.4%, precision
  12.9%. A better rare-event Brier alone does not establish realistic lane timing,
  merge service or congestion clearing. Raw log loss.533vs baseline.01666 also
  exposes unsupported recorded decisions; no smoothing/runtime noise is added.

## What was learned / remaining structural gap
- V5 solves the convex observable one-hot decision least-squares problem over
  the SAME209joint physical profiles,44active; no additional LC evaluation.
  Driver posterior is Bayes under a symmetric categorical discrepancy likelihood,
  inverse-temperature.33104 learned from causal prefix240..270→270..300 RAW Brier.
- The modal profile has strategic.19785/cooperative.72842/speedGain.21791/
  keepRight0/assertive.14238. These are fitted settings, not hidden-type recovery.
  Conservative secure-gap settings can compensate for a proposal-model error;
  this experiment does not identify the true causal driver distribution.
- Emission probabilities are inference discrepancy only, NEVER runtime event
  noise. Their poor v5 categorical score is distinct from raw physical scores.
- Latest LANE0a1e7e93/snapshotv2 source refit remains separately running; do not
  conflate its source-coordinate contract with these completed old-source scores.
- **NOT ADOPTED:** INT/EVAL must test free-running traffic/event distributions
  and expected action shifts. IDM shrinkage delivery remains the preferred CF input.

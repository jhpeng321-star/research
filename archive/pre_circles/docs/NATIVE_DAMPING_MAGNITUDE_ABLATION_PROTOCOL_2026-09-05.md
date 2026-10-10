# Two-case diagnostic of near-zero nominal-controller engagement

Status: both diagnostic cases completed. Outcomes and saved-action verification
are in `NATIVE_DAMPING_MAGNITUDE_ABLATION_RESULTS_2026-09-05.md`. The protocol
below retains the comparison and interpretation specified before those outcomes.

This bounded addition follows the owner's supplied diagnosis of the episode-20
failure. It does not change the live three-lineage campaign, its checkpoint
selection, or its acceptance criteria. It uses the two remaining full-rollout
slots under the conservative 100-rollout development ceiling: 98 already
accounted for by previous and registered work, plus these two equals 100.
No further development rollout capacity is implied.

## Question and intervention

Does the episode-20 policy still produce substantial traffic effects when
almost all of its learned correction magnitude is removed, while retaining its
sign-dependent control logic? The unchanged controller routes exact zero to
native SUMO and nonzero through nominal IDM, projection and speed commands.
The nominal calculation is not SUMO's exact native next acceleration.

The diagnostic actor is `clip(pi_20(local_history), -1e-6, +1e-6)` m/s².
This cap is above the frozen native-zero tolerance of 1e-8. It retains the
base actor's sign and zero/nonzero classification at the **same input** and
never amplifies its magnitude. The actor runs on each diagnostic trajectory's
own causal local history, including its own previous command. It does not
replay the original trajectory's future actions, observations or activation
sequence. The rest of the collector, nominal equation, safety projection and
actuator remains byte-identical to the campaign.

A constant positive epsilon would change learned signs and the projection's
sign-reversal decisions. Exact zero would simply reproduce native passthrough.
Neither is this magnitude ablation. Its deterministic log-probability entries
are zero storage placeholders, never valid PPO training likelihoods.

## Fixed cases and compute

Use the frozen optimizer-71 episode-20 checkpoint, SHA-256
`9aef1dae1c2fc0b627476d1287693f86cbebb3568fddee6f01098df9d3f48b60`.
Cases are 911001/2000 (previously helpful) and 911003/2200 (previously harmful).
These cases were selected after seeing outcomes. They are exposed diagnostic
cases, not a fresh validation cohort or a representative estimate of prevalence.

Reuse each case's existing native and learned rollouts from the completed
episode-20 comparison. Add exactly one ablated-policy rollout per case, at the
same 2,400-second horizon and fixed requested populations: 4,800 additional
simulator seconds. The new diagnostic outputs are separate from checkpoint
selection and cannot be ingested as training data. The two runs execute
sequentially in one diagnostic process alongside the three training lineages.
No new seed, reward fit, threshold tuning or acceptance data is involved.

Runner: `experiments/idm/probe_damping_magnitude_ablation.py`.
Output: `runs/probe/native_damping_magnitude_ablation_o71_e020_v1/`.
The runner records identities, source hashes, input-comparison hashes, epsilon,
case selection and allocated budget before simulation. Missing/mismatched
baselines or an incomplete started run cause failure rather than a silent rerun.
The source snapshot and registration are immutable while this diagnostic runs.

## Interpretation fixed before diagnostic outcomes

Report braking, squared-acceleration effort, full requested-trip system time,
source waiting, TTC exposure, completions and hard events against both existing
arms. Also report command counts and projection/fallback reasons. A similar
large effect with almost no requested residual would support investigating the
engaged nominal/projection/actuator path in these two cases. A much smaller
effect would make the original learned magnitude consequential in these cases.
Opposite or sharply different outcomes warrant reporting interaction and
sensitivity rather than assigning a percentage of harm to one component.

Projection and the closed-loop trajectory can change when magnitude changes.
This experiment therefore estimates the causal effect of replacing the policy
with its magnitude ablation under the fixed simulation cases. It cannot uniquely
decompose nominal-model error, timing, projection, learned signs and nonlinear
traffic dynamics. It cannot prove observation insufficiency, validate learning
in general, or establish a successor controller's performance.

The original episode-20 result stays failed regardless of these findings.
Training continues with its frozen settings. Any controller or reward redesign
requires a versioned contract and validation; this diagnostic does not silently
introduce one. Three focused offline tests passed, checking sign/zero retention,
invalid epsilon rejection, and feasible nominal engagement at tiny residuals;
Ruff passed. No software test is presented as a traffic-efficacy result.

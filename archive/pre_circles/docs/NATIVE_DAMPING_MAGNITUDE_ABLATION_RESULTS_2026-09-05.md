# Completed magnitude-ablation diagnosis

Both registered diagnostic rollouts completed successfully. Substantial traffic
effects remain when the frozen actor's requested correction is capped at
±0.000001 m/s². Removing the original magnitude also changes the outcomes
substantially. The result supports an action-interface concern and magnitude
sensitivity together; it does not identify one exclusive cause of failure.

Protocol: `NATIVE_DAMPING_MAGNITUDE_ABLATION_PROTOCOL_2026-09-05.md`.
Raw outcomes and audit:
`runs/probe/native_damping_magnitude_ablation_o71_e020_v1/summary.json` and
`audit.json`. The diagnostic session 20771 exited with code zero; its PID 32568
is terminal. Do not restart it. The parallel training coordinator remains the
separate session 96488.

## Verified evidence

The read-only audit checked frozen scientific sources, diagnostic source,
cached comparison hashes, paired identities, all 2,400 interval-cost totals
per case, and recalculated the reported metrics. The frozen capped actor
exactly reproduced **55,122** saved actions using original per-tick batch
shapes. The maximum absolute requested correction was
9.999999974752427e-7 m/s². No raw action entered the native-zero tolerance.
Both cases completed all requested trips. The audit generated no new simulation.

All percentages below use each case's native IDM denominator. Composite
improvement is the equal mean of relative severe-braking and squared-
acceleration-effort improvements; effort is not fuel consumption. These two
cases were selected after their previous outcomes were known.

## Previously helpful case: 911001 / 2000 veh/h

The original learned policy improved the composite by **5.17%** and system
time by **0.78 s/request**. With magnitude capped, composite improvement was
**9.76%** and system-time improvement **1.10 s/request**. The capped policy
reduced braking by 12.37%, effort by 7.16%, and TTC<3 s exposure by 60
vehicle-seconds relative to native. Its TTC exposure was 7 vehicle-seconds
higher than the original learned policy, so every metric did not improve.

Speed commands were issued on **83.43%** of capped-policy decisions versus
82.94% for the original learned policy. The beneficial effect did not vanish
with almost all requested correction magnitude removed.

## Previously harmful case: 911003 / 2200 veh/h

The original learned policy worsened the composite by **24.79%**, increased
system time by **22.40 s/request**, and increased TTC<3 s exposure by **517.5
vehicle-seconds** relative to native.

With magnitude capped, the composite still worsened **12.30%**, system time
increased **8.78 s/request**, and TTC exposure increased **199 vehicle-seconds**.
Braking worsened 14.15% and effort worsened 10.44%. This remains harmful under
the original system-time, TTC and harmful-tail bounds. No added collision or
teleport events occurred relative to native, and all 1,252 trips completed;
those facts do not erase the failures.

Speed-command frequency was **78.76%**, versus 79.25% for the original policy.
Aggregate engagement frequency was similar, but its exact vehicle/time sequence
and trajectory were not held fixed. Compared directly with the original
learned policy, capping improved system time by 13.63 s/request and reduced TTC
exposure by 318.5 vehicle-seconds. Original residual magnitude therefore had
consequential effects in this case even though it was below 0.01 m/s².

## Interpretation and next engineering decision

These are two causal comparisons of complete closed-loop policies on fixed
simulation requests. They establish that the capped policy is not equivalent
to native IDM: benefits and harm remain despite microunit requested residuals
and similar aggregate command frequencies. They also establish that removing
the original correction magnitude changes outcomes. They do not justify
assigning a percentage of harm to nominal-model error or calling the remaining
effect purely analytical. The capped actor still uses learned signs; safety
projection can change targets and fallback decisions; its evolving local
history also changes future signs and timing.

The practical implication is to stop interpreting the current residual as a
bounded perturbation of SUMO native acceleration. It is a residual around an
engaged nominal model, with a separate exact-zero native-handback mode. This
distinction must remain explicit in the method and in any future controller
redesign. A future native-relative interface should make native restoration
and small-action behavior verifiable, rather than relying on the small numeric
size of network outputs. An explicit nominal-path comparator is also needed
before attributing benefits specifically to learned correction magnitude.

The current checkpoint remains ineligible. The ablation is not a substitute
checkpoint: it is a post-outcome diagnostic and is harmful in the 2200 case.
The result does not prove that observations are insufficient or that no later
checkpoint can learn useful feedback. It also does not prove that more training
will repair this interface or the separate mismatch between reward and all
acceptance constraints. In particular, TTC exposure has no direct reward term,
and the separate source-wait limit is not enforced by total system time alone.

The already running three-lineage campaign continues unchanged through its
registered assessments. Judge later checkpoints on their full physical
comparisons, retain this diagnostic as mechanism evidence, and do not explain
away failed criteria using training reward or all-trip completion. If the
campaign yields no eligible candidate, prioritize a versioned action-interface
and objective-alignment repair over unchanged additional training or a new
observation classifier sweep. No such successor or extra simulation is launched
by this report.

The two completed diagnostic rollouts use the last two slots in the conservative
100-rollout development ceiling. The remaining registered campaign evaluations
are already included in that accounting. Acceptance data, walls and RSU
predictor data remain unopened, ungenerated or paused according to their
existing prerequisites. No accepted damping checkpoint is claimed.

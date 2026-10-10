# IDM Phase-5 dose-response pilot diagnosis

Status: completed development-only stopping audit. This artifact does not
authorize PPO training or an independent learned-controller claim.

## Frozen pilot

- Demand: 2200 veh/h.
- Development seeds: 4107, 4109, 4110 and 4111.
- Deterministic landmarks: wave onset and severe-braking onset, seven cases.
- Arms: zero residual and `-0.10` / `+0.10 m/s²` for 20 controller steps.
- Safety/wave horizon: 40 seconds. Service horizon: 60 seconds.
- Frozen five-second filter, deterministic lane-role gate, handoff logic,
  analytical IDM controller, front/rear safety projection, acceleration/jerk
  limits, reward, wall treatment and seed partitions were unchanged.
- Output: `runs/eval/idm_phase5_action_local_dose_response_v1`.

The runner is resumable. Landmark and branch artifacts are atomic and bound to
protocol hash
`405cc643307ad2eb63a65edba8dadbd9b29d0f4d1b770407b8d19889c853d8b8`.

## Integrity

All branches preserved the discovered decision state, completed the declared
horizon, requested treatment only for the frozen lane-role and produced no
collision or teleport observation. Zero-request filter-decay tails are recorded
separately from unexpected non-role requests. The corrected overall dose gate
fails: only 6 of 14 treatment arms (42.9%) received at least 16 of the requested
20 lane-role steps.

## Dose chain

Across all 14 arms, the mean requested exposure was 2.000 residual-seconds,
1.350 was delivered while the role remained eligible, 0.748 remained after the
filter during the pulse, and 0.243 was causally executed relative to the paired
zero branch. Pair matching covered 96.3% of comparable command rows.

Among the six dose-qualified arms, 94.2% of requested steps were delivered,
the filter retained 59.9% of delivered exposure, and the IDM/safety projection
transmitted 34.2% of the paired filtered exposure. Mean paired executed exposure
was 0.390 residual-seconds, only 19.5% of the intended 2.000 residual-seconds.

The first loss is intermittent lane-role eligibility in eight arms. In the six
qualified arms, the first large continuous attenuation is the five-second
filter, followed by a stronger projection-limited loss. The most frequent v4
projection reasons remain IDM follow-speed and comfortable-acceleration limits;
the pilot also records jerk projection.

## Physical outcomes

After excluding insufficient-dose arms, four arms were neutral and two were
harmful under the frozen ±0.02 replay deadband and hard constraints. Helpful
arms: zero. The four neutral arms came from seed 4109; both harmful arms came
from the wave-onset case in seed 4110. Zero was the oracle action in five of
seven cases. The two nonzero oracle selections were improvements smaller than
the helpful deadband, not successful labels.

The seed-4110 negative residual reduced acceleration energy but added enough
severe braking to make the combined safety effect harmful. Spatial diagnostics
found acceleration-energy separation above 0.02 in 2/14 absorber responses,
2/14 immediate-follower responses, 4/14 influence-region responses and 11/14
lane/corridor responses. The action can propagate, so broad endpoints alone do
not explain the failure; the observed propagation was neutral or traded wave
energy against severe braking.

## Frozen decision

The pilot fails the predeclared gates for another full action-local diagnostic
and PPO authorization:

1. At least 80% dose coverage did not hold.
2. No helpful action-state pair was observed.
3. Beneficial and nonbeneficial outcomes did not span independent seeds.
4. Executed dose did not show a beneficial dose-response.
5. The result is not solely endpoint dilution: measurable propagated effects
   included physical harm.

Do not begin PPO. A technically trained policy would be expected to learn zero
residual or exploit noise and would not constitute a working IDM damping
controller. Under the predeclared stopping rule, freeze the analytical IDM
damping controller, preserve this negative residual result, run the independent
analytical-versus-native-IDM validation, and use unresolved safe cases as
headroom for the damping-preserving moving-wall escalation.

The optional `+0.20 m/s²` arm is not justified merely to search for a positive
label: `+0.10` already produced measurable corridor propagation without a
helpful case, and stronger positive authority would change dose while retaining
the same limiting projection and safety tradeoff. It should be reconsidered
only with a new, predeclared physical hypothesis—not as outcome tuning.

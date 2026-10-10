# Frozen P30 Projected-Release Treatment

Status: frozen development treatment before implementation and replay. This
protocol does not authorize RL, predictor training, moving-wall experiments,
new seeds, or paper-scale data.

## Scientific question

The unchanged 15-case P30 phase audit found six helpful, five neutral and four
harmful final outcomes. None of the four harmful cases was harmful at the end
of the 20-command active phase. Two were helpful at that point and later
reversed. Harmful cases added most of their severe-braking exposure during the
legacy release or post-release response. Only 179 of 8,190 spatial front-fit
slots were valid, so the present wave-front estimator cannot be an online gate.
All 15 legacy branches also handed control to native IDM on the first release
invocation; none executed a controlled release command. Eight of 15 then
realized more than +0.5 m/s2 and the observed maximum was +1.50 m/s2 despite
near-zero hold-end acceleration in 14 cases. This is the concrete mechanism the
new handoff contract tests.

The smallest causal follow-up is therefore a release-mechanism isolation test:
keep candidate, admission, active command and dose identical to frozen P30 and
change only the return to native IDM. An early efficacy/benefit gate is not part
of this treatment because the diagnostic showed that early response did not
identify final benefit.

## Treatment identity

The candidate action set is
`activation_speed_ceiling_p30_settle_release_v1`.

The earlier linear-blend v1 was used only for two targeted development smokes.
It repaired one reversal and improved the other from -0.138 to -0.054 without
making it nonharmful. It is retained as a diagnostic artifact and is not the
frozen 15-case pilot treatment. A smoothstep-only v2 produced the same difficult
case trajectory because its leader follow-speed ceiling was already binding.
A separate 30-command P30 diagnostic made that case neutral (+0.0149) without
meaningful delay or service loss, but the second frozen reversal lacked enough
predicted runway for a 30-command active option. These results motivate the
fixed five-step settle plus five-step release below: it tests duration headroom
without changing the 20-command admission contract.

Its nonzero action has the same mode index, name, family and value as P30:

- index: 1;
- mode: `speed_ceiling`;
- family: `speed_ceiling`;
- value: 0.0;
- terminal reference: disturbance P30;
- requested active duration: 20 one-second commands.

Candidate enumeration, secure-gap admission, runway ranking, target identity,
contention, P30 reference filter, analytical acceleration, comfort bounds, jerk
bound, front/follower TTC projection and SUMO safe speed mode remain unchanged.
The original `activation_speed_ceiling_v1` implementation and artifacts remain
the immutable comparator.

## Lifecycle

```text
inactive -> P30_active (20 commands) -> projected_release -> native_IDM
                                      -> vehicle_exit
```

There is no probe or conditional hold. The treatment begins to differ from P30
only after active command 20.

`getSpeedWithoutTraCI(vehicle_id)` is diagnostic only because the first smoke
showed that it can echo the active TraCI speed override. The release instead
constructs a treatment-independent one-step native proxy:

```text
v_free_next   = min(v_allowed, v + a_native_max * dt)
v_native_next = min(v_free_next, v_follow)  # omit v_follow without a leader
a_native      = (v_native_next - v) / dt
lambda_r      = 0                                for r = 1,...,5
q_r           = (r - 5) / 5                     for r = 6,...,10
lambda_r      = 3 * q_r^2 - 2 * q_r^3           for r = 6,...,10
a_blend       = lambda_r * a_native
```

for release command `r=1,...,10`. Commands 1--5 settle at zero requested
acceleration under the full shield; commands 6--10 follow a zero-slope
smoothstep to the native proxy. This is a deterministic release shape, not an
online efficacy gate. The blended command then passes through the
same final projection used by active control, except that positive release
acceleration may rise to the vehicle's declared native acceleration limit;
P30's active positive-acceleration cap remains unchanged. The common 0.5 m/s3
jerk limit is still authoritative. The projection also includes:

1. comfort and jerk bounds;
2. native SUMO follow-speed and front-TTC ceilings;
3. the existing follower-TTC floor;
4. a release-only rear secure-gap speed floor.

For a candidate next speed `v_next`, SUMO first predicts the follower's next
speed `v_follower_next` using that follower's own car-following model. The
one-step rear gap is then estimated as

```text
g_next = g_rear
       + 0.5 * ((v_ego + v_next)
              - (v_follower + v_follower_next)) * dt.
```

Bounded bisection finds the smallest `v_next` for which (1) `g_next` exceeds
SUMO's secure gap plus 5 m, (2) the predicted follower command does not demand
more than its declared deceleration capability and (3) the existing four-second
follower-TTC floor is satisfied. Front safety remains dominant if front and
rear constraints conflict. Follower deceleration and reaction-time values are
logged even though SUMO's own follow-speed and secure-gap APIs remain the sole
rear-response model in this v1 treatment.

If follower follow-speed or secure-gap data are unavailable, the release asks
for the maximum comfortable next speed, the existing TTC floor and SUMO safe
speed mode remain active, the degraded condition is logged, and the pilot
cannot count that handoff as fully verified.

Release commands 9 and 10 must both satisfy all of the following:

- absolute projected/native-proxy target-speed difference no greater than 0.5
  m/s;
- absolute projected/native acceleration difference no greater than 0.25
  m/s2;
- no front/rear projection conflict and a feasible rear response.

The release is exactly 10 commands, matching the legacy release budget and
leaving 30 native-IDM response steps inside the frozen 60-step horizon. Native
IDM resumes on the next invocation. Failure of the two-step command-matching
gate is logged as an unmatched handoff and hard-fails the treatment; it is not
silently extended or tuned. A safe vehicle exit ends the lifecycle. The same
lane is blocked from reactivation only while release is active; no extra
cooldown is introduced in this mechanism-isolation treatment.

## Required diagnostics

Every release row records treatment instance, phase, age, lane and vehicle;
current, reported uncommanded and native-proxy speeds; blend fraction; desired,
projected and realized command; projection
bounds/reasons/conflict; leader and follower state; TTC and secure-gap floors;
rear data availability; convergence count; handoff/exit/timeout state; and all
release reasons. Active P30 rows retain their existing schema so exact command
identity can be checked.

## Frozen matched pilot

Use only the already inspected 2,000 veh/h development cases:

- recurrence 1: 4123/t61, 4127/t76, 4129/t354, 4130/t55, 4131/t382,
  4132/t49;
- recurrence 2: 4123/t146, 4127/t200, 4129/t436, 4130/t192, 4132/t291;
- recurrence 3: 4127/t289, 4129/t528, 4130/t331, 4132/t574.

Matched arms are native IDM, exact native replay, frozen P30 and the new
projected-release treatment at the identical decision and target. Seeds 4123,
4127, 4129, 4130, 4131 and 4132 remain development-only and excluded from later
prospective evidence.

Before outcome interpretation, all of these integrity gates must pass:

- exact native-versus-native replay identity within the frozen deadband;
- identical predecision raw traffic state, decision step, lane and target;
- exactly 20 target-only active commands in both treatment arms;
- exact P30/new-treatment command and response identity through active command
  20;
- no out-of-target command;
- every release command uses the full projection;
- exactly 10 projected release commands followed by native IDM;
- no unmatched handoff, missing rear shield when a follower exists, projection
  conflict, collision or teleport.

The treatment passes the development mechanism gate only if:

- all 15 cases pass every safety/service and handoff constraint;
- target/immediate-follower low-TTC, severe-braking and hard-braking onsets do
  not increase relative to frozen P30 in any qualified case;
- aggregate corridor severe-braking and low-TTC exposure are no worse than
  P30;
- both predeclared reversal cases, 4127/t200 and 4129/t354, improve over P30 by
  more than the frozen 0.02 deadband and finish nonharmful (score at least
  -0.02);
- harmful qualified outcomes fall from four to at most two, with no new harmful
  result among the 11 previously helpful or neutral cases;
- at least two independent seeds improve over P30 by more than 0.02;
- median new-minus-P30 score is nonnegative and aggregate improvement is
  positive;
- at least four of the six existing helpful cases remain helpful and all six
  remain nonharmful;
- delay-rate increase is no more than 1.0, arrival loss no more than one and
  merge-discharge loss no more than one per case, matching the frozen bounded
  harm contract.

These are development gates, not publication claims. If identity or release
dose fails, repair implementation. If identity and release pass but outcome
harm is not reduced, reject this release-only mechanism and design a separately
frozen front-feedback action. Do not add seeds or train a policy to hide a
failed physical treatment.

# Frozen IDM-to-P30 Projected-Blend Authority Protocol

Status: frozen on 2026-08-29 before implementation or blended-treatment runs.
Development only. It cannot authorize RL directly.

## Scientific question

Frozen P30 has real but heterogeneous authority: among the 15 phase-aligned
development cases it is helpful in six, neutral in five and harmful in four.
The modified-FollowerStopper ceiling failed because it could only strengthen
P30 and was inactive in 13/14 treatment-naive cases. This screen asks whether a
graded action between exact native IDM abstention and full P30 can retain useful
damping while reducing P30's harmful tail.

High-density insufficiency remains the later moving wall's responsibility. This
action does not command more strongly than P30.

## Frozen action

Action-set identity:

`activation_idm_p30_projected_blend_v1`

Action 0 is true abstention: no bid, no `setSpeed`, and exact native IDM. Active
actions are fixed for the full 20-command option:

```text
index 1 -> lambda = 0.25
index 2 -> lambda = 0.50
index 3 -> lambda = 0.75
index 4 -> lambda = 1.00
```

`lambda=1` must reproduce frozen P30 exactly. Malformed actions abstain.

`getSpeedWithoutTraCI` is diagnostic only because it can echo a TraCI override.
The active native endpoint uses the existing treatment-independent one-step
proxy:

```text
v_free_next  = min(v_allowed, v + a_vehicle_max * dt)
v_proxy_next = min(v_free_next, v_follow)  # omit v_follow without a leader
a_proxy_raw  = (v_proxy_next - v) / dt
```

The P30 raw endpoint is unchanged:

```text
a_p30_raw = min(a_analytical_p30, (U_p30 - v) / dt)
```

Let `Pi_t` be the existing deterministic projection from the same current
state and prior applied acceleration. The delivered endpoints and blend are:

```text
A_proxy = Pi_t(a_proxy_raw)
A_p30   = Pi_t(a_p30_raw)
A_mix   = (1 - lambda) * A_proxy + lambda * A_p30
A_cmd   = Pi_t(A_mix)
```

Blending projected endpoints prevents the common jerk/comfort interval from
collapsing all raw strengths to the same command. The final projection remains
authoritative and must be idempotent in every nonconflict state.

Candidate enumeration, P30 terminal rear admission, runway ranking,
contention, target, filtered P30 reference, 20-command latching, safety
projection and legacy controlled release remain unchanged. The lane is blocked
while a blend member releases, preventing simultaneous same-lane release and a
new option. Projection conflict is a hard treatment abort followed by legacy
release; it is never a normal observation.

## Frozen 15-case authority screen

Only the already inspected P30 decisions are used:

- recurrence 1: 4123/t61, 4127/t76, 4129/t354, 4130/t55, 4131/t382,
  4132/t49;
- recurrence 2: 4123/t146, 4127/t200, 4129/t436, 4130/t192,
  4132/t291;
- recurrence 3: 4127/t289, 4129/t528, 4130/t331, 4132/t574.

Sources are the three
`runs/eval/idm_p30_p50_m2000_l{1,2,3}_p30_development_v1` directories. Frozen
P30 comparators are the three
`runs/probe/idm_p30_phase_spatial_m2000_l{1,2,3}_development_v1` directories.
All six seeds remain excluded from prospective evaluation.

Each case has exact native, native replay and four active blend branches. No
strength, threshold, duration, target, release parameter or gate may be tuned
after the first branch is inspected.

## Identity, mechanics and safety gates

All must pass:

1. exact 15-case set, recurrence, decision step, target, raw predecision
   context and non-classic observation identity;
2. exact native-versus-native replay;
3. `lambda=1` matches frozen P30 at every command within 1e-9 m/s2 and has an
   outcome difference no larger than replay noise;
4. every active arm completes 60 steps, delivers all 20 target-only commands,
   and has no unexpected command, treatment abort or projection conflict;
5. zero added collisions or teleports;
6. low-TTC increase at most two steps, delay increase at most 1 s/vehicle, and
   arrivals/discharge no worse than -1 relative to native;
7. stable decision cohort and immediate follower have minimum TTC at least
   4.0 s and acceleration no lower than -4.5 m/s2 in every arm;
8. each strength's aggregate exposure below -3.0 m/s2 is no greater than P30;
9. projected acceleration, jerk, allowed speed, front and rear constraints all
   remain satisfied.

## Action-credit and oracle-headroom gates

A case/strength is command-distinct from P30 when at least two shared commands
differ by at least 0.05 m/s2. At least two intermediate strengths must be
distinct in at least 5/15 cases, and at least 5/15 cases must realize three or
more distinct projected strength levels spanning at least 0.10 m/s2 on two or
more commands. Any redundant strength is reported and pruned before learning.

For every case, compute an active oracle over only safety/service-admissible
strengths and a deployable oracle over admissible strengths plus abstention.
The blend-specific incremental gain is the deployable-oracle effect minus the
better of abstention and full P30.

All headroom gates must pass:

1. blend-specific gain exceeds the 0.02 replay deadband in at least three cases
   spanning at least two seeds;
2. actionable winners include at least two distinct strengths and span at
   least three seeds;
3. mean deployable-oracle effect exceeds the best constant action by more than
   0.02;
4. at least two of P30's four harmful cases become nonharmful under an
   intermediate active strength, and at least one becomes helpful;
5. active-oracle harmful count is at most three, its tenth percentile is no
   lower than P30, at most three cases fall below -0.05, and none falls below
   -0.20;
6. helpful active-oracle outcomes span at least three seeds and nonbeneficial
   outcomes span at least three seeds.

The oracle is an optimistic action-space diagnostic, not a policy result.
Passing authorizes only the observation and option-reward audit.

## Contingent observation/reward cohort

If and only if the 15-case authority screen passes, generate the same frozen
four active branches for all 30 already recorded P30 decisions across seeds
4121--4132 in the three source directories above. These are development seeds,
not prospective evidence. This expansion is required because six seeds cannot
support the frozen leave-one-seed-out sufficiency minimum.

The observation audit treats each decision as one group and every strength as
an action-conditioned row. It requires at least eight seeds, 24 qualified
nonneutral option-versus-abstain contrasts, 24 informative option pairs, and
both active-better and abstain-better outcomes in at least three seeds. If
these minima are absent, the result is `INSUFFICIENT`, not a pass.

The primary fixed-L2 utility model uses leave-one-seed-out evaluation. It
compares legacy deployable observations against one frozen repair containing
named leader/follower slots, front/rear secure-gap margins, runway, predicted
dose, terminal rear headroom, target-speed difference, predecision projection
margin and option identity. Seed, demand identifier, recurrence, time, realized
dose and all postdecision values are forbidden.

Observation pass requires grouped pairwise AUROC at least 0.70, balanced
accuracy at least 0.65, AUROC improvement at least 0.10 over the legacy actor,
at least 30% activation coverage and 50% oracle-helpful recall, median regret
at most 0.02, p90 regret at most 0.06, mean regret at least 10% below the best
constant, and positive improvement after removing any seed or recurrence.

The option reward is computed once after the fixed 60-second corridor horizon,
not from variable-lifetime agent returns. For qualified branches it is a
monotone bounded transform of the frozen severe-braking and acceleration-energy
composite. Hard safety, completion or service failure overrides reward to -2.
Reward pass requires exact replay equality, every qualified nonneutral sign to
agree, every option pair separated by at least 0.06 to be ranked correctly,
correlation at least 0.80, neutral reward difference at most 0.02, no component
above 80% of pooled absolute mass, and every failed branch below every
admissible branch in its decision.

Even a complete pass authorizes only implementation of the option-level
training wrapper and repaired encoder/critic. It does not itself authorize a
long RL run.

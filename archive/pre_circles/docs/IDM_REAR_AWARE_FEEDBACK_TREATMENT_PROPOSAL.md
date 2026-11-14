# Proposed Rear-Aware IDM Feedback Pulse

Status: scientific treatment proposal only. It is not implemented or approved
for training. Implementing it changes the decentralized treatment definition
and requires an explicit user decision.

## Purpose

P30 and P50 show that a fixed terminal-speed percentile has sparse authority,
cannot adapt to heterogeneous wave response and can remain harmful even after a
full 20-second dose. The proposed replacement is a low-authority feedback pulse
that probes the traffic response, protects the immediate follower and releases
early when benefit is not emerging.

Native IDM remains the exact abstention arm and hard fallback. The existing
comfort, jerk, TTC, secure-gap, SUMO safe-speed and collision protections remain
authoritative.

## Candidate selection

For every disturbance, compute for each upstream CAV:

- usable runway to the detected shock;
- estimated shock-interception time from relative CAV/shock speed;
- front and rear secure-gap headroom;
- front and rear TTC/closing state;
- predicted time remaining before merge or lane exit.

Reject candidates with unsafe rear state, insufficient probe time or no
positive interception. Rank remaining candidates by an explicitly frozen score
that prefers timely shock interception, adequate runway and rear headroom.
Distance-to-merge alone must not select the absorber.

## Front feedback and rear protection

Use a FollowerStopper/Jam-Absorption-Driving-style target rather than a fixed
percentile ceiling. The front command is a bounded piecewise feedback law of
leader gap and relative speed: strong slowing is permitted only below a close
gap threshold, intermediate gaps interpolate toward leader/downstream speed,
and a sufficiently open gap returns toward the native IDM desired speed. P50
may remain a nominal upper reference, but it is not an unconditional terminal
target.

Compute a rear-protection acceleration floor from follower gap, follower
closing speed, follower deceleration capability and secure-gap margin. The
requested command is no more negative than this floor. The existing hard safety
projection is then applied last. If front damping demand conflicts with rear
safety, release toward native IDM instead of strengthening braking.

## Slow-in, conditional-hold, fast-out lifecycle

1. `probe` for 3--5 seconds with a jerk-limited ramp from native IDM.
2. Measure phase-local downstream wave/severe-braking trend, target projection
   survival, immediate-follower response, TTC and service indicators.
3. Enter `hold` only if wave response improves and rear/safety/service signals
   remain admissible.
4. Release early on absent response, worsening wave energy, rear compression,
   TTC deterioration, service loss, candidate lane change or lost authority.
5. Cap probe plus hold at 20 seconds.
6. `fast_out` smoothly but promptly returns to the exact native IDM command;
   no uncontrolled SUMO release is allowed.

Every phase logs requested and realized seconds, command deltas relative to
native IDM, projection reasons, follower state, wave response and release
reason. This separates probe harm, hold benefit and post-release effects.

## Proposed six-seed mechanism pilot

Use only already inspected development seeds 4123, 4127, 4129, 4130, 4131 and
4132 at the existing 2000-veh/h risk events. This set deliberately contains
previous helpful and harmful P30 responses and cannot support independent
claims.

Matched arms are:

1. native IDM abstention and exact replay;
2. frozen P30 at the same decision and target;
3. rear-aware feedback pulse at that same decision and target.

Before efficacy interpretation require exact predecision physical identity,
zero control commands, target-only treatment, no added collision/teleport, no
safety-projection bypass, and at least five of six feasible probes receiving
their declared 3--5-second dose. Phase traces must differ from native IDM above
replay noise.

Retain the treatment for broader development only if it reduces harmful
qualified activations relative to P30, produces meaningful benefit in at least
two seeds, produces nonbeneficial outcomes in at least two seeds, adds no TTC or
follower-braking failure, and does not hide discharge/service loss. These are
mechanism-development gates, not paper verification.

No observation model or RL training should be changed during this pilot. If the
feedback pulse still has weak or inconsistent effects, the honest pivot is a
conservative abstention-domain controller and later coordinated escalation—not
a larger activation classifier.


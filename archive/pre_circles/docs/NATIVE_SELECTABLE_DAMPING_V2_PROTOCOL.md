# Native-selectable damping v2.0 — bounded repair protocol

Owner instruction, 2026-09-05 23:02, supersedes the v1 campaign schedule.
The campaign is checkpoint-paused at optimizer 71/72/73 episodes 36/8/8;
124,800 optimized global seconds are preserved, with no incomplete successor
rollouts. See its `parallel_execution/checkpoint_safe_pause/summary.json`.
No checkpoint is accepted. Acceptance data stay sealed; wall training stays paused.

## Prior evidence and the omission

The v1 episode-20 evaluation failed: lower-loading braking worsened 4.43%,
effort 5.53%, and total system time 6.09 s/request. All deterministic corrections
were below 0.01 m/s2, yet overrides were issued on 81.16% of CAV decisions.
The two exposed microunit ablations still changed traffic substantially; the
2200 case still failed service/TTC. These results establish a discontinuous
control-interface concern, not observation insufficiency or a causal percentage
of harm attributable to the approximate IDM formula.

Historical `D:/research_v2/env/rewards.py:_control_cost` charged zero for mode 0
and 0.1 per selected active CAV decision. `env/actions.py` restored the cached
original tau, called `setSpeed(-1)`, and cleared actuator EMA state. The original
hardcoded tau=1.0 handback was a documented confound; do not repeat it.
This v2 controller never changes tau and clears command memory on release;
its causal ego-speed observation filter deliberately persists across modes.

Read-only audit of ten exposed old medium i60 probes (42–46, 56–60) verified
800 arrivals each and paired mean travel-time change -0.9535 s, 7/10 wins.
Saved TTC totals were 316 vs 1160 and collisions 17 vs 24, with historical metric
and model caveats. Source: `D:/research_v2/runs/eval/peek_v30m_talk/summary.json`
and corresponding `runs/probe/probe_v2_i60_medium_p010.seed*.json`.
This supports selective learned control in that old model, not causal proof
the cost produced its win or transfer to current IDM. Stronger flat cost v30
and health-gated v31 failed their exposed candidate comparisons. The July 29
postmortem also records reward saturation making native operation the optimum.

There is **no recorded scientific justification** for omitting native preference
from v1. Its new continuous residual head and physical team objective omitted
the earlier discrete action and cost. Zero-initialized output was not an adequate
replacement: a continuous stochastic policy almost never selects exact zero.

## One coherent controller

Action is `(native/active, continuous correction)`. Native has explicit
probability mass and always releases an owned override immediately with
`setSpeed(-1)`. Native leaves SUMO IDM, tau and speedMode untouched. Active zero
correction is explicitly controlled behavior and is never called native.

Active acceleration is a causal filtered-speed/gap reference plus a learned
correction in [-0.5,+0.3] m/s2. It is projected to the existing comfortable
[-1,+0.5] m/s2 and 0.5 m/s3 target-jerk bounds, observed front/rear TTC bounds,
speed limit and nonnegative speed. Native safety speedMode=31 remains enabled.
Actual acceleration can differ under SUMO safety; these are command bounds.
There is no approximation of native acceleration in this active interface.

The reference uses a 5 s ego-speed EMA, measured leader speed and bumper gap,
with a 2.5 m + 1.5 s reference gap, bounded +/-2 m/s gap correction and a 5 s
speed-response time. Small gaps blend toward leader speed; larger gaps allow
catch-up. A trainable binary head is initialized to prefer native in steady
traffic and active with measured relative-speed/filter deviations above
0.5 m/s, once five seconds of history exist. This is an initialization, not
a fixed health gate; both mode and continuous response can learn.

Native release starts a five-second re-engagement cooldown. It never forces
continued active control. The actor observes its own mode and cooldown, and
cooldown masks the active action with zero masked-action likelihood gradient.
Infeasible/stale measurements release to native; rejected requested intervention
is still charged, so unsuccessful commands do not become free actions.

Actor inputs: existing five 17-field local frames plus ego EMA, EMA-ready flag,
owned mode and cooldown (89 fields). Only ego state/map, nearest measured front
and rear within 100 m, and own controller memory enter the actor. No matched
native outcome, global health, downstream truth or future leader trajectory is
an online input. Ideal same-lane coverage, lane-boundary truncation and <=1 s
measurement age remain explicit simulator assumptions, not sensor validation.

Research adaptations: [Stern et al.](https://arxiv.org/pdf/1705.01693) motivates
causal average-speed and gap management; its 38 s ring tuning is not copied.
[Hart et al.](https://arxiv.org/pdf/2109.14268) motivates short leader-following
mechanism tests. [Jang et al.](https://arxiv.org/pdf/2402.17050) motivates explicit
actuator semantics and aligned outcomes. We initialize around a transparent
reference but do not claim a competent imitation expert before testing it.
No centralized external speed-planner input is imported. These papers do not
establish efficacy in this merge corridor.

## Reward units and ordering

One reward is emitted per global second. Relative to matched native requests,
the physical return is native-minus-policy cost:

`0.5 B/613 + 0.5 E/41204.235963623185 + 0.1 J + 0.1 W + 0.1 L/max(5,L_native) + 2 H`.

B is all-vehicle time with acceleration < -3 m/s2 (vehicle-seconds).
E is integral squared acceleration (summed m2/s3), an effort proxy, not fuel.
J is requested-not-arrived time divided by all requested vehicles (seconds),
including uninserted queues. W is source waiting per requested vehicle (seconds),
deliberately weighted separately in addition to its inclusion in J. L is
leader TTC<3 s exposure (vehicle-seconds); H counts newly recorded collision
and teleport events. Coefficients carry reciprocal units so return is dimensionless.
The B/E scales are inherited exposed development scales, not refitted to outcomes.

Intervention cost at second t is `0.02/T * requested_active_CAVs/live_CAVs`,
with zero fraction when no CAV is present. T is the complete objective horizon;
therefore the **maximum total cost is 0.02**, regardless of CAV count or horizon.
It is negative, not an accumulating reward for being idle. This adds a small
regularizer: it can reject weak benefits below its cost, intentionally, but
does not cap the reward for physical improvement. A 5% reduction in both B/E
at the reference scales earns +0.05 before other changes, exceeding the maximum
cost. Always-native earns zero, and cannot pass the positive damping benefit gate.
The actor loss sums agent score contributions; it does not multiply this
global reward by population. No episode centering or advantage normalization
is applied. Gamma=lambda=1 retains all downstream consequences after CAV exit.

Constraint violations: J or W degradation >2 s/request, added TTC exposure
>max(5 vehicle-s, 5% native), worse completion/backlog or added hard events.
Any violation removes positive aggregate physical return and adds a terminal
penalty of 2 per failed constraint. This prevents a large service gain from
rewarding an unsafe episode. Safety indicators remain empirical constraints,
not a formal safety proof. All scalar components and veto are separately saved.

Ordering tests hold other outcomes fixed: helpful +0.03 > native 0 > unnecessary
-0.02 > harmful -0.07. Additional tests carry late costs through early CAV
actions and verify hybrid conditional likelihoods and masked gradients.

## Separate finite budget and decision

This is not an extension of v1's exhausted 100-development-rollout allowance.
One seed, one design, one budget, no hyperparameter sweep:

- Short SUMO scene: leader–CAV–follower, identical IDM/sensor/actuator, 120 s.
  Four profiles: steady, two braking pulses, catch-up, sharp necessary braking.
  4 native + 1 exact-native identity + 1 switching + 4 reference + 4 unchanged
  analytical comparator + 12 training + 4 deterministic learned = **30 rollouts,
  3,600 global seconds**. Training cycles profiles three times; Adam 1e-4,
  PPO two epochs, time minibatches 32, unchanged full-team signed credit.
  Short B/E scales are frozen from the four native scenes before training;
  service/source are uninformative in the finite no-exit platoon and cannot
  validate corridor service. Keep all trajectory records, including failures.
- Short physical gate: learned steady intervention <=5%, pulse effort reduction
  >=5%, no increased TTC/hard events in any profile, endpoint progress loss no
  more than 30 m for either follower. Passing is mechanism evidence only.
- Up to **8 full 2,400 s corridor diagnostic rollouts (19,200 seconds)**:
  exposed 911001/2000 and 911003/2200, native, unchanged analytical, untrained
  filtered reference, and learned short-check checkpoint. These deliberately
  revisit helpful/harmful v1 cases; they are not prospective validation.
  No full-corridor training is authorized by this check. Report absolute B/E,
  complete service/source/TTC/hard events and paired changes, even if short gate
  fails. Never select a different checkpoint after looking at these outcomes.

Total maximum: **22,800 new global seconds**. Every rollout allocation is logged
before launch; failures consume allocation. Source hashes are frozen before the
first simulation, and checkpoints/results are never overwritten or silently
repeated. A failed mechanics test stops before training; a failed physical gate
forbids an expensive campaign. A learned check that imitates native or merely
raises reward is not evidence the damping controller works.

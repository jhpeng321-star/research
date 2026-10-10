# Recovery correction derived from the frozen v2 evidence

This is continuation of the standalone trained-damping goal. The v2 campaign
remains stopped. Its sources, negative results and checkpoint are preserved.
The work below uses existing evidence and a single separately bounded
**480-second mechanism check**, with **zero training and zero corridor runs**.
It does not authorize an expensive campaign or open acceptance data.

## Structural source of slow recovery

On v2's large-gap branch the target is `EMA(ego speed) + 2 m/s`, converted to
acceleration through a 5 s response time. Let `q=exp(-1/5)`. For sustained
acceleration a, the one-second-updated EMA lags current speed by
`L*a`, where `L=q/(1-q)=4.516655566 s`. Consequently, with a constant learned
correction d, the steady branch satisfies:

`a = (2 - L*a)/5 + d`, hence `a = (2 + 5*d)/(5 + L)`.

The zero-correction limit is **0.210157863 m/s2**; maximum constant correction
+0.3 produces **0.367776261 m/s2**. This is a branch-specific asymptote while
the leader remains detected and projection does not bind, not a global
acceleration ceiling. A transient command or native action can differ.
It explains why the allowed +0.5 m/s2 was not used during sustained recovery.
The untrained reference commands 0.210282/0.210239 at seconds 60/61, closely
matching the equation. The learned corrections there were approximately
-0.0011, not a learned solution to the recovery deficit.

This is our adaptation's defect, not a failed reproduction of published PI
control. [Stern et al., section 3.2](https://arxiv.org/pdf/1705.01693) averages
ego speed, adds gap-dependent catch-up and blends target/leader/prior commanded
speed with gap-dependent weights. Our v2 substituted an additional fixed 5 s
acceleration response. The paper does not validate our substituted dynamics or
its parameters. Its ring-specific constants should not be copied to this merge.

## The information needed is already observable

For a detected leader opening the gap at `w=max(v_leader-v_ego,0)`, even instant
maximum ego acceleration a requires at least `w^2/(2a)` additional gap before
matching a **constant-speed** leader. With remaining sensor range R-g, define
`M = R-g-w^2/(2a)` in metres. A negative M means retention is impossible under
that acceleration bound for that constant-leader scenario. Positive M is only
necessary headroom, not a guarantee; leader acceleration, latency, jerk and
safety limits can make retention worse. Future leader braking can improve it.

The calculation uses only measured bumper gap, ego/leader speed, observed
sensor coverage and the CAV's known active acceleration bound. It uses no native
counterfactual, downstream state or future trajectory. In both saved sharp-brake
traces, M first becomes negative at **second 57**, five decisions before the
policy loses detection and releases at second 62. Absence is treated as unknown,
not a zero gap or a measured zero-speed leader. This establishes observation
availability, not policy efficacy or a calibrated predictor.

Machine-readable evidence is in
`runs/probe/native_selectable_damping_v2_recovery_diagnosis/audit.json`.
The analytical helpers and six tests also check the independent EMA recurrence
and the extremum of constant-leader relative motion. No SUMO was used for that
derivation. These calculations are not a replacement training simulator.

## One fixed recovery prototype, before any more learning

The v2.1 prototype keeps native/release, mode prior, 89-field CAV observation,
correction range, reward, +0.5/-1 m/s2 command limits, jerk projection and
cooldown. Only the active reference changes:

`a_ref = clip[0.8*(v_leader-v) + 0.25*(g-2.5-1.5*v) + 0.05*(EMA(v)-v), -1, +0.5]`.

Relative-speed feedback has units 1/s, gap feedback 1/s2, EMA feedback 1/s,
and reference headway 1.5 s. Measured leader speed and gap continue to drive
recovery as the gap grows; they no longer disappear behind a capped ego-speed
offset. The retained EMA term supplies mild smoothing. The learned correction
and explicit native choice remain part of the eventual trainable design.
The mechanism check itself uses correction zero; it is not a trained controller.

The fixed gains were selected from the ideal continuous linear following law,
not a simulation sweep. With k_g=.25, k_v=.8, k_f=.05, h=1.5, tau=5, the speed
transfer function is

`H(s) = (k_v*s+k_g)*(tau*s+1) / [(s^2+(k_v+k_g*h)*s+k_g)*(tau*s+1)+k_f*tau*s^2]`.

Its denominator is `5s^3+7.125s^2+2.425s+.25`; positive coefficients and the
cubic Routh condition hold. The squared denominator-minus-numerator magnitude
on the imaginary axis is `0.115625*w^2 + 10.515625*w^4 + 25*w^6`, nonnegative.
Thus the **ideal unsaturated continuous model** is stable and does not amplify
leader speed disturbances. This is not a claim of string stability for SUMO,
the hybrid policy, heterogeneous traffic, or the sampled saturated controller.
Those require physical evidence. A sampled linear check is diagnostic only.

## Frozen small check and stop rule

One version, one fixed setting, no optimization: use the existing 120 s steady,
two-pulse, catch-up and sharp-braking profiles with the same SUMO fixture,
sensor, actuator bounds and seeds. Reuse the four hash-verified native results.
New allocation is **four rollouts, 480 global seconds**. A failure consumes
allocation; do not repeat it automatically. No actor/critic update, no corridor
experiment, no acceptance cases and no wall generation are included.

Gate: exact native behavior in steady flow; >=5% pulse effort improvement;
no increase in severe braking, TTC or hard events in any profile; <=30 m
endpoint loss for either follower; no sharp-recovery sensor-loss/native rebound;
sharp-recovery total effort no worse than native. Compare the actual reference
to native and to preserved v2 results. A pass supports only the active mechanism;
learning and corridor service/source protection remain unproved. A failure
stops this prototype without another parameter search or training extension.

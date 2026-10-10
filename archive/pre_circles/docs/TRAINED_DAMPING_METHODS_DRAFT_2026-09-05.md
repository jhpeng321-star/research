# Manuscript methods draft: local learned damping before RSU escalation

**Editorial status, 2026-09-05:** this is the methods draft for the restored
trained-damping project. The implementation and registered experiment design
below are present; controller efficacy, acceptance and escalation results remain
unresolved. No numerical results from the historical wall manuscript are reused.
The training campaign is still running. This document is not a completed paper.

Working paper framing: **Learning Local Traffic Damping with Conditional
Coordination from CAV Reports**. The intended research question is whether a
locally executable learned controller can provide useful standalone damping,
and whether an RSU can identify when coordinated intervention adds value using
reports acquired by CAVs. The second question remains conditional on validating
and freezing the first controller.

## A. Control problem and information boundary

We study longitudinal control of a subset of vehicles in a highway merge
simulation. Each designated CAV executes the same learned feedback policy using
its own observation history. The number and identities of active CAVs change
as vehicles enter and leave. The damping policy does not require an RSU command,
a corridor-wide health score, or a coordinated wall during execution.

The simulation runs at 0.5 s resolution, with one damping decision per second.
An episode lasts 2400 s regardless of when an individual CAV leaves. Each
vehicle's requested departure, route, driver type and CAV designation are fixed
before either member of a paired comparison runs. Consequently, delaying
insertion does not remove vehicles from the evaluated population.

Three information interfaces are separate. The actor receives a local history.
The command adapter additionally uses the CAV's own known IDM parameters. During
training, a critic and outcome logger may access simulator-wide quantities.
Neither the critic nor the native reference trajectory is needed at inference.
This establishes a computational information boundary; deployment under realistic
sensor errors remains an empirical question.

## B. CAV-acquirable observation

For each CAV, concatenate five 17-field frames sampled at one-second intervals,
ordered from time t-4 s through t. The resulting actor input has 85 scalars.
A frame contains:

1. Own speed, measured acceleration, allowed speed, lane index, signed mapped
   distance to the merge, distance-valid indicator, and previous projected
   acceleration correction: seven fields.
2. Front bumper gap, front closing speed, detection indicator and observed
   front coverage: four fields.
3. Rear bumper gap, rear closing speed, detection indicator and observed
   rear coverage: four fields.
4. Frame age and frame-valid indicator: two fields.

Front closing speed is own speed minus front-vehicle speed; rear closing speed
is rear-vehicle speed minus own speed. Speeds and speed differences are divided
by 30 m/s, own acceleration by 4.5 m/s², lane index by two, mapped distance by
1000 m, and gaps/coverage by 100 m. Age is divided by 5 s. The previous projected
correction remains in m/s². These scales are fixed and not fitted to acceptance
data. Missing frames are zero-filled with validity zero. A missing detection
has zero gap and relative-speed fields plus detection zero, so it is not encoded
as a detected vehicle at zero gap. Coverage distinguishes an observed empty
region from an unobserved region.

The present sensor emulator returns only the nearest front and rear vehicle on
the current lane, within 100 m and within the lane's geometric boundaries. It
does not follow neighbors across a lane boundary into the next lane segment.
Although simulator lane inventories implement the emulation, only the selected
neighbor gaps and relative speeds enter the local frame. Other vehicles' IDM
parameters, intended actions and future trajectories do not enter the actor.
Own kinematics are assumed obtainable from onboard state estimation; relative
kinematics from onboard perception; and lane/merge geometry and allowed speed
from vehicle/map information. These are availability assumptions, not a claim
that a camera/radar perception stack has been validated.

Current rollouts use ideal instantaneous measurements with explicit geometric
coverage. The interface can represent missing/stale observations, but the campaign
does not establish robustness to realistic noise, occlusion, delay or map errors.
The hypothesis is that local gap and closing-speed histories contain useful
feedback information. Their sufficiency and the contribution of each feature
have not yet been demonstrated by accepted controller results or ablations.

## C. Bounded learned correction and native handback

The nominal longitudinal model follows the IDM form introduced by
[Treiber, Hennecke and Helbing](https://arxiv.org/abs/cond-mat/0002177).
For own speed v, allowed speed v0, observed front bumper gap s and closing speed
Δv, the implementation computes

\[
s^*=s_0+\max\left(0,v\tau+\frac{v\Delta v}{2\sqrt{ab}}\right),\qquad
\hat a=\operatorname{clip}\left(a\left[1-(v/v_0)^4-(s^*/s)^2\right],-b_{emg},a\right).
\]

Here a, b, τ, s0 and the emergency deceleration limit are the CAV's own known
controller parameters. The interaction term is omitted when no front vehicle
is detected. Numerical handling covers zero allowed speed and vanishing gap.
This is a local nominal approximation; it is not SUMO's exact next native
acceleration, particularly with limited perception and discrete simulation.

The shared actor has two fully connected 64-unit tanh layers and a scalar
Gaussian head. A sampled latent z is transformed to the requested correction

\[
u=-0.1+0.4\tanh z\quad\mathrm{m/s^2},
\]

giving bounds [-0.5,+0.3] m/s². The output-head weights initialize to zero and
the mean bias to atanh(0.25), yielding deterministic correction zero rather than
the asymmetric interval's midpoint. The learned log standard deviation starts
at -2 and is clamped to [-5,0] during use. Training samples the distribution and
accounts for the transform Jacobian in its log likelihood. Evaluation uses the
transformed latent mean, which need not equal the transformed distribution's
expected action.

For a nonzero request, the adapter projects nominal acceleration plus correction
onto the intersection of the residual bounds, commanded acceleration bounds
[-1,+0.5] m/s², a 0.5 m/s³ command jerk bound relative to measured acceleration,
and nonnegative/allowed target-speed bounds. Front and rear observations further
constrain the commanded acceleration using one-step closing-speed conditions:

\[
a_{cmd}\leq\frac{v_f-v+\max(0,s_f-s_0)/T_f}{\Delta t},\qquad
a_{cmd}\geq\frac{v_r-v-s_r/T_r}{\Delta t},\qquad
T_f=3\,\mathrm{s},\quad T_r=4\,\mathrm{s},\quad\Delta t=1\,\mathrm{s}.
\]

Each neighbor constraint applies only when that neighbor is detected. These
heuristic bounds use current observations; they do not prove future collision
avoidance. The adapter returns control to native driving if the current frame
is older than one second, either direction has zero coverage, the feasible
interval is empty, or projection would reverse the correction's sign. A request
or projected correction within 1e-8 m/s² of zero also yields native handback.

An issued target uses `slowDown` over the one-second control interval. The
adapter requires speedMode 31 and releases only its own existing override using
`setSpeed(-1)`. SUMO documents duration-based speed changes, native handback and
the speed-mode checks in its [TraCI command reference](https://eclipse.dev/sumo/docs/TraCI/Change_Vehicle_State.html).
Native simulation constraints may override the target, so commanded bounds must
not be reported as guaranteed realized acceleration or jerk bounds. The logger
separately records requested correction, projection/fallback reasons, command
counts and subsequent average-acceleration tracking error.

Exact zero means native passthrough, whereas a nonzero correction engages the
nominal-model command path. Thus continuity with native behavior as the request
approaches zero is not established. A native comparison alone cannot isolate
the value of learned feedback from the effect of engaging that command path.

## D. Team objective with full-horizon credit

At each one-second interval, aggregate two 0.5 s measurements over all simulated
vehicles. Let Bt be vehicle-seconds with acceleration below -3 m/s², Et the
integral of squared acceleration, St requested-but-not-arrived vehicle-seconds,
and Ht newly reported colliding-vehicle involvements plus starting teleports.
Et is an acceleration-effort proxy with units m²/s³ summed across vehicles; it
is not measured fuel or battery energy. Ht is simulator event instrumentation,
not an estimated probability of collision or a count of unique crash incidents.

The interval cost is

\[
C_t=0.5\frac{B_t}{613}+0.5\frac{E_t}{41204.235963623185}
       +0.1\frac{S_t}{N}+2H_t.
\]

N is the fixed requested population for that case. The braking/effort scales are
frozen from four exposed native development episodes before training; they do
not change with the current policy or acceptance outcomes. Requested departure
times are integrated within each simulation interval. Insertion and arrival
events are assigned to their reported 0.5 s boundary. Uninserted source queues
and unfinished requests remain in the finite-horizon accounting.

Each training case has a cached native trajectory with identical requests and
simulation identity. The reward is `r_t = C_native,t - C_policy,t`. The final
reward receives a further penalty of two when system time increases by more
than two seconds per requested vehicle or arrivals fall by more than one.
This native trajectory is an action-independent training reference, not an
online prediction or actor observation. Scalar reward terms do not replace
the separate physical acceptance constraints.

The critic uses 22 training-only inputs: normalized episode time, due/inserted/
arrived/source-queued request counts, active CAV count, and four statistics in
each of the approach, merge, downstream and ramp regions (vehicle count, speed
mean, speed standard deviation and fraction below 5 m/s). It has two 64-unit
tanh layers and a scalar value head. These regional statistics are privileged
simulation inputs and must not later be reused as purported CAV-only RSU inputs.

Advantages are computed on the complete global timeline with γ=λ=1 and terminal
value zero. Every action at time t receives the corresponding team advantage,
including future costs that occur after its CAV has departed. This retains
downstream credit but does not identify that CAV's individual causal effect.
The repaired trainer uses the signed advantages without per-episode centering.
In the discarded implementation, subtracting their episode mean canceled a
constant terminal penalty from every actor sample. With γ=λ=1, a terminal
penalty of -2 must instead shift every earlier actor advantage by -2 when the
critic predictions are held fixed; regression and saved-data replay verify this.

## E. Policy optimization and registered comparisons

The optimizer uses a clipped surrogate adapted from
[Proximal Policy Optimization](https://arxiv.org/abs/1707.06347).
For a minibatch M of global times, active CAV set A(t), per-CAV likelihood ratio
ρit and shared team advantage At, the implemented actor loss is

\[
L_\pi=-\frac{1}{|M|}\sum_{t\in M}\sum_{i\in A(t)}
\min\{\rho_{it}A_t,\operatorname{clip}(\rho_{it},0.8,1.2)A_t\}.
\]

This sums individually clipped CAV terms and averages over global times; it is
not a single clipped ratio of the joint multi-CAV action. Empty actor times do
not create actor samples. The critic fits full team returns by squared error.
Each complete episode supplies two optimization epochs, with 128 global times
per minibatch, learning rate 1e-4, gradient norm bound 0.5 and approximate-KL
early stop 0.03. The implementation is the explicit PyTorch trainer; the older
RLlib per-vehicle terminal path is not used for this campaign.

Training uses 16 registered seed/loading cases, with 75% covering mainline
loadings 1900–2200 veh/h and 25% covering 2600–2700. Ramp demand is 400 veh/h.
The mainline rate holds through 1200 s, declines in three 100 s steps to
1500 veh/h, and stays there until demand ends at 1800 s. Observation continues
to 2400 s. Requests follow the implemented piecewise Poisson generator. Driver
types are assigned with probabilities 0.1/0.7/0.2, and each request is independently
designated a CAV with probability 0.2; realized penetration varies by seed.
CAVs retain their assigned IDM type, and speed-factor variation within each
type is disabled. This is one fixed demand/geometry family, not a validation
of every IDM traffic regime or realistic driver distribution.

Three optimizer lineages are registered through 100 optimized episodes, with
development comparisons at episodes 20, 62 and 100. Lineage 71 inherits four
valid repaired updates; the other two start from deterministic zero correction.
Thus the lineages do not share identical initialization histories. Report all
nine development checkpoints and the lineage spread rather than only the selected
checkpoint. The eight development cases cover 2000/2200/2600/2700 veh/h with two
seeds per loading. Development selection is constrained by case-level harm and
service limits as specified in the frozen campaign protocol.

Every comparison includes native IDM and a fixed same-interface analytical
controller requesting `clip((v_front-v_CAV)/5 s, -0.5,+0.3)` when its front
observation is current and valid, otherwise native passthrough. This comparator
shares sensing and command projection with the actor and has no gain search.
Its inclusion tests whether learned feedback is competitive with a simple fixed
law. Noninferiority alone would not establish that learning is necessary.

After all development stages, one eligible candidate may be frozen for the
reserved 32-case, three-arm acceptance comparison. The
[prospective protocol](NATIVE_DAMPING_ACCEPTANCE_PROTOCOL_2026-09-05.md) specifies
all-case service/event/TTC limits, lower-load improvement, overload mean limits,
stratified paired uncertainty and analytical noninferiority. Analysis passing
alone cannot accept a checkpoint without the frozen-bundle/trajectory identity
checks. No acceptance outcome is available for this draft.

## F. Boundary of the subsequent RSU contribution

The intended integrated controller first executes accepted damping D*. Only
after D* is frozen can paired damping-versus-wall outcomes define the incremental
benefit of escalation. The RSU's online inputs must be built from timestamped
CAV reports, with sensing coverage and communication assumptions declared.
Simulator-wide health, future outcomes and privileged critic features cannot
be passed through as if they were acquired reports.

Wall feasibility, safe command ownership/handoff, benefit labels and the
threshold-selection/calibration procedure remain to be established after the
damping and treatment gates. There is currently no validated wall threshold,
learned RSU predictor, or collapse-prevention result for this restored pipeline.
The final paper must add those results and integrated untouched evaluation
before claiming the full contribution in the working title.

## Editorial evidence map

Observation/nominal/projection: `env/native_damping.py` and
`env/native_damping_sumo.py`. Actor/inference: `idm/damping_actor.py` and
`policies/sensor_local_damping.py`. Physical accounting and critic interface:
`env/damping_rollout.py`, `idm/damping_credit.py`. Actual training loss and
penalty repair: `idm/damping_ppo.py`. Scenario definition:
`idm/damping_scenario.py`. Campaign and acceptance:
`docs/NATIVE_DAMPING_FULL_CAMPAIGN_2026-09-05.md` and the linked prospective
protocol. These paths identify implementation evidence; they are not literature
citations or independent evidence of efficacy.

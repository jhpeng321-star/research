# Sensor-local learned damping: first implementation and failure reconciliation

Date: 2026-09-05. Status: **controller foundation and bounded learner implemented; the short trained prototype fails deterministic development acceptance.** See `NATIVE_DAMPING_V1_LEARNING_CHECK.md` for the stopped first learner, the verified penalty-credit repair, final comparisons, and exact next work. The main goal remains standalone trained damping, then a frozen damping base, then CAV-reported RSU escalation and the paper.

## What failed previously, and what changes now

### 1. The learner's action often did not reach traffic

The old action-local residual pilot qualified only 6/14 treatments for the requested duration. Even qualified treatments retained only about 19.5% of intended exposure after eligibility, filtering and projection. No helpful qualified effect was observed in that small pilot. See `IDM_PHASE5_DOSE_RESPONSE_V1_DIAGNOSIS.md`.

The successor removes the compulsory selected-absorber/P30 response and fixed 20-second maneuver from its action interface. Each reporting CAV can request a bounded correction every second. The proposed physical interval remains [-0.5,+0.3] m/s². Command bounds are intersected once, rather than stacking the old residual filter and selector on top of the new controller. Conflicting constraints return to native control; safety limits are not loosened to manufacture action authority.

Requested corrections, projected targets, operation/rejection reasons, subsequent observed acceleration and command-tracking error are separate fields. The latter is **not** named a causal executed residual: a same-time nominal model is not a matched native counterfactual.

### 2. The action and reward horizon did not match

The old action was evaluated by its downstream 60-second effect, but a selected CAV's training return could end when it left. The corrected paired binary reward subsequently passed its audit, and a stored-sample option wrapper now exists; neither establishes correctness for an ongoing continuous team controller. See `IDM_DECENTRALIZED_REWARD_CREDIT_DIAGNOSIS.md`, `IDM_BINARY_OPTION_REWARD_RESULTS.md`, and `idm/paired_option_env.py`.

The new credit primitive computes returns over **global simulation time first**, then associates the same time's advantage with active actor samples. Vehicle disappearance never marks a team termination. An unfinished sampling chunk bootstraps its critic value; a true episode objective end does not. Normalization is over active team times, not duplicated CAV rows, and a constant delayed cost is not centered into zero signal. Synthetic tests explicitly assign harm arriving after a CAV exits back to its earlier action.

The reward primitive records equal-weight normalized braking and acceleration-energy penalties, a separate system-time/service penalty (initial coefficient 0.1), and newly occurring collision/teleport penalties. Its denominator is the fixed requested population, not live CAV count. Source waiting must be included in system time. Frozen numeric reference scales must still be built from the declared full-length native development baselines. These unit-tested arithmetic rules are not a completed physical reward-alignment audit or an integrated trainer.

### 3. A fixed-maneuver selection failure was treated as a verdict on damping

The later P30 screens attempted to learn selection/timing of a prescribed maneuver. The reconstructed timing observations failed to recover its measured hindsight headroom. That remains a scoped negative result, not proof that the current continuous feedback policy has no useful observation. See `IDM_TIMING_OBSERVATION_REPAIR_RESULTS.md`.

The successor learns an ongoing response from local history. Do not replay old P30 labels into this new action space or require an old binary AUC threshold to approve this different learning problem. The next efficacy experiment is a capped training campaign once its sampling/credit implementation is verified.

### 4. Global measurements exceeded the deployable observation contract

The new actor consumes only a typed local frame: self speed/acceleration, permitted speed, lane/map position when supplied, previous issued correction, nearest visible same-lane front/rear gaps and relative speeds, sensed coverage, age and missingness. Five one-second frames give 85 scalar inputs. The feature order and configuration have a versioned fingerprint.

The SUMO adapter emulates a kinematic onboard sensor by selecting the nearest object within 100 m of observed lane geometry in each direction. The nearest object occludes objects behind it. Coverage stops at the current lane boundary; it does not pretend to see 100 m through a junction. Physical bumper gaps are calculated from visible geometry, avoiding the minGap-subtracted TraCI neighbor convention. Only the selected neighbor's speed enters the measurement. No HDV private car-following parameters, global health, hidden demand, future route/intention, or global density enters the actor. The CAV's own controller parameters may be used by its nominal controller.

This is an idealized same-lane sensor model, not validated radar/camera perception. Cross-edge perception, measurement noise and field deployment are not established. The observer exposes partial coverage explicitly. Tests perturb occluded/out-of-range traffic without changing actor inputs.

## A necessary correction to the nominal-control proposal

The local IDM equation is a **nominal approximation**. It is not certified as SUMO's exact native acceleration, which also depends on discretization, other constraints and traffic not visible in the local frame.

Therefore:

- An untouched zero-action vehicle receives **no speed, mode, tau or lane command**.
- After this controller has issued a speed ramp, returning to zero releases that owned override once with `setSpeed(-1)`.
- A nonzero request becomes a duration-matched `slowDown` target, with native safety mode 31 required and never disabled.
- Bounds apply to issued commands. SUMO can override them for native safety; observed execution must be logged. A zero request after previous intervention does not undo the traffic history or imply trajectory equality with a world that was never treated.
- Native handback itself is native behavior, not a proven jerk-limited release maneuver. Release transients and follower consequences remain part of the real training/evaluation safety audit. The new rear guard is measured closing-time based; it is not a claimed reconstruction of an HDV's private secure-gap model or a long-horizon safety guarantee.

The asymmetric action head is also handled explicitly: a naive centered tanh map over [-0.5,+0.3] produces -0.1 at raw zero. The new actor shifts its initial latent mean so its deterministic action is zero. Exploration is bounded and its transformed log probability is tested; zero initialization is not a trained policy.

SUMO command semantics were checked against the official [vehicle-state documentation](https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html).

## Implemented components

- `env/native_damping.py`: immutable contract, nominal IDM model, typed local measurements, causal history, simultaneous command projection and native fallback.
- `env/native_damping_sumo.py`: local sensor emulator, exclusive command ownership, no-op/release behavior and measured execution telemetry.
- `idm/damping_actor.py`: small shared local actor, zero-initialized asymmetric bounded head, sampling and matching log probabilities.
- `idm/damping_credit.py`: explicit team cost components, global-time return/GAE calculation, masked actor credit and normalization.
- `policies/sensor_local_damping.py`: executable sensor -> history -> actor -> actuator composition, strict control clock, departure handling, and version-checked inference weights. Saving weights does not mark scientific acceptance.
- `experiments/idm/run_native_damping_mechanics.py`: exposed-seed, prefix-only native/zero/scripted comparison with source/network fingerprints and per-command logs. Refuses reserved seeds, horizons outside 250–600 s, and existing nonempty output directories.

Legacy controllers, the historical wall pipeline, and `train.py` have not been switched to this implementation. In particular, the old per-agent termination/reward path is still inappropriate for the proposed new team-credit objective.

## Completed real-SUMO mechanical evidence

Artifact: `runs/probe/native_damping_mechanics_v1_s4107_m2000/summary.json`.

Setup: exposed development seed 4107, mainline 2000/ramp 400 veh/h, SUMO 1.27.1, 0.5 s integration and 1 s control, 300 s total. CAV designations use a deterministic 20% hash sample, preserving every vehicle's original IDM parameters in all arms. Native, zero and scripted (-0.2 during [100,200)) arms started identically. This fixture uses the existing pulse-demand scenario; it is explicitly not the future full-horizon training/evaluation population.

Results:

- Native and zero trajectories were **exactly identical** over all 300 recorded states; zero made no actuation writes over 2824 CAV observations.
- All three branches had identical preintervention trajectory digests.
- Scripted control issued 1071 speed ramps and changed the traffic trajectory.
- There were 35 releases to native control and no outstanding command owner at the end.
- No colliding-vehicle ticks or starting teleports were recorded in these three short branches.
- 1066 issued ramps had subsequent measured execution; five vehicles left before that observation. About 90.6% of measured ramps tracked their commanded interval acceleration within 0.01 m/s². Median absolute tracking error was approximately zero; the 95th percentile was 0.180 m/s² and maximum 0.863 m/s². The tail confirms that issued commands cannot be equated with guaranteed execution.
- The three branches took approximately 14, 16 and 16 seconds respectively. These short light-loading mechanics timings are not a benchmark for full 2400 s RL episodes or a promised campaign duration.

This closes two engineering questions in this fixture: true-zero identity and nonzero causal authority. It does **not** establish beneficial damping, learned observability, full-profile safety, or prospective performance. No new training campaign, predictor data generation, or reserved-seed evaluation ran.

## Verification and implementation continuation

The fixed-clock collector, explicit request ledger, training-only critic and
PyTorch PPO learner have now been implemented. See
`NATIVE_DAMPING_V1_LEARNING_CHECK.md` for the frozen run budget, reward equation,
collector diagnostic (including negative exploratory safety outcomes), and
training results. The four items below describe the work package that followed
the initial mechanical evidence; they are no longer all unimplemented.

The 38 new focused cases cover physical bounds, future/missing/occluded observations, zero behavior, release ownership, real model-to-command wiring, checkpoint/schema/map behavior, actor gradients, and post-departure credit. The final full repository regression run passed **1068 tests in 126.08 s**. Ruff lint and formatting checks passed on all ten new Python files. These checks establish engineering behavior, not learned traffic efficacy.

The next implementation package is specific:

1. Build the fixed-clock training collector around this controller, including global physical metrics, requested/source-waiting/arrival accounting and the full 2400 s episode contract. Do not inherit the old 1080 s cap or vehicle-lifetime termination semantics.
2. Wire a training-only critic and time-major PPO loss/buffer to these team returns. Record global simulation steps separately from actor sample counts; demonstrate that a departed actor's last action retains downstream credit in the actual buffer, not only in a pure helper.
3. Freeze and derive reward reference scales from native development runs; audit reward/physical outcome directions and service/execution/release costs. Store the scenario, observation-map, normalization and reward identities with resumable training checkpoints.
4. Run the bounded 20k learning check with that integrated path. Do not resume the old activation trainer or expand wall generation. Subsequent development training remains capped as described in `TRAINED_DAMPING_TO_PAPER_ROUTE_2026-09-05.md`.

There is no accepted learned controller yet. The useful progress is an executable, tested foundation that addresses concrete prior failures and produces the telemetry needed to judge the next training run.

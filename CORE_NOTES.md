# C4 control core

Implementation follows the unchanged `src/wms/control/contract.py`. All inference
and reward operations use Torch and support CPU batches, including SUMO's B=1.
No policy training, SUMO execution, or GPU execution is part of this work.

## Public APIs and shapes

- `obs.build_observations(state, report_age_s=None) -> Observation`:
  `own [B,K,13]`, `reports [B,K,16,11]`, boolean `report_mask [B,K,16]`
  and `cav_mask [B,K]`. Scalar state fields follow the contract; `length_m`
  defaults to the project's shared vehicle length. Report age accepts a scalar,
  `[B,K]` source ages, or `[B,K,K]` recipient/source ages.
- `reward.team_reward(state, next_state, accel [B,K], weights: RewardWeights)`:
  exactly `reward`, `cost_ttc`, `delay`, `throughput`, `smooth`, `hard_brake`,
  `fairness_delay`, each `[B]`. `reward.episode_summary(step_metrics)` returns
  detached Python logging scalars, averaging independent batch episodes.
- `policy.CavPolicy(hidden_dim=64, n_heads=4)(obs, group_signal [B,K,8],
  deterministic=False) -> (accel [B,K], log_prob [B,K])`.
- `policy.RsuGroupEncoder(hidden_dim=64, n_heads=4)(obs, positions [B,K])`
  returns zone-conditioned signals `[B,K,8]`.
- `policy.TeamCritic(hidden_dim=64, n_heads=4, n_layers=2)(state)` returns
  privileged team value `[B]`; it consumes all present HDVs and CAVs. Scalar
  kinematics take precedence over packed `veh_kin [B,K,7]`.
- `safety.safe_accel(state, accel [B,K]) -> [B,K]` filters CAV actions,
  preserves HDV proposals, and zeros absent/invalid CAV proposals.
- `bridge.make_cav_controller(policy, group_encoder, safety, record=None)`
  returns an episode-local callback `state -> [B,K,4]` containing next position,
  lane, speed and applied acceleration. `physics_step_index` may be an integer,
  scalar tensor or uniform tensor. Omitted indices use an internal clock.
  `record` can be a list or callable receiving decision dictionaries with
  `obs`, `actions`, `safe_actions`, `group_signal`, `log_probs` and step index.

## Design choices

- Observations preserve contract physical units and feature order. Network
  scaling alone uses the single `obs.FEATURE_NORMALIZATION` dictionary, with
  no clipping. Missing sensed neighbours have `(gap,dv,present)=(100,0,0)`.
  Nearest same-lane neighbours are selected by longitudinal front-bumper
  position, within inclusive `|ds| <= 100 m`; bumper gaps subtract the leader's
  length for front sensing and ego's length for rear sensing, floored at zero.
  Distinct equal-position overlaps are sensed rather than treated as missing.
  Both relative speeds mean neighbour minus ego. Reports sort other present
  CAVs by `|ds| <= 400 m`, with slot-index tie breaking, then pad to 16.
- Actor inputs are only local features, CAV reports and the CAV-only RSU signal.
  The shared actor uses report multi-head attention and a reparameterized Normal
  sample with stable tanh plus affine log-Jacobian correction. No-CAV output
  slots are zero. Empty report/set attention uses an explicitly masked dummy
  key, avoiding all-masked softmax NaNs. Networks have no dropout.
- The RSU uses self-attention and masked mean pooling, conditioned per recipient
  on position and smooth upstream/merge/downstream zone memberships. Actor/RSU
  weights do not depend on slot count; the critic adds full kinematics and CAV
  identity in a two-block privileged set transformer.
  Default parameter counts: actor **31,362**, RSU **43,464**, critic **71,745**,
  each comfortably below the one-million-parameter bound.
- Reward uses the current presence cohort and next speeds, including vehicles
  departing this transition. Throughput is the positive difference of sigmoid
  crossing indicators (1 m smoothing width). Delay is vehicle-seconds lost;
  acceleration penalties are per-step sums. The complete weighted objective
  is divided by `max(present_count,1)`. TTC is a separate constraint: a softplus
  approximation to `relu(ttc_limit-TTC)` summed over closing same-lane pairs,
  without vehicle normalization or folding it into reward.
- Fairness is soft 90% upper-tail CVaR via a softplus Rockafellar--Uryasev
  objective (0.01 s temperature). A detached bisection solves the tail threshold;
  envelope-theorem differentiation supplies first-order gradients. A constant
  entropy-bias subtraction makes equal-delay cohorts return their common delay.
- Safety solves a stop-distance bound for an action held for 1 s followed by
  braking at 3 m/s², allowing the leader to brake immediately at 3 m/s². It
  reserves a 2 m bumper gap and 0.09375 m for worst-case trapezoidal stopping
  overshoot. A conservative log-sum-exp soft-min limits the proposal; the
  actuator floor remains -3 m/s². Nearest-leader selection includes overlaps.
- The bridge decides on even physics ticks and holds the proposed action on
  odd ticks. Safety is re-evaluated each tick. Ramp CAVs merge only in the
  inclusive merge zone with both target-lane bumper gaps >=6 m; simultaneous
  requests are resolved downstream-first, counting accepted merges. Lane safety
  is evaluated after that decision. Motion uses WM-S `direct_command_motion`
  without detaching actions or cached graphs. Its model dependency is loaded
  lazily at bridge construction so standalone sensing/networks/safety stay
  independent of the world-model package.

## Known limitations

- Masks, nearest-neighbour/report membership, ordering and lane changes are
  discrete. Kinematic values, selected features, motion and rewards carry
  gradients within those choices. Actuator saturation can zero gradients.
  Use at least one padded vehicle slot, even for an empty traffic scene.
- Safety cannot guarantee avoidance in a physically infeasible initial state
  requiring acceleration below -3 m/s². It assumes fixed leader identity and
  leader braking no stronger than 3 m/s²; unseen vehicles, future lane changes
  and ramp-end obstacles are not covered by this same-lane stopping bound.
  The 6 m merge gate alone is not a speed-dependent lane-change guarantee.
- Source waiting contributes only when represented by present stationary slots.
  Slots must retain vehicle identity through transitions. Sigmoid throughput is
  a smooth crossing estimate; fairness is per-step tail delay, not lifetime
  delay or subgroup fairness. Episode logging averages per-step fairness rather
  than claiming an episode-level lifetime CVaR. The detached threshold solver
  implements first-order CVaR derivatives, not exact higher-order derivatives.
- Construct a new callback per episode/independent rollout; step zero resets the
  cache. Continuation rollouts must retain their absolute physics-step index.
  Newly occupied slots on hold ticks receive zero until the next decision;
  replacement identities without a presence transition require a fresh callback.
  Batched episodes share the control clock. Decision records retain autograd
  graphs; detach/clear them when used solely for long-running logs.

## CPU verification

Verification uses the supplied `rl_driving` Python environment, `PYTHONPATH=src`,
`TEMP=TMP=E:/agent-work/tmp`, no bytecode/cache writes, and small CPU tensors.
The focused tests exercise feature values/order, report range/order/padding,
actor/critic masks, set equivariance and zones, seeded sampling/log density,
reward formulas/gradients, safety behind stopped/braking leaders, decision
holding/merging, and backward flow through an actual ten-step tiny WM-S rollout
to both actor and RSU parameters.

Full-suite verification must obey the prohibited-path rule even for synthetic
fixtures. A temporary Python audit hook refuses opening any path containing a
prohibited token or a `sealed` component, including in Python subprocesses.
This exposes existing fixture/access conflicts rather than permitting those
reads. Temporary fixture data stays inside this worktree and is cleaned locally.
The guarded full suite completed in 732.88 s: **644 passed, 22 failed, 1 skipped**
(667 collected, including two temporary verification modules scanned by the
existing architecture test). Failures comprised:

- 17 prohibited-path accesses: six mix-data tests, one validation-list fixture,
  nine synthetic pre-push fixtures, and the simulation-layer import scanner
  attempting to read a prohibited planner module. All opens were refused.
- Three existing Windows launch-script tests passing non-string values in the
  subprocess environment (`TypeError: environment can only contain strings`).
- One existing committed Drive registry/generator mismatch.
- One temporary-fixture layout issue: `test_errors_exit_2` discovered this
  worktree's parent Git repository. Verification now sets a local Git ceiling
  for fixture repositories; a targeted recheck is recorded below.

The skipped existing test requires Python 3.11's unavailable `tomllib` in the
provided Python 3.10 environment. C4 itself passed all 25 original focused tests
in that run. Final inspection added two regression tests for equal-position
overlaps and simultaneous merges. Full-suite green status is blocked by the
strict access rule and pre-existing failures outside the authorized edit scope;
neither prohibited accesses nor changes to those files were authorized.

Final checks after the overlap/merge fixes:

- `PYTHONPATH=src python -m pytest -q tests/unit/test_c4_core.py
  -p no:cacheprovider --tb=short`: **27 passed**, 67.23 s.
- Guarded targeted `test_errors_exit_2` plus the existing import-graph tests,
  excluding the prohibited simulation-tree scanner: **163 passed, 1 deselected**,
  25.02 s. The fixture Git-ceiling issue is resolved.
- `python -m ruff check --no-cache` on the five implementation modules and the
  new unit-test file: **all checks passed**.

The complete guarded run invoked pytest with `-q tests/unit -p no:cacheprovider
--tb=line --durations=5`; the temporary runner supplied a worktree-local
`--basetemp` and inherited the audit hook through `PYTHONPATH`. Temporary
verification helper files and generated fixture repositories were removed after
checking. Disabling pytest's cache provider produces one benign `cache_dir`
configuration warning. The full suite was not repeated after the two final C4
regressions because its prohibited accesses and existing failures remain blockers;
the affected C4 and allowed architecture checks were rerun successfully.

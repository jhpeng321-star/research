# Standalone learned damping: frozen full-training campaign

**Scheduling update:** the owner requested parallelism. Three independent
optimizer lineages now run concurrently in isolated directories, with the same
scientific identity and budget. Read
[the active parallel execution record](NATIVE_DAMPING_PARALLEL_EXECUTION_2026-09-05.md)
for process ownership and observation. The serial-coordinator details below
describe the earlier transition and must not be used to restart it.

Status: **episode-20 development evaluation complete and ineligible; training
continues toward episode 62**. Optimizer 71's first assessed checkpoint has
20 optimized episodes (48,000 optimized steps) in
`runs/rllib/native_damping_campaign_v1/`.
Source, partitions, budgets and development selection are implemented and
verified. Damping is not accepted.
The paper goal remains trained standalone damping, frozen D*, then CAV-reported
RSU escalation. No wall or predictor data generation is authorized by this stage.

Checkpoint: `lineage_71/trainer_020.pt`, SHA-256
`9aef1dae1c2fc0b627476d1287693f86cbebb3568fddee6f01098df9d3f48b60`.
Its evaluation identity is recorded in
`development/optimizer_71_episode_020/manifest.json`; the eight-case summary
is complete. The [completed result and raw-record audit](NATIVE_DAMPING_O71_E020_DEVELOPMENT_RESULTS_2026-09-05.md)
record lower-group composite worsening of 4.98%, multiple service/TTC failures,
and exact reproduction of 291,784 deterministic actions. The analytical
comparator also fails the same physical screen. This first evaluation collected
the cached native and analytical references: 24 full rollouts. Later checkpoint
evaluations reuse those references and require eight policy rollouts each.

Training outcomes are mixed. For example, 910001/1900 improved system time by
0.521 s/requested vehicle, whereas the final 910016/2700 training rollout
worsened it by 22.434 s/requested vehicle and also worsened braking/effort.
All 1,471 requests finished in that last rollout, but the service constraint
failed and its extra terminal penalty was applied. The update was finite and
changed the actor. These exploratory trajectories do not constitute a
held-out learning curve or evidence of accepted standalone damping.

For continuation, read `active_stage.json` and verify its PID and command line
against the live process before acting; the file or lock alone is not proof
of liveness. The first tool session 71265 / Python PID 4040 exited successfully
after the episode-20 development evaluation. The coordinator automatically
launched optimizer 71 through episode 62; its trainer PID was 30172 when
verified, with episode 21 / 50,400 optimized steps completed. Use coordinator
session 41027 and its current state for continuation. A poll timeout is not
completion and must not trigger a restart.

The registered continuation coordinator is now active:
`tools/continue_damping_campaign.py`, tool session 41027, PID 32444 at launch.
It adopted the verified live trainer PID 4040 through a Windows process handle;
it waits for successful process exit and completed stage/evaluation records.
It then finishes lineage 71 at episodes 62/100, followed by lineages 72 and 73
at 20/62/100, serially. It adds no stages, budget, tuning or acceptance runs.
Read `continuation/state.json` and verify its coordinator and trainer PIDs
against live processes before taking any continuation action. Do not launch
another stage while this coordinator is live. Observation timeouts preserve the
same process handle; a nonzero exit or incomplete stage stops continuation.
The coordinator has five passing focused orchestration tests. Its source is
separate from the frozen learner; the campaign's registered sources are unchanged.

## Why this follows the short check

The four-update prototype regressed in three of four exposed comparisons.
It is rejected as a finished controller, but four updates do not establish
convergence or observation insufficiency. Its corrected PPO v2 implementation
passed the terminal-penalty regression and real-data diagnosis. This campaign
continues that verified learner without changing its reward weights, features,
action bounds, safety projection, neural architecture or optimizer settings.
The stopped PPO v1 checkpoints are excluded.

Optimizer seed 71 continues the four valid updates in
`runs/rllib/native_damping_v2_credit_repair_o71/trainer_004.pt`. This is a
development optimization state, not an accepted D*. Seeds 72 and 73 start from
zero deterministic correction. The campaign checks the inherited learner's
source hashes, PPO/controller schema, SUMO binary, network and checkpoint hash.

## Seed and scenario contract

The pre-registration audit is
`runs/probe/native_damping_campaign_seed_audit_2026-09-05_v2.json`.
It inventoried 116,484 paths across this project, the older reference project,
and both shared network asset workspaces; scanned 1,701 source/metadata files;
and found no candidate references or static seed-range overlap. It did not open
sealed artifact payloads. Its scope cannot prove absence of undocumented runs.
The first audit recorded inaccessible pytest caches; the successful retry
explicitly excluded those caches and retained the failed record.

- Training: 910001–910016, one frozen loading per seed. Twelve cases cover
  1900/2000/2100/2200 veh/h equally, and four cover 2600/2700 equally. Every
  four-case block contains three lower cases and one overload case.
- Development selection: 911001–911008, two cases at each of
  2000/2200/2600/2700. These are selection data and will not become final tests.
- Damping acceptance: 912001–912032, eight cases per same four loadings,
  reserved only. The current scenario API rejects their generation.

All three partitions are disjoint and excluded from the existing final-test
partition. The existing final and predictor-validation outcome files remain
sealed. Registering a new six-digit block does not rename or rehabilitate any
previously used seeds.

The explicit fixed-request fixture is unchanged: p20 CAV assignment, fixed
individual IDM driver types, mainline plateau through 1200 s, decline toward
1500 veh/h, demand ending at 1800 s, and observation through 2400 s. The
generator's old exposed request hashes are regression-checked. Role validation
now admits only the new registered seed/loading pairs in addition to the old
explicit development API.

## Learner and analytical comparator

The 85-input local-history actor and training-only 22-input critic are unchanged.
PPO uses the actual signed team advantages, with no per-episode mean subtraction.
All command/sensor limits remain in the existing controller contract.

Reward scales stay frozen at B_ref=613 and E_ref=41,204.235963623185 from the
original four native development references. Each newly used training case gets
its own complete cached native counterfactual before the dependent policy run;
those native outcomes do not change the scales or become actor inputs.

The analytical comparator is bounded visible-leader velocity matching:

`delta = clip((v_leader - v_CAV) / 5 s, -0.5, +0.3) m/s²`.

No visible/current leader means native passthrough. It uses the same local
observation, 1 s command cadence, nominal IDM calculation and safety/actuator
path as the learned actor. Its five-second response is specified before new
development outcomes, with no gain search. Relative-speed feedback has precedent
in [velocity-difference car-following models](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.64.017101);
this bounded IDM residual comparator is an engineering adaptation, not a
calibrated reproduction of the full FVD model or a claimed optimum.

Results must show this comparator alongside native IDM. Beating native alone
does not establish the value of learned feedback, because nonzero commands
also engage a nominal controller approximation. Analytical competitiveness and
observation ablations remain scientific questions; no outcome is assumed.

## Fixed training extent and compute accounting

Each lineage finishes at 100 optimized full episodes, or 240,000 optimized
team steps. Checkpoints are evaluated at episodes 20/62/100, corresponding to
48,000/148,800/240,000 optimized steps. These are the complete-episode versions
of the proposed 50k/150k/final checks; no intermediate checkpoint is selected.

- Lineage 71: four inherited valid episodes plus 96 new episodes. Its previous
  charged simulation allocation was 19,200 s, including the stopped attempt;
  another 230,400 s makes its charged total 249,600 s.
- Lineages 72 and 73: 100 new episodes each, or 240,000 s each.
- Total new policy simulation: 710,400 s. Including previous charged work:
  **729,600 s**, below the 750,000 s cap. Unused budget is not permission for
  another hyperparameter or feature campaign.
- New native training references: 16 × 2400 = **38,400 s**, separately accounted
  simulation overhead. They are cached once across all lineages.
- New development references and analytical comparisons: 8+8=16 rollouts.
- Development checkpoint evaluation: 3 lineages × 3 checkpoints × 8 cases =
  72 rollouts. Together with those 16 references/comparisons and the preceding
  eight full development comparisons/references, this uses 96 full rollouts
  within the proposed 100-rollout development ceiling. Charging each earlier
  900-second mechanics/collector package as another whole rollout gives a
  conservative total of 98, still below that ceiling.

Later untouched damping acceptance still requires a frozen candidate and a
separate 96-rollout native/analytical/learned comparison. It is not launched by
the training runner. Its prospective uncertainty and comparator criteria are
specified in `NATIVE_DAMPING_ACCEPTANCE_PROTOCOL_2026-09-05.md`.
Full-run duration is not inferred from short prefixes;
the prior full deterministic episodes took about 138–197 s at lower loadings,
and overload runtime remains to be measured in this campaign.

## Development selection, frozen before outcomes

`idm/damping_selection.py` applies all constraints to all eight cases. For a
case, braking and energy improvements are fractions relative to that case's
native totals; their mean is the composite improvement. These selection
metrics are distinct from the normalized scalar optimization reward.

- No additional reported collision involvements or starting teleports.
- System time and source waiting may each increase by at most 2 s/requested
  vehicle; arrivals may fall by at most one; unfinished requests may rise by
  at most one; final uninserted source backlog may not increase.
- TTC<3 s exposure may increase by at most the greater of 5 vehicle-seconds or
  5% of native exposure. This is a chosen development noninferiority margin
  for a longitudinal proxy, not a validated collision-risk tolerance.
- No individual case may worsen the equal-weight composite by more than 10%.
- Across the four lower-loading cases, mean composite improvement must reach
  5%, with neither mean braking nor mean energy worsening.
- Across overload cases, mean braking and energy may each worsen by at most 2%.

Among eligible checkpoints, rank by lower-loading mean composite improvement,
then lower system time, then earlier optimized step, then optimizer seed.
If none is eligible, select none. A good mean cannot erase a bad constrained
case. All nine checkpoint evaluations must be retained; no early outcome
authorizes changing criteria or opening acceptance data. Fresh acceptance will
also require uncertainty estimates and a frozen bundle, not this eight-case
development screen alone.

## Execution and stopping

Runner: `experiments/idm/train_damping_campaign.py`.
Campaign directory: `runs/rllib/native_damping_campaign_v1/`.
The first command is optimizer 71 through episode 20; later stages use the
same source/scenario identity with `--resume`. Shared reference collection and
all writes are serialized by the campaign lock. Incomplete started rollouts
are never silently repeated. Nonfinite learning, identity mismatch, missing
on-policy likelihoods, broken save/restore or process failure stops the stage.
An ineligible development checkpoint is retained and does not trigger a new
ad-hoc experiment; the prespecified training stages remain the route.

Preflight prepared all 24 training/development scenarios with **zero simulation**.
The complete repository regression suite passed **1,087 tests in 146.01 s**.
The new tests cover role separation, sealed acceptance rejection, accounting,
masked analytical control, harmful-case selection, unchanged old request
populations, and optimizer/random-state restoration.

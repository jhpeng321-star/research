# F4 — observable post-move close-call reward

Scope: MAIN 2026-11-05 15:15, item F4. Implementation is confined to the
cooperation-layer reward and its CPU contracts. The reward implementation and
registration are common to B and R. A bypasses this reward path.

## Fixed registration

`tools/coopmwm/track2_safety_reward.py:REGISTRATION` is the executable registration;
`registration()` returns an independent copy. It is included in P2 reward receipts,
per-CAV reward evidence and the production learning receipt.

- Threshold: **strict TTC < 1.5 seconds**, prescribed by F4 and matching the
  existing replay-bank event counter.
- Weight: **1.0 reward-second per observable pair-event per 1-second control tick**.
  F4 did not specify a numerical weight. This is an implementation registration
  of unit weighting, fixed before tests, with no outcome fitting or tuning.
- Formula: `own_speed_reward + neighbour_weight * mean(neighbour_speed_rewards)
  - 1.0 * observable_post_move_pair_events`.
- An event is a same-lane adjacent follower/leader pair with closing speed greater
  than `1e-6 m/s` and `max(0, leader_front - follower_front - leader_length) /
  closing_speed < 1.5`. This matches the existing closing-TTC convention.
  Persistent close calls count once per control tick, rather than once at onset.

## Move attribution and observation boundary

Only accepted `track2.active_moves` commitments containing the acting CAV are
eligible. Attribution begins with that member's first recorded nonzero library
`context.move_offsets` value. Rejected proposals, `none`, and delayed-onset
zero-offset moves do not open attribution. A zero-offset tail retains exposure
until the commitment ends; expiry is exclusive. History is episode-local.

Measurements must be strictly later than that first offset. Own observations
are at t+1. Authorized V2V observations are at t relative to the t+1 outcome:
a newly actuated move at t cannot be penalized using a remote packet from t.
Current and delayed geometry are never combined into one TTC pair.

The existing reward authorization is reused: current own sensing, explicit
selected publishers, exact t-1 timestamps, range and publisher-count limits,
direct-sensing range and lane coverage. Own sensing wins duplicate IDs, then
nearest publisher/ID. Pairs conflicting with higher-priority rows are excluded.
The domain includes the CAV as follower or leader and neighbour/neighbour pairs,
including sensed HDVs; no automation-type or driver parameters are required.
Each pair counts at most once per CAV outcome. This is local observed evidence,
not a network safety count or complete coverage claim.

"Post-move" is **temporal attribution**, not proof that the move caused an event
relative to a no-move counterfactual. No unobserved counterfactual is fabricated.
Terminal CAV exits retain the existing measured terminal-zero rule; absent
terminal pair coverage is not replaced with global outcomes.

## APIs and receipts

- `track2_reward.neighbourhood_reward(..., first_actuation_s=None)` adds
  `close_call` evidence and `close_call_penalty`. No attribution means zero penalty.
- `training_episodes` and `reward_update` derive attribution from episode samples
  automatically. The production launch already calls this path.
- `close_call` includes fixed registration, first exposure time, pair IDs,
  observation time, bumper gap, closing speed, TTC, count and negative penalty.
- Aggregate receipts include `close_call_events`, `move_exposed_cav_ticks`, and
  `close_call_penalty_sum`. Counts sum per-CAV local exposures; penalties retain
  the existing equal-acting-CAV mean per step. Counts are not deduplicated across
  different CAV reward domains.
- `ArmRuntime._reward_targets` checks the fixed registration, episode-derived
  attribution, observable IDs, timestamps, TTC arithmetic, count, penalty, and
  augmented reward formula. An actuated move cannot use an old envelope missing
  F4 evidence. Non-move historical CPU envelopes remain compatible.
- Existing full-rollout discount, reward unit, return horizon, trigger advantage,
  optimizer settings and base-policy update path are reused.

The fixed F4 registration is additive to the existing neighbour-weight
registration; there is no arm-specific weight or tuning switch. An integrator
should record this module's registration/source commit when identifying the
F4 continuation recipe; the original CLEAN1 recipe did not contain this term.

## CPU verification

Use `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q` with:

- `tools/coopmwm/test_f4_safety_reward_cpu.py`
- `tools/coopmwm/test_track2_backend_cpu.py`
- `tools/coopmwm/test_launch_integration_cpu.py`
- `tools/coopmwm/test_track2_monitor_cpu.py`

These exercise the production reward envelope, trainer validation, delayed
trigger return credit, real layer optimization and frozen base, A bypass, B/R
parity, strict TTC boundary, rear/own/neighbour events, expiry, empty/terminal
observations, packet priority, duplicate handling and delayed V2V timestamps.

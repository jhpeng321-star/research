# F4 — observable post-move close-call reward

Scope: MAIN 2026-11-05 15:15, item F4. Implementation is confined to the
cooperation-layer reward and its CPU contracts. The reward implementation and
registration are common to B and R. A bypasses this reward path. F4 is an
**explicit, default-off addition** to the existing team reward.

## Opt-in and continuation recipe

The common production configuration slot is `f4_close_call`. Absent or `false`
means OFF even when the existing `track2_reward` is enabled. OFF preserves the
launch-parent reward, serialized episodes/envelopes/receipts, accepted historical
move evidence, monitor hashes and learning behavior. It does not add F4 fields or
parse/validate move attribution. Turning off the entire team reward is unnecessary.

`F4_CONTINUATION_RECIPE.json` explicitly enables this addition via `config_patch`:

```python
from tools.coopmwm import track2_safety_reward
cfg['f4_close_call'] = {
    'enabled': True,
    'registration': track2_safety_reward.registration(),
}
```

The checked-in JSON patch binds the full fixed registration, including weight
**1.0**, provenance, threshold and B/R arms. Apply it to the common continuation
configuration before the existing configuration/source bindings are generated.
`launch_production.backend_config` and `monitor_config` validate and carry this
same slot unchanged. Neither arm can override the weight or provenance.
Monitor registration remains immutable within a history: a changed objective
belongs to a separately registered monitor sequence, with the old history retained.

## Fixed registration

`tools/coopmwm/track2_safety_reward.py:REGISTRATION` is the executable registration;
`registration()` returns an independent copy. `settings(cfg)` returns that copy
only for the exact enabled registration, and `None` when absent/false. When ON,
the registration is included in P2 reward receipts, per-CAV reward evidence and
the production learning receipt.

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

- `track2_reward.neighbourhood_reward(..., close_call_enabled=True,
  first_actuation_s=None)` adds `close_call` evidence and `close_call_penalty`.
  No attribution means zero penalty; default calls retain the pre-F4 output.
- `training_episodes` and `reward_update` derive attribution from episode samples
  only when the explicit F4 slot is enabled. The production launch calls this path.
- `close_call` includes fixed registration, first exposure time, pair IDs,
  observation time, bumper gap, closing speed, TTC, count and negative penalty.
- Aggregate receipts include `close_call_events`, `move_exposed_cav_ticks`, and
  `close_call_penalty_sum`. Counts sum per-CAV local exposures; penalties retain
  the existing equal-acting-CAV mean per step. Counts are not deduplicated across
  different CAV reward domains.
- `ArmRuntime._reward_targets` checks the fixed registration, episode-derived
  attribution, observable IDs, timestamps, TTC arithmetic, count, penalty, and
  augmented reward formula when enabled. Enabled envelopes must supply F4 evidence.
  With F4 absent/false, historical envelopes remain compatible for active moves too.
- `track2_monitor` binds the enabled registration into its component contract and
  checks finite nonpositive penalty sums, integer event/exposure counts and
  observation consistency. It validates
  `reward_sum = own_reward_sum + team_addition_sum + close_call_penalty_sum`.
  The raw event count is **not** equated to the equal-CAV-mean penalty aggregate.
  Publication, history reload and status reporting use this same contract.
- Existing full-rollout discount, reward unit, return horizon, trigger advantage,
  optimizer settings and base-policy update path are reused.

The fixed F4 registration is additive to the existing neighbour-weight
registration; there is no arm-specific weight or tuning switch. The explicit
opt-in is bound in `F4_CONTINUATION_RECIPE.json`; the original CLEAN1 recipe did
not contain this term.

## CPU verification

Use `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q` with:

- `tools/coopmwm/test_f4_safety_reward_cpu.py`
- `tools/coopmwm/test_track2_backend_cpu.py`
- `tools/coopmwm/test_launch_integration_cpu.py`
- `tools/coopmwm/test_track2_monitor_cpu.py`
- `tests/unit/test_track2_reward.py`

These exercise the production reward envelope, trainer validation, delayed
trigger return credit, real layer optimization and frozen base, A bypass, B/R
parity, strict TTC boundary, rear/own/neighbour events, expiry, empty/terminal
observations, packet priority, duplicate handling and delayed V2V timestamps.
The fix-round regressions also publish real nonzero B/R envelopes through
`launch_production.update_metrics` and `launch_training.monitor_update`, reload
their history, reject malformed safety aggregates, and compare absent/false
behavior against the actual `8bf7f533` launch-parent implementations. Differential
checks include JSON bytes, active/idle/terminal/empty/historical inputs, monitor
hashes, real CPU optimizer state, policy tensors and learning receipts.

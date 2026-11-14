# V6EXPLORE — entropy temperature + zero-mean acceleration start (2026-10-15)

Both switches default OFF; defaults are bit-identical (no new RNG draws, no extra metrics,
no new checkpoint keys). Branch `c4/v6explore`.

## Evidence (why)

The running v5 policy (`configs/c4_policy_v5.toml`, fixed `entropy_weight` 3e-3 inherited from
`c4_policy_gpuw.toml`) collapsed its exploration over 517 updates:

- per-action entropy (differential entropy of the squashed Gaussian) fell 1.39 -> -0.35 nats;
- the imagined gain vs all-native (D_all) rose -4.4 -> ~-0.2 and plateaued from update ~300;
- the group-cooperation gain C_G fell 0.83 -> ~0 — the policy converged to near-native
  behaviour without exploring.

An older comment in `c4_policy_gpuw.toml` records the same failure at weight 1e-3 ("entropy
kept collapsing"). Also: the tanh decode maps a zero pre-activation to the range MIDPOINT
(-0.9 m/s^2 for the [-3, 1.2] bounds), so the random initial policy mostly brakes (the v5 run
opened with action mean -0.67).

## Changes

1. **Entropy temperature** (`src/wms/control/dreamer.py`): `trainer.entropy_target`
   (default `None` = current fixed weight). When set, the actor's entropy bonus becomes
   `alpha * entropy` with a learned SAC-style dual `log_alpha` (Adam, `trainer.alpha_lr`
   default 3e-4) minimising `-(log_alpha * (entropy_target - entropy))`: alpha rises while the
   mean entropy is below the target, falls above it. `log_alpha` starts at
   `log(entropy_weight)` (0 when the weight is 0), so enabling a target continues from the
   fixed bonus the run would otherwise use. `trainer.lane_entropy_target` does the same for the
   categorical lane head (requires lane rules + lane head; the lane REINFORCE objective uses
   `lane_alpha` in place of `lane_entropy_weight`). `alpha` / `entropy` (and `lane_alpha` /
   `lane_entropy`) are logged every update in `updates.jsonl`; `log_alpha` and its optimiser
   state are checkpointed and restored on `--resume` (a pre-V6EXPLORE checkpoint restarts the
   temperature at `log(weight)` with a warning). The targets also appear in the
   `RUN_SUMMARY.json` config digest.

2. **Zero-mean acceleration start** (`src/wms/control/policy.py`): `policy.init_zero_accel`
   (default `false`). On a fresh actor, `CavPolicy.init_zero_mean_accel()` zeroes the
   acceleration MEAN row of the head and biases it so the decoded mean acceleration is 0 m/s^2
   on any input (bias `= atanh(-midpoint/scale)`); the log-std row and every other weight keep
   their random defaults. This is an initialisation, not a designed controller (owner rule: no
   imitation / hand-designed behaviour) — a neutral zero-mean start is permitted.
   `set_accel_bounds` re-centres the bias while the flag is set, so a configured cap
   (`[policy] accel_max_mps2`) keeps the fresh actor zero-mean; loading a state dict overwrites
   the bias as usual. Plumbed via `[policy] init_zero_accel` in `train_cli.policy_settings`
   (the key is only returned when enabled, so default digests stay unchanged) and reported as
   `policy_init_zero_accel` in the run summary.

## Config

`configs/c4_policy_v6_explore.toml` (extends `c4_policy_v5.toml`): `entropy_target = -1.0`
nats per action dim, `alpha_lr = 3e-4`, `init_zero_accel = true`, output
`runs/active/C4-POLICY-V6-EXPLORE`. Target rationale: -1.0 keeps ~2.4 nats more entropy than
the collapsed -0.35 plateau while still allowing the actor to commit (uniform over [-3, 1.2]
would be log 4.2 ≈ 1.44).

## Tests

`tests/unit/test_c4_v6explore.py`: default-off bit-identity (updates + checkpoints), alpha
starts at the fixed weight and rises/falls toward the target on the toy actor, checkpoint/resume
exactness, lane-alpha direction + logging, `init_zero_accel` decoded mean < 0.05 m/s^2 on random
inputs (and the default random start's braking asymmetry), config resolution (v5 unchanged) and
a CLI smoke digest check. Existing dreamer/policy/srcgrad/policyv5/accelcap suites pass
unchanged.

# T2C fix round — cpuV production wiring and CPU qualification

## Explicit opt-in and default-off compatibility

`runtime_config(..., track2=True)` selects Track 2. Absent/false returns the
committed `c4/track-base` serialized configuration. `run_tracks(..., legacy_run=...)`
directly calls the previous runner when OFF, with no component construction/RNG draws.

Preparation/accounting uses the same boundary: `init_root(root, track2=True)`,
`own_ledger(root, track2=True)`, `charge(..., track2=True)`, and
`preflight_dry(..., track2=True)`. `bind_ledger(root, cfg)` selects from the config.
ON requires the criterion-bound authority, complete config/registration/manifest,
and SHA-bound OWNER_APPROVAL. Absent/false preserves baseline authority bytes,
legacy-root admission, historical superseded-charge CAP subtraction, and approved
legacy preflight receipts. Baseline-comparison regressions cover each boundary.
Omitting the flag cannot approve a Track 2 quote through legacy preflight.

Public `quote()`, `budget()`, `wall_clock()` and CLI preparation also default to
the committed baseline. Select `track2=True` / `--track2` for the new preparation.

## MAIN 02:30 redesign and budget cut

The quote **asserts total INCLUDING smoke <= 5,000,000 sim-s**. Current total and
phase allocation are **4,310,220 sim-s**, leaving 689,780 below the owner ceiling.
Target: **cpuV**, CPU only, one own root/ledger, shared 24+16/40 worker limit.
All allocations remain config/criterion/request/approval-bound:

- Training: 24 updates, **16 full 1,200-second episodes per arm/update**. B/R each
  get eight matched practice references per update (50% practice weighting).
  Training plus references: 1,843,200 sim-s.
- Dense WM teaching and WM counterfactual credit are **OFF**. Their modules are
  neither constructed nor called, and both verification phases have zero budget.
- B/R receive identical real-outcome local team reward through the P2
  `track2_reward` adapter: own realised reward + registered weight times the mean
  realised reward of observable CAV/HDV neighbours. Weight **1.0** is a proposed,
  approval-bound equal-weight registration, with provenance in config; no tuning.
  A's native PPO reward path is unchanged.
- Augmentation: 24 **proposals** per arm; verify at most two per arm with the native
  `sumo_verify.verify` + `verify_perseed.confirm_all8` both-seed admission pipeline.
  Only admitted, cooperative, positive own-seed executions enter imitation or
  practice. Effective samples and AWR weights come from those modified-plan tapes.
  Partial/zero admission retains the original own bank. **143,280 total ceiling**,
  ONCE before smoke/run; the full run reuses the approval-bound admission cache.
- Replicated isolated checks: **B/R, u6/u12/u24, N=5**, 30 cells, 978,300 sim-s.
- u2/u4 probes: single run per A/B/R arm on three registered full-window states,
  conservative ceiling 586,980. Lessons: 21,840; bank gate: 35,820.
- u8 and shared u24 tests: 96 complete episodes per arm each, 691,200 total.
- Smoke: six complete episodes plus two matched practice references = **9,600**.
  Shared augmentation is charged separately to `augment_verify`, never duplicated
  by smoke and full run. Isolated expansions are dry in smoke.

Each sample allowance is a **whole** measured all8 gate (35,820 sim-s). Expanded
specs must fit, preserve the entire registered window, and be reserved in full
before dispatch. Legacy `verify_fraction` metadata is not sampling authority:
`verification_mode=fixed_total_coordinator_only`, `internal_sumo_dispatch=False`.
No component-owned extra physics is allowed. Sampling is deterministic uniform,
the same per-update count for B/R, with no carry/retry allocation.

The newly SHA-bound isolated criterion compares B's gain against common frozen
u0 (its group-minus-best-solo gain is algebraically zero), with bootstrap CIs and
B's branch safety totals versus equally repeated matched reference totals.
B/R paired CIs are reported. **This is not the learned-A comparison**: learned
A/B/R are compared by the shared T1D final harness. No extra learned-A isolated
replicates are simulated under the budget-cut schedule.

```powershell
python -m tools.coopmwm.rerun_cpus quote --track2
python -m tools.coopmwm.rerun_cpus preflight --track2 --prep PREP_DIR --stage STAGE_DIR
```

## Production backend

`rerun_tracks_backend.backend_factory` is the default backend for `run` and
`smoke` when `--adapter` is absent. It reuses the reviewed coop3 paths:

- `v2v_runner.prepare_intent`, `v2v_live.hooks`, native checkpoint migration and
  intent policy; `accel_runner.wave_jobs`, `accel_plant.episode`, and native
  own-action PPO for A. B/R bases stay frozen and never enter base PPO.
- Real per_seed_v1/cooperative-only/effective bank readers and intent refresh;
  source plan/seed/AWR weight/limits are retained, not replaced by label counts.
- `practice_starts.wave_call` for own-bank snapshot/CRN restart and prefix replay;
  the same snapshot/seed supplies each common-u0 reference. Repeat caps apply to
  the underlying practice example, not individual imitation seconds.
- `sumo_verify.capture`, `build_branches`, `LocalBranchRunner`, and full-prefix
  checks; reviewed `MixedPolicy` CRN/member selection for isolated branches.
  Layer/trigger weights are included in complete diagnostic checkpoints.
- Native all8 move plans carry exact per-second **relative offsets** on each
  branch's dynamic base action, rather than a static absolute approximation.
- `v2v_launch3.before_launch` immediately before each actual process submission:
  immutable source/runtime/gate, HOLD, resources, and own-ledger budget checks.
  The Track 2 launch binding has `track2=true` and includes the new executable
  sources. OFF retains the old launch contract and superseded-charge accounting.

One coordinator owns all commands, reservations, workers, updates and receipts.
The backend's entire synchronous process lifetime stays inside its permit.
`T2C_RUNTIME.claim` prevents duplicate coordinators and is never auto-adopted.
Scheduled diagnostics load their immutable update checkpoint while later training
can proceed. Subprocesses are CPU-only and execute the original 1,200-second
horizon: 300-second native warmup plus 900 control ticks. Synthetic plants use
1,200 ticks; they never write real charges or claim SUMO learning.

Extra component observations project to **each acting CAV's** sensing range and
causal t-1 partner reports, with per-actor active move/onset/RNG state. Row order
does not select the actor. Isolated all/solo branches evaluate exactly the selected
members, including disconnected groups. Overlap is resolved once per actuator:
the target CAV's own context has priority, then stable actor ID. Only whitelisted
observable fields are forwarded. Private
execution snapshots go to exact replay, never the WM/trigger input. The native
request/safety/CommandOwnership chain remains the sole actuator writer, including
the existing speedMode 27 and 1.2 acceleration cap.

### Approval-bound sibling adapters

The approved config's `production` object supplies:

- `component_factories`: the six enabled names (`merge_features`, `trigger_head`,
  `coop_layer`, `track2_reward`, `plan_augment`, `practice_weighting`) ->
  `tools.coopmwm.<adapter>:<callable>`;
- `move_to_plan`: sibling Move -> native plan/duration/window adapter;
- `release`: CPU/non-stub/qualified/single-writer/frozen-base/default-off proofs;
- `launch_binding_path`, `launch_binding_sha256`, `control_root`, `mover_root`.

Factories receive `base=` plus `cfg=, arm=, moves=, components=` in dependency
order. Explicit entry points avoid guessing sibling constructor symbols. They
implement these consumed component APIs using their reviewed modules:

- `merge_features.encode(obs)` -> values and `missing_share`;
- `trigger_head.choose(obs, features, proposals, rng)` -> a member of the slate;
- `coop_layer.residual(obs, features)`; `.update(rows=, bank_batch=, bank_options=,
  realised_rewards=, group_advantages=None, dense_targets=, dense_mask=, trigger=, kl_cap=)`;
- P2 `track2_reward.realised_rewards(before_local_obs, after_local_obs, elapsed_s=)`
  -> finite per-vehicle realised rewards for exactly the earlier visible IDs.
  This adapter must implement the registered native outcome reward. T2C projects
  local observations before this call and independently applies the registered
  neighbourhood formula; global scalar rewards and remote IDs are refused.
  `accel_plant` calls the optional hook only after a successful real SUMO tick;
  per-actor realised rewards accumulate until the next decision and enter only
  that actor's layer-learning rows. No latent or all-road input reaches P2.
- `plan_augment.augment(bank, count=, arm=)`;
- `practice_weighting.select(bank, augmented, count=, repeat_cap=)`.

Layer and trigger expose `parameters()`/state dictionaries; the layer optimizer
owns exactly their trainable parameters, disjoint from the frozen base. Bank
learning is a separate full pass with per_seed_v1 weights, kl_targeted/.045 and
reviewed effective/Tobit `bank_imitation.pair_loss`. Each effective source sample
retains `native_sample` with its original V2V/intent/time links for the adapter.
The returned `bank_receipt` binds `batch_sha256, move_rows, effective_rows,
effective_loss, options`; incomplete learning consumption is refused.

The old dense-label filter remains covered as dormant code. The current launch
does not construct teachers/calibrators/WM-credit adapters or supply their targets
or advantages to learning. Dense and WM-credit monitor fields are null/zero.

Production configure resolves every factory export and the T1D boundary before
dispatch. Missing siblings refuse rather than selecting a test backend. The
sibling implementations are not on this worktree: CPU tests use test-only
adapters, as permitted by the shared spec. Deployment needs those merged exports,
its existing Linux qualification, staged artifacts and bound owner approval.
No live SUMO/GPU/SSH qualification occurred in this fix round.

### Changing membership during a move

Before executing a retained move, each actor checks that all its members remain
locally observable and commandable. Loss of any member cancels that entire
commitment until the next registered decision tick; visibility return alone does
not restart it or draw a new trigger decision. The native base and valid residual
continue. The actor owns its own zero-correction fallback even when another
overlapping context requests it. Actor exit discards its context; reappearance
initialises a new context, as for any newly entering CAV. Fresh invalid proposals
and nonobservable/uncommandable residual outputs still refuse.

Training keeps the original sampled move/onset and reward row, with additive
`move_cancelled_at_s` and `move_unavailable_members` fields when it is cancelled.
No cancellation fabricates a new decision/label. The same lifecycle and ownership
apply to isolated all/solo and shared-final hooks; evaluation produces no learning
rows. CPU turnover evidence and exact test commands: `T2C_FIX_MEMBERSHIP_R1.md`.

## Shared T1D final evaluation and state registration

The final path imports `tools.coopmwm.planner_eval` and requires
`prepare_track2(registration=, checkpoints=, contract=)` and
`analyse_track2(registration=, checkpoints=, records=, contract=)`.
This explicit adapter maps to T1D's native paired-world analysis, not independent
per-arm tests. Final registration lists 96 explicit TRAIN worlds. Each branch
echoes arm/world/full-window/checkpoint/file-hash identities. B/R checkpoints must
contain base, layer, trigger and separate optimizer; A contains base/optimizer.
Results require 10,000-draw paired bootstrap CIs (95%, seed 20261104) for B-A/R-A/B-R
and all collisions/teleports/hard-brakes/TTC counters. Receipt:
`T2C_FINAL_PAIRED_EVAL.json`. Isolated results remain separately recorded.

Each isolated registration replicate additionally supplies a `states` list:
`snapshot_path, snapshot_file_sha256, receipt_path, receipt_file_sha256`. The real
backend reads exactly these TRAIN states, checks their bytes and the registered
state/window-set digests, and preserves their complete replay windows. No seed
list is inferred or read from a protected partition.

## Launch, smoke and one-command monitoring

```powershell
python -m tools.coopmwm.rerun_cpus smoke --config T2C_CONFIG.json --registration ISOLATED_REGISTRATION.json --prep PREP_DIR --stage STAGE_DIR
python -m tools.coopmwm.rerun_cpus run --config T2C_CONFIG.json --registration ISOLATED_REGISTRATION.json --prep PREP_DIR --stage STAGE_DIR
python -m tools.coopmwm.rerun_cpus monitor --root RUN_ROOT
python -m tools.coopmwm.rerun_cpus monitor --root RUN_ROOT/smoke
```

Both commands use the **same config, approved run root, preparation, stage and
OWNER_APPROVAL**. Lifecycle and reward registrations are part of the bound request.
Smoke writes to `RUN_ROOT/smoke/` (monitors/checkpoints/components), uses one-use
`T2C_SMOKE.claim` and `t2c_smoke_...` episode/reference IDs. The full run retains
`T2C_RUNTIME.claim`, root-level output names and `t2c_...` IDs. Both charge the
same root ledger. Common `T2C_AUGMENT_<arm>.claim/.json` caches are sealed/config-bound;
their native all8 jobs run only once, with original full windows and phase cap.
Full training starts with fresh common-u0 models, never smoke-trained weights.
If a smoke claim exists, full admission requires its completed, sealed matching
`smoke/T2C_SMOKE.json`. Duplicate/partial/interrupted smoke or run claims are never
deleted/adopted/replayed. A shared one-use start claim prevents first-start races.

Smoke runs one update/two episodes per arm with all six enabled B/R components and
coop_reward OFF; requires nonzero triggers, realised reward rows, verified augmentation,
weighted starts, available merge features, monitor receipts and 30-cell dry
isolated expansion. Test-only `--adapter tools.coopmwm.test_t2c_runtime_cpu:cpu_backend`
reports `CPU_PLANT_SMOKE_PASS`, `new_sim_s=0` and virtual work separately.

Every arm/update writes a self-hashed contiguous `learning_monitor/<arm>/uNNN.json`:
per-family trigger rates, realised local team reward and its row count, bank CE,
layer KL, merge missingness, native reward and safety counters. Component receipts
include each acting CAV's own/neighbour reward, IDs, weight and provenance.
A has null inapplicable metrics. DIAGNOSE: saturated triggers, non-rising realised
team reward over three updates, or near-zero KL. Learning diagnostics never pause;
only explicit safety stops pause. Real learning efficacy requires inspecting
these receipts and probes in the later approved run.

## Staging closure

The manifest uses real readers and discovers linked bank targets under the
staging home. Both lexical aliases and resolved bytes are guarded and inventoried.
It now invokes **`_decision_snapshot` / `_ls_snapshot`** for each imported LS
decision. `snapshot_dependencies` closes pending envelopes, gzipped tensor
snapshots, and the envelope's cross-root RESULT marker and compressed payload.
Deployment-absolute envelope paths are mapped into the local mirror. Missing
dependencies fail manifest construction; deleting a declared input fails staged
validation even after resealing. Generation tapes follow the real
`label_attempts/<generation>/branches/<verification>/<branch>/RESULT.json(.gz)`
layout. Approval binds the complete manifest document and file hashes, config,
registration, criterion, quote, budget request and verdict. A sealed inventory
alone is never treated as proof of transitive completeness.

# T2C clean rerun — CPU integration and qualification

The experiment is selected with `runtime_config(..., track2=True)`. Absent/false opt-in
returns the committed `c4/track-base` configuration. `run_tracks(..., legacy_run=...)`
directly calls that previous runner when OFF, without constructing Track 2 components
or consuming RNG. The parity tests compare serialized configuration against that actual
commit and compare action, reward, updated model state and RNG under matched seeds.

## Current registration

A runs plain MAPPO PPO from the common u0. B/R share the frozen common base, separate
layer/trigger optimizer, KL ceiling .045, observable merge features, group-credit layer
advantages, own cooperative-only per-seed banks and effective bank imitation. B uses WM
DAgger labels; R uses random labels over the same proposed move distribution. Both use
the same augmentation procedure and 50% practice-weighted starts, repeat cap 2, from
their own banks/augmented examples. The existing unweighted fallback is still .25;
the selected weighting consumer supersedes it in this experiment. `coop_reward` is OFF.

Proposed registrations (all approval-bound): 60 s decisions, 8 proposals including
none, .1 verification fraction for teaching and credit, 24 augmented examples per
arm/update with .1 verification fraction rounded UP separately per arm/update.
Verification samples are deterministic uniform samples of the update's rows. A
verification allowance is one **whole** measured all8 gate (35,820 sim-s); expanded
specs must fit and retain their original windows. These are conservative ceilings,
not newly measured component timings. Quote includes all matched practice references,
u2/u4 three-state probes, 96 replicated isolated cells, lessons, u8/u24 fixed tests and
the separately bounded smoke. Run `python -m tools.coopmwm.rerun_cpus quote` for totals.
The large verification allocation is deliberate: dense verification at every registered
decision is expensive; no owner approval or reduced-cost measurement is implied.

## Executable boundary

`rerun_tracks_runtime.Runtime` owns the complete update loop. All arms and diagnostics
use one coordinator and one aggregate limiter: 24 episode permits, 16 diagnostic permits,
40 total. Episodes execute concurrently; frozen-checkpoint diagnostics can overlap the
next update's episodes. Every branch wave is reserved in full before dispatch. One
immutable `T2C_RUNTIME.claim` prevents a second coordinator/duplicate launch in the same
root. An interrupted claim requires investigation; it is never automatically adopted.

Backend **factories** are explicit integration adapters, not guessed sibling exports.
No production substitute is supplied for modules still on sibling branches. The CPU
tests supply numerical substitutes using real tensors, autograd and SGD, and deterministic
plants. A live backend must pass CPU-only/non-stub qualification, report host `cpuS`, and
pass real preparation preflight bound to exactly its config/registration. This CPU fix
round does not constitute a live Linux/SUMO qualification.

Backend protocol consumed by the coordinator:

- `cpu_plant`: explicit bool. Synthetic work writes only `SYNTHETIC_CHARGES.jsonl`; real
  work uses this root's sole `CHARGES.jsonl`. `load_base(path, arm=, cfg=)` loads a CPU
  module; all three loaded u0 states must match before A learns. B/R are frozen.
- `adapters(base)` provides the seven factories accepted by `track2_components`.
  They receive `cfg=, arm=, moves=, components=` in dependency order. Required instances:
  `merge_features.encode(obs)`; `trigger_head.choose(obs, features, proposals, rng)`;
  `coop_layer.residual(obs, features)` and `.update(rows=, bank_targets=,
   group_advantages=, dense_targets=, trigger=, kl_cap=)`;
  `group_credit.estimate(obs, move, base)`;
  `dense_teacher.label(obs, proposals, kind=, rng=)`;
  `plan_augment.augment(bank, count=, arm=)`;
  `practice_weighting.select(bank, augmented, count=, repeat_cap=)`.
  Layer and trigger expose `parameters()` and the layer exposes `optimizer` owning
  exactly their trainable parameters, disjoint from the frozen base. Layer update
   returns finite `dense_ce`, `dense_accuracy`, `bank_move_ce`, `layer_kl`.
   The update additionally consumes `dense_mask`, aligned with all on-policy rows.
   False entries contribute no dense CE but retain policy/group-credit supervision.
   When none are eligible, dense CE/accuracy are null and learning continues.
   `label` returns `{target, calibrated_probability, provenance, imitation_eligible}`.
   Provenance is `{kind, receipt_sha256}`: `planner_calibration` (T1C calibrated
   P(this target beats base)), `sumo_verified` (realised verification receipt), or
   `no_target` (null target). Runtime independently enforces P >= **0.8** for
   unverified B **and** R labels; verified examples bypass this filter. Eligibility
   claims inconsistent with this rule are refused. The registration/criterion and
   approval payload bind the threshold and symmetric filter.
- `new_episode(arm=, update=, index=, start=, horizon_s=, options=)` returns a plant with
  `observe()`, `full_window()`, `step(commands, speed_mode=27)`, `finish()`.
  Observations have exactly `time_s, own_sensing, v2v_reports, cav_ids, bounds`:
  own sensing and partner reports only. Current/future reports are excluded. Each move
  belongs to the acting observable CAV group; no global vehicle/truth channel exists.
  Plant adapters must expose the registered decision group per episode and must not
  perform additional unquoted teaching/verification decisions internally.
  Runtime is the sole command writer: base + move offsets + residual, existing
  per-member feasibility bounds and 1.2 acceleration cap. No HDV commands.
  All 1,200 per-second ticks execute. Finish returns reward, training batch, all safety
  counters and an explicit safety-stop decision.
- `base_actions(base, obs)`, `ppo_update(base, batches)`, `save_checkpoint(path, base,
  components)` bind the existing low-level learner and checkpoint persistence. The
  checkpoint adapter must preserve layer, trigger and optimizer state for resumption.
  B/R never enter the base PPO update; their advantages go only to the layer update.
- `bank_examples(arm, update, cfg=)` returns own-arm verified examples with unique `id`,
  `arm`, `plan`, full `window` and restart/seed identity. Augmentation must use the
  registered moves/variety ranges. Both arms follow the same procedure. Practice
  selection is checked for source membership, count, arm and repeat cap.
- `expand_verification(kind, payload, cfg=)` returns `(spec, snapshot)` pairs expanded
  through existing all8/isolated machinery. The coordinator checks full-window identity,
  budgets and dispatches via `run_branch(spec, snapshot)` under a permit. That call must
  keep its entire subprocess lifetime inside the call and may not dispatch elsewhere.
  `grade_verification(kind, payload, results)` returns a graded receipt (including
   `advantage` for credit). `practice_reference(start, common_weights=)` returns the
   full matched same-snapshot/same-seed all-base reference pair.
   Teacher verification must return a realised boolean `label_correct`, evaluated
   against the candidate target, including filtered proposals. Runtime preserves
   verification ID and source row ID in `components/uNNN_<arm>_dense_labels.json`.
- `resolve_states(job)` returns registered `(snapshot, receipt)` pairs. Use
  `diagnostic_identity(states)` when registering hashes. Explicit TRAIN seeds, snapshots,
  member groups and full-window digests are checked before expansion.
  `diagnostic_weights(base, components)` exports a checkpoint consumable by the qualified
  branch controller (including the layer and trigger). Runtime reuses existing
  `policy_synergy.specs_for` for all-group/each-solo branches, replacing its legacy
  background-new check with an all-common-u0 reference. Background and reference remain
  common base for every arm. Never route a layer checkpoint into an unqualified legacy
  controller. Existing `policy_synergy.metrics` performs the decomposition; the T2C
  paired evaluator consumes the complete 96-cell grid. Probes use the first three
  registered states, full windows, at u2/u4.
- `expand_scheduled(kind, arm=, update=, cfg=)` and `record_scheduled(...)` integrate the
  existing lesson and fixed-test paths. Empty/ungraded work is refused, fixed tests need
  96 whole episodes per arm. All branches use the same pool and own ledger.

## Smoke and monitoring commands

With a prepared new root containing all three approved u0 checkpoints and the registered
config/explicit TRAIN-world registration:

```powershell
python -m tools.coopmwm.rerun_cpus smoke --config T2C_CONFIG.json --registration ISOLATED_REGISTRATION.json --adapter YOUR_QUALIFIED_MODULE:backend_factory
python -m tools.coopmwm.rerun_cpus monitor --root PATH_TO_RUN_ROOT
```

The smoke runs one update and two full episodes per arm, with all seven required
components enabled for B/R and the reward shaper OFF. It requires nonzero move triggers,
credit rows, dense labels, augmented examples, weighted starts, available merge features,
monitor receipts and dry expansion of all 96 isolated cells. Its proposed cap is
`SMOKE.total_sim_s` (training + matched references + 18 whole verification allowances).
Synthetic CPU smoke explicitly reports `CPU_PLANT_SMOKE_PASS`, `new_sim_s=0`; virtual
work is not a SUMO-learning claim. The CLI regression executes this exact command using
`tools.coopmwm.test_t2c_runtime_cpu:cpu_backend` on a fresh synthetic fixture.

Every arm/update writes `learning_monitor/<arm>/uNNN.json`, plus an update receipt and
checkpoint. Required metrics: rates for every family, WM credit and checked subset,
dense-label CE/accuracy, bank-move CE, layer KL, missing-feature share, reward and
collisions/teleports/hard-brakes/TTC events. A reports explicit nulls for inapplicable
layer metrics, with zero component counts. History is contiguous and self-hashed.
Dense reporting separates imitation accuracy from realised `dense_checked_accuracy`;
`dense_filtered_share`, `dense_eligible_labels` and `dense_checked_labels` expose the
denominators. No eligible dense targets adds a DIAGNOSE, never a learning pause.

Pre-registered DIAGNOSE rules: total non-none trigger rate 0 or 1, credit not rising
over three updates, dense accuracy flat within 1e-6 over three updates, layer KL <=1e-8.
These never pause training. Only the integration's explicit safety-stop decision pauses.
The CPU plants prove parameter updates, receipts and orchestration; learning efficacy
must be inspected in real per-update receipts and the registered probes/tests.

## Approval and staging

`budget_request_document(cfg, registration, manifest)` requires a complete manifest and
checks common-checkpoint SHA. OWNER_APPROVAL binds budget-request, quote, verdict,
criterion, config, registration **and manifest file** hashes. The request additionally
binds the manifest's document hash. Empty/incomplete inventories, resealed substitutions,
unsafe destinations and common-u0 mismatches are refused. Staging uses real bank readers,
discovers linked targets under the staging home, hashes compressed tapes/receipts/snapshots,
and materializes lexical aliases plus the original target destinations. No historical
bank directory name is hard-coded. Both lexical paths and resolved targets are guarded
before reads.

# DRAFT brief: Part B action and wall inputs (code only, zero SUMO)

**Status:** DRAFT, 2026-09-30, world model session (docs only). Not dispatched.

**Dispatch conditions:**

- **Earliest dispatch:** after v0.3.2 (5a0123ac) is reviewed and merged into
  zcode/wd01. It edits the same files (partB_v0data/featurize_v0.py,
  deployable_features.py, partB_v0/data.py), so the file ownership must be
  disjoint first.
- **Executor:** GLM Flash, in a fresh worktree from zcode/wd01 after the
  v0.3.2 merge.

## Why

Part B V^π(s, τ) must be conditioned on what the CAVs are doing
(DESIGN_PARTB_V2.md §1.1, §3). Today that conditioning is missing:

- the V0 featurizer stamps every CAV token with action = "native" and
  phase = none;
- wall_tokens are all-zero (featurize_v0.py lines ~104–138);
- the skeleton wall token (part4/partB/featurize.py) packs a crude
  "member-s summary" into the remaining-time slot.

Three facts make this wrong for the current data:

- In chunks 1–3 and fix27, the base runs are damping-fleet, and the
  `__native`/`__pi` arms are per-CAV switching. So the true per-CAV actions
  vary and are known.
- The chunk-4 `__native_base` runs are fleet-native.
- The wall pilot (campaign_walls_v1) will produce wall episodes.

## Scope

1. **Per-CAV action at time t** (CAV token action one-hot
   {native, damping, pi, wall}):
   - The action comes ONLY from `switch_schedule.json` and
     `suffix_commands.jsonl` (the `owner` per command), never from
     `dispatcher_rows.jsonl`. dispatcher_rows can re-stamp old rows, a known
     slice bug.
   - Pre-t0 base segments use the base run's own policy (damping fleet;
     verify from the base's commands where they are logged).
   - A CAV that has no command at t, and no schedule entry, is marked
     `unknown`. A mask channel carries that; it is never silently set to
     "native".
   - Test: the reconstructed actions for w8481m2700p02t480__native match the
     schedule/commands (damping 556 / native 487 owners), and no code path
     opens dispatcher_rows.jsonl.
2. **Wall tokens** from the wall pilot's per-branch `wall_events.jsonl` and
   `branch_record.json` (schema in
   campaign_walls_v1/wall_driver.py, header lines 1–30):
   - Fields per wall: wall_id; phase ∈ {requested, assembling, holding,
     released, aborted}; wall speed (the planner-chosen commanded speed);
     time in phase; n_members; the members' mean s_down and lane span; the
     planned hold remaining.
   - The planned hold remaining is **planner-owned state**. It is allowed as
     input because the planner knows its own decision, and it is flagged as
     such.
   - CAV tokens of members get member = 1 and the wall's phase, speed and
     time in phase. Non-members keep their own action.
   - Outcome fields (assembled_released, aborted_*) and anything after t are
     NEVER features. A test enforces this: shuffle the events after t, and
     the features at t must be unchanged.
   - Replace the skeleton's crude member-s hack with explicit fields. Bump
     WALL_RAW/CAV_RAW in one place, and give model.py a config-driven width.
     Old checkpoints fail loudly on a width mismatch.
3. **Deployability:**
   - Action and wall state are planner/RSU-known (the RSU issues them). They
     are allowed in deployable_strict.
   - Member positions come from CAV self-reports.
   - No HDV-derived wall features.
   - The strict-variant test from v0.3.2 is extended to the new fields.
4. **Fixtures:** there is no wall pilot data yet (the pilot awaits owner
   authorization). Tests use synthetic `wall_events.jsonl` built from the
   campaign_walls_v1 smoke/driver test fixtures. They cover:
   - assemble → hold → release;
   - abort in assembly;
   - not_feasible.
5. **No rebuild and no training in this task.** Only the builder flag
   `--with-actions` is added. The v0.4 dataset rebuild, with actions and on
   chunk-4 data, is a separate registered task.

## Constraints (copy into the dispatch)

- Zero SUMO.
- Never kill processes by image name or pattern; only stop PIDs the task
  spawned.
- Never use git reset, git clean, checkout -- . or stash drop.
- No HOLDOUT or CAL file opened, even for format inspection.
- No D: paths (WD_FORBID_D=1).
- No shared-record edits.
- maxSubagents 0.
- Paste real stdout.

## Acceptance

- Tests: action reconstruction from schedule/commands; no dispatcher_rows
  reads; unknown-action mask; wall token phases on the 3 synthetic
  lifecycles; no future or outcome leakage (the shuffle test); strict-variant
  field audit; model width config.
- partB_v0 and partB_v0data suites green, plus output/wd01/tests.
- CHANGES table (item → file/function → test).

## Open (for main / owner)

- Should chunk-4 `__native_base`, plus the future wall-pilot branches, be the
  v0.4 training source? That makes V's native vs damping vs wall contrasts
  identifiable, which the switching-mix data cannot do.
- This needs a data registration once chunk 4 completes and the wall pilot
  is authorized.

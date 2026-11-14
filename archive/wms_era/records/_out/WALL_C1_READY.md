# WALL-C1 READY — engineering verified; real-data preparation partial

Stable record: **WALL-C1-PREP-20261009**, branch `wms/wallc1`, base `963e118`.
CPU-only, no training, no SUMO. Work resumed from the cancelled uncommitted LPWO
builder, preserving its artifacts. All source operations in this execution are
workspace-local. Executor: OpenCode GPT-6.1-Sol, with two bounded engineering
subagents; no GLM executor identity is claimed.

## Split counts and artifact status

The existing local LPO receipt reports **1307 training / 189 validation** windows;
training policy counts in the interrupted LPWO receipt are damping **358**, native
**254**, trajectory **560**, wall **135**. Their sum is **1307**.

The interrupted `_out/lpwo/BUILD_RECEIPT.json` reports **1172 train / 169 val**,
**155 removed** (135 train, 20 val), **1719 excluded run keys** (1564 inherited,
155 added). Removed training labels: **114 engaged**, **13 unassembled owned**,
**8 refused**; validation: **16 engaged**, **3 unassembled owned**, **1 refused**.
These are preserved receipt values, not independently verified run outcomes or
accepted corrected split counts. The receipt accepts **273** pack_v3 windows by
wall-events absence and **907** auxiliary windows by construction. This fails
the current requirement to verify schedules before retaining a run.

[`lpwo/SUPERSESSION.md`](lpwo/SUPERSESSION.md) marks these artifacts verification
pending. The new builder excludes every wall-labelled, wall-owned, PI-exposed or
schedule-unverifiable run in its entirety and hashes retained schedule evidence.
It preserves the original train/val world grouping, rechecks inherited PI keys,
hashes the input lists against their LPO receipt and refuses existing output dirs.
Missing schedules in direct-speed/damping packs also fail closed. A corpus with
no retained verified schedules cannot support retraining.

**Corrected real split counts: unavailable. Actual new real materialised sets:
0 wall_same, 0 wall_pilot, 0 seen_controls produced by this execution.** The
owner-supplied pilot inventory size of 200 directories has not been enumerated or
verified. Final selected/engaged/paired/rejected sizes come from the new receipts.

## Sets, controls, and scoring

- `wall_same`: removed train-origin arms in worlds still retained for LPWO
  training; only verified wall timelines can materialise. Nonengaged same-world
  arms retain an explicit exposure/engagement status, not an engaged-wall label.
- `wall_pilot`: explicit TRAIN chunk-4 pilot bindings, engaged outcomes only;
  refused/unassembled/unverifiable pilots excluded. Same-world pilots can overlap
  `wall_same`; report cohorts separately and do not sum them as independent states.
- `seen_controls`: corresponding retained train/val controller windows for
  successfully materialised world/t0 targets. If there are no corresponding
  windows, this set is honestly empty; do not manufacture controls.
- Every state needs real **10-step history**, t0 shift **9 physics ticks = 4.5 s**,
  finite nonnegative **[2,30] entrance counters**, a complete **120 s / 241-frame**
  future, wall-member IDs, ownership/assembly/hold/release-or-abort timeline.
  Births/slot reuse preventing complete initial-vehicle history are rejected.
- Native pairs require identical full physical initial states (IDs, seven kin
  fields, CAV flags, counters), not merely world/seed matching. A shifted t0 may
  already be post-intervention, causing genuine pairing rejection. Wall-only
  HDV scoring still works; paired delta ranking uses only verified pairs.
- `score_wall` reports HDV |ds| at **10/30/60/120 s**, wall-minus-native delay
  delta sign agreement and decision regret, structured-minus-blackbox comparisons,
  common-state denominators, world bootstrap, errors and unavailable arms.
  `wms_eval.ranking` is absent on this branch; the narrow two-arm regret rule uses
  existing `horizon_report` and `traffic_eval` helpers, with backend provenance.
- Structured and trajectory models use recorded-CAV teaching. Existing ID-blackbox
  learns CAV motion and ignores teaching; report it separately as an unequal-CAV
  contract. Do not advertise a teacher-matched three-model result.
- Existing LPO structured checkpoints saw walls and are **ineligible** for the
  unseen-wall claim. A structured checkpoint retrained on the same LPWO split is
  required before comparison. No model/rollout file was edited.
- Closed-loop wall execution requires the precise per-state session / stable ID
  / absolute time / 1 s controller vs 0.5 s plant callback described in
  [`SCORING_AND_CHAINS.md`](../output/wd01/wm_n1/lpwo/SCORING_AND_CHAINS.md).
  `--mode wall_exec` explicitly refuses until that hook is integrated.

## Guard and memory results

Canonical `protocols/holdout_exclusions.json` and `tools/holdout_guard.py` are
absent in this workspace. No external source was opened to obtain them. The
production CLIs require explicit staged copies; there is no fallback exclusion
set. `DataGuard` enforces workspace + allowed roots, TRAIN, excluded worlds,
CAL/HOLDOUT/sealed/chunk-5/J4B/INC1 denial before data opens, and calls canonical
`assert_final_holdout_eligible(worlds, data)` with supplied metadata to avoid
default external reads. Guard/tool hashes and checked paths/worlds enter receipts.
Synthetic refusal/order tests pass; **real canonical guard audit pending**.

Packs and identity arrays are `mmap_mode='r'`; one run/state is materialised or
scored at a time. `lpwo.bounded_cpu` disables CUDA visibility, bounds CPU threads,
samples each subprocess RSS every 50 ms, kills at **5,500,000,000 bytes** and writes
a new memory receipt. This is a sampled safety margin, not an instantaneous OS
quota. Use it for the commands below; real-data peak RSS remains unmeasured.

## Coordinator staging and exact real-data commands

The smallest next action is to **stage permitted TRAIN sources and canonical
guard files into this workspace**. This agent must not read the external originals.
Staging is a coordinator operation; never replace existing files or use junctions
to bypass workspace boundaries. Preserve source hashes and metadata bindings.

Required local layout under `_out/wallc1_inputs`:

- `canonical_exclusions.json`, `canonical_guard.py`: exact governance files from
  `E:/research-dense/protocols/holdout_exclusions.json` and `tools/holdout_guard.py`.
- `pack_v3`, `pack_dspeed`, `pack_damping`: approved packs with their INDEX,
  manifests and relative array filenames; `identities_v3`, `identities_dspeed`,
  `identities_damping`: matching canonical identity sidecars. Original paths are
  registered in `_out/lpo/BUILD_RECEIPT.json`; stage only approved TRAIN material.
- `runs/<campaign>/<run>/run/{switch_schedule.json,branch_record.json,wall_events.jsonl}`:
  verified per-run metadata for permitted LPO keys, retaining the INDEX `root_rel`
  bindings. Include permitted `campaign_walls_pilot_chunk4` runs. No directory
  traversal of other campaigns or sealed splits.
- C1FAST `full_bb_*` TRAIN_RECEIPTs for coordinator flag/config comparison; this
  branch verifies only local flags, not the unavailable external receipt identity.

Run these exact PowerShell commands with cwd
`E:/rd-wt/ws-wallc1/output/wd01/wm_n1`. Each output path must be new:

```powershell
$P = 'C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe'
$W = 'E:/rd-wt/ws-wallc1'
$I = "$W/_out/wallc1_inputs"
$S = "$W/_out/lpwo_verified_20261009"
$O = "$W/_out/wallc1_states_20261009"
$G = @('--guard-spec', "$I/canonical_exclusions.json", '--guard-tool', "$I/canonical_guard.py", '--allowed-root', "$W/_out")
& $P -B -m lpwo.bounded_cpu --receipt "$W/_out/WALL_SPLIT_MEMORY.json" -- -m lpwo --lpo-dir "$W/_out/lpo" --out $S --pack-v3-index "$I/pack_v3/INDEX.json" --extra-index "pack_dspeed=$I/pack_dspeed/INDEX.json" --extra-index "pack_damping=$I/pack_damping/INDEX.json" --runs-root $I @G
if ($LASTEXITCODE -ne 0) { throw 'split failed; preserve receipts' }
& $P -B -m lpwo --validate --lpo-dir "$W/_out/lpo" --out $S @G
if ($LASTEXITCODE -ne 0) { throw 'split validation failed' }
& $P -B -m lpwo.prepare_pilot_bindings --index "$I/pack_v3/INDEX.json" --campaign-root "$I/runs/campaign_walls_pilot_chunk4" --out "$W/_out/wallc1_pilot_bindings.json" @G
if ($LASTEXITCODE -ne 0) { throw 'pilot guard failed' }
& $P -B -m lpwo.bounded_cpu --receipt "$W/_out/WALL_SETS_MEMORY.json" -- -m lpwo.materialize_wall --wall-removed-windows "$S/wall_removed_windows.json" --retained-train "$S/windows_train_noPI_noWall.json" --retained-split "$S/windows_val_noPI_noWall.json" --pack "pack_v3=$I/pack_v3" --identities "pack_v3=$I/identities_v3" --pack "pack_dspeed=$I/pack_dspeed" --identities "pack_dspeed=$I/identities_dspeed" --pack "pack_damping=$I/pack_damping" --identities "pack_damping=$I/identities_damping" --pilot-index "$I/pack_v3/INDEX.json" --pilot-pack pack_v3 --pilot-bindings "$W/_out/wallc1_pilot_bindings.json" --out-dir $O @G
if ($LASTEXITCODE -ne 0) { throw 'materialisation failed; preserve failure receipt' }
```

After separately authorised retraining, stage LPWO `checkpoint.pt`, `physics.json`
and TRAIN_RECEIPT under `_out/wallc1_checkpoints/{structured,black_box_traj,black_box_id}`:

```powershell
& $P -B -m lpwo.bounded_cpu --receipt "$W/_out/WALL_SCORE_MEMORY.json" -- -m lpwo.score_wall --models "wms=$W/_out/wallc1_checkpoints/structured/checkpoint.pt" --models "blackbox_traj=$W/_out/wallc1_checkpoints/black_box_traj/checkpoint.pt" --models "blackbox_id=$W/_out/wallc1_checkpoints/black_box_id/checkpoint.pt" --sets "wall_same=$O/wall_same.json" --sets "wall_pilot=$O/wall_pilot.json" --sets "seen_controls=$O/seen_controls.json" --mode recorded_cav --out "$W/_out/wallc1_scores_20261009" @G
```

## Comparator chains and GPU estimate (prepared, not executed)

`output/wd01/wm_n1/lpwo/wall_bb_chain.sh` is LF and takes explicit absolute host
paths: `PY CODE OUT TRAIN_WINDOWS VAL_WINDOWS EXCLUDED_WINDOWS BASE_PACK
BASE_IDENTITIES TRAIN_OUTCOMES DSPEED_PACK DSPEED_IDENTITIES DSPEED_OVERLAY
DAMPING_PACK DAMPING_IDENTITIES DAMPING_OVERLAY`; `GPU=gpuA|gpuB` is a host label.
Both four-stage smoke gates must pass before full runs. Full stage steps are
**2500/2500/1875/1250 = 8125 per model**, horizons **10/30/60/120 s**;
local flags `--blackbox-traj` and `--black-box`. Default zero loader workers,
four threads, explicit lists/physics; absolute total `MAX_HOURS` cap defaults to
**8 h for both models combined**, including smokes. No automatic rerun.

Planning estimate for two black boxes: **4–8 GPU-hours total**, conditional on
weighted mean step latency **0.882–1.764 s** over the exact allocation of
**16330 steps = 16250 full + 80 smoke**. This is a planning assumption, not
measured throughput. Use smoke
stage latency to project `sum(stage_steps * mean_stage_seconds)` for each model;
abort if the remaining hard cap is insufficient. Historical `_out/C1.md` estimated
11–12 h for a different adaptive-budget two-model chain and cannot validate this
fixed-step estimate. The hard cap can legitimately stop this chain before finish.

## Verification and recordkeeping

Focused synthetic LPWO tests, existing C1 materialisation/horizon regressions,
Ruff, LF contract checks and ledger consistency/summary checks are the required
local checks. Final command/results and sampled RSS are recorded in the memory
receipt and task record. No source-value efficacy claims were added; scoped
scientific source verification for this engineering record has zero outcome
sources. Full historical source verification is separate from this task.

Verified: **136 new LPWO tests**, **42 existing C1 regression tests**, scoped Ruff
and `git diff --check`; sampled final test RSS **213,884,928 bytes** per process.
The metadata-only ledger selection has **10 passed / 1 failed / 16 deselected**:
historical referenced evidence (including CUR-02 adoption_cleanup and CUR-03
escalation_test_build) is absent on this worktree. This pre-existing completeness
failure remains unresolved; frozen evidence was not modified to satisfy it.
RESULTS_SUMMARY regeneration/check passed. The first combined verification hit a
120 s tool timeout; the next recorded run found three cwd-dependent test-fixture
assumptions and the corrected focused suite passed. Receipts are preserved in
[`WALL_C1_VERIFICATION.json`](WALL_C1_VERIFICATION.json).

`lpwo.closeout_check` reports **1560 historical hard errors**, exactly **1560**
on the pre-task HEAD ledger, **0 introduced errors**, **7 warnings**, and entry-
scoped source verification **0 sources / 0 expected / 0 performed / 0 mismatches**
(engineering/no outcome sources). The global zero-hard-error close-out criterion
is therefore blocked by unstaged historical evidence; this task does not repair
or reinterpret those frozen records.

Master ledger, README, ROUTE, INDEX and OPEN_QUESTIONS carry the stable ID and
partial status; interrupted artifacts remain preserved. This preparation does
not yet establish unseen-wall predictive generalisation.

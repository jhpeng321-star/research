# WALL-C1 scoring and LF comparator chain

Implementation/verification only. No chain execution, GPU, training, SUMO or
real-state/checkpoint scoring was performed for this task. External receipts
were unavailable under the workspace-only read constraint. **Flag identity is
pending outside receipt confirmation.** This document is the scoped task record;
it does not assert scientific outcomes or a completed physical wall comparison.

## Scoring contract

From `output/wd01/wm_n1`, the authored CPU scoring CLI is:

```powershell
& C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -B -m lpwo.score_wall `
  --models wms=path/to/structured/wm_s.pt `
  --models blackbox_traj=path/to/traj/wm_s.pt `
  --models blackbox_id=path/to/id/wm_s.pt `
  --sets wall_c1=path/to/materialized_wall_states.json `
  --sets seen_controls=path/to/materialized_seen_controls.json `
  --guard-spec path/to/spec.json --guard-tool path/to/guard_tool.py `
  --allowed-root path/to/inputs --allowed-root path/to/outputs `
  --mode recorded_cav --out path/to/new_wall_score
```

All paths resolve within this workspace. Every checkpoint requires an explicit
sibling `physics.json` with `dt_s=0.5`; the scorer refuses missing physics rather
than using the adapter's defaults. It instantiates the existing
`WmsCheckpointModel` adapter with explicit physics for all three model kinds
(the generic blackbox loader drops the physics override). The report retains
this configuration. Input is a streamed C1 JSON object
with `windows`, `states` or `records` array, using `c1v2.iter_windows`; at most
one raw window plus its current trajectories is resident. Compact per-state
summaries are retained for reporting. No bounded-memory claim is made for an
unbounded number of summary rows. `--limit N` is optional and positive.

Each window requires explicit `world`, `t0_s`, an inline initial `state` or
`initial_state`, and at least one inline `recorded` trajectory. Wall-only records
are scored on wall: missing native is **unavailable**, not a rollout error or an
input refusal. `seen_controls` records are scored using their exact own arm
(`native`, `base`, `arm0`, etc.) and its matching recorded trajectory; base/arm0
are not silently relabeled native. Vehicles use the local
`id/s/v/lane/present/is_cav` schema. The own-arm initial frame must equal the
supplied initial snapshot. A different-initial sibling is unavailable and cannot
enter paired ranking. Native/wall ranking requires both successful arms with
identical initial snapshots and no explicit `paired=false` declaration (which
may encode additional full-physical-state mismatches from materialisation).
Cross-model shared identity is exact `(world,t0_s,key,source_arm)`; duplicates of
that complete identity are errors, while different keys or source arms at the
same world/t0 are distinct states. Source arm uses `source_arm`, then `arm`, then
the inferred own recorded arm. Raw/path-backed
states (`recorded_paths`, `base_run_dir`, `state_row`) are refused. Trajectories
must cover the complete 0.5 s grid through 120 s; longer recordings are cropped.
Short or missing trajectories cannot silently trigger hold-speed teaching or
nearest-frame extrapolation. Predicted grids and finite values are checked too.

### Production DataGuard boundary

The CLI requires `--guard-spec`, `--guard-tool`, and repeatable `--allowed-root`.
Each bootstrap path and allowed root must resolve inside this workspace. It
constructs the shared production interface
`lpwo.guard.DataGuard(spec: Path, tool: Path, allowed_roots: list[Path])`.
The scorer never substitutes its synthetic test guard in production:

- Preflight `check_path` for **every** checkpoint, sibling physics file and state
  set before the first data open. Recheck at adapter construction and each
  streamed-set open. Guard denials returning `False` or raising are fatal.
- Physics JSON is read through `DataGuard.read_json` after its path check; state
  arrays retain the streaming reader after `check_path`, rather than loading
  large sets via `read_json`.
- Call `check_record` on each streamed state before validation or any predictor
  call. Record guard exceptions are outside the per-arm model-error handler and
  cannot be swallowed or retried as native. Inputs remain inline-only.
- Output paths are checked before writes. `main` calls `audit()` on success or
  refusal once guard construction has succeeded, prints the JSON audit, and
  writes `GUARD_AUDIT.json` only if this invocation created its output directory.
  A rejected pre-existing output is never modified. Audit failure returns 2.

The integrated `lpwo/guard.py` requires explicitly staged canonical exclusion
metadata and guard code, checks workspace/allowed-root boundaries, enforces
TRAIN-only source records, and invokes `assert_final_holdout_eligible(worlds,
data)` with the supplied data. There is no default external-file read or fallback
exclusion set. Ordering and actual guard behavior are verified synthetically;
real canonical sources still need staging and identity verification.

The initial mode is **recorded-CAV/exogenous**, with the existing local model
contracts preserved:

- `wms`: HDV physics + residual/driver parameters; recorded CAV teacher.
- `blackbox_traj`: learned absolute HDV dynamics, no IDM/MOBIL base; same teacher.
- `blackbox_id`: physics + residual and CAV controller-ID embedding; learned CAVs,
  no controller code and no CAV teaching. This is a secondary unequal-CAV-contract
  comparator, not the teacher-matched HDV-mechanism contrast. Its actual arm
  string is fed to the local controller vocabulary: `wall` is wall, `native` is
  native, and `base`/`arm0` map to `other`, not a native fallback.

Replacing CAVs after a completed rollout would not teacher-force their influence
on HDVs. The scorer does not claim that operation establishes a fair ID-model
teacher comparison. The existing ID vocabulary includes `wall`; its embedding
may be untrained when wall is excluded.

The scorer calls each requested arm directly. It does **not** use
`c1v2._predict` or `c1v2.evaluate_state`: `_predict` can retry unsupported wall
arms as native. Unsupported predictors produce explicit per-arm errors and
missing paired rows; the CLI writes summaries then returns 2 for any arm errors.
Input refusals also return 2. Existing output directories are never reused;
partial output files after interruption/refusal are retained.

Outputs are `states__MODEL__SET.json` in the C1 per-state layout and
`WALL_SCORE.json`. HDV median absolute `ds` (metres) at 10/30/60/120 s uses
`car_metrics`, `c1v2._car_compact` and `horizon_report.horizon_block`; summary
includes median, IQR, p90, population SD, median within-state p90 and explicit
blow-up denominators. Wall HDV summaries and shared wall HDV comparisons include
**all valid wall trajectories, paired or unpaired**. Seen-control summaries keep
their own arm names. Arm-error and arm-unavailability counts are separate; a
wall-only/seen-only scoring run can return 0 with no rankable native/wall pairs.
The headline arm is explicitly `wall`; the scorer does not
use `horizon_report.tested_arm`, which would prefer native for this set.

### Paired decision metric

Cost is all-section normalized delay seconds over `[t0,t0+h)`, computed with
the existing `traffic_metrics` frame operations, 0.5 s integration, and
`v0_mps` (default local evaluator value 13.89). It includes all present HDVs
and CAVs, but closed initial-slot rollout misses later arrivals. This is a
short-horizon offline surrogate, not full requested-service cost, safety or
wall efficacy. No entry flag is enabled.

For each horizon, `delta = wall cost - native cost`; negative is improvement.
Predicted negative delta selects wall; ties within 1e-9 select native. Regret is
`recorded cost(selected) - min(recorded native cost, recorded wall cost)`.
Sign agreement includes ties (positive/zero/negative with 1e-9 sign tolerance).
Every model's ranking statistics use the same complete-pair state intersection;
union/shared/valid/excluded counts are explicit, and recorded deltas must agree
across models. Across-model comparisons are structured minus each blackbox;
negative regret/error differences favor structured. Bootstrap resamples whole
worlds using `horizon_report.paired_bootstrap` (median paired differences,
not a difference of unpaired medians). Bootstrap regret scalars are passed
through its generic compact `ds_med` slot; units remain regret seconds.
That shared helper keys only by world/t0, so after exact four-part source
matching, the scorer substitutes unique **summary-only** indices for t0 to
prevent same-world/same-time source arms collapsing. World grouping remains the
original world; physical trajectories/timestamps are never altered.

**Local module discrepancy:** `wms_eval.ranking` does not exist in the inspected
worktree. No API or successful import is invented. `score_wall.rank_pair`
implements the narrow two-arm regret rule; sign/Spearman reuse
`wms_eval.traffic_eval`. Report metadata identifies that backend. Integration
with a future shared ranking module requires its actual API and parity checks.

## Required `wall_exec` rollout hook (not implemented here)

`--mode wall_exec` refuses before loading files or models. A wall cannot be
implemented by changing the arm string or replaying recorded future motion.
The required integration points are:

1. `wms_eval.wms_plugin.WmsCheckpointModel.predict_states` must explicitly
   select recorded teaching versus executable control and create a per-state,
   resettable wall-session object with stable vehicle-ID/slot mapping. Native
   must have an explicit executable-control comparator too. In executable mode
   do not call `_teacher` for future CAV motion and do not retain future HDV
   presence/junction masks as hidden input. Recorded futures remain scoring
   labels only.
2. `wms.model.WmSModel.rollout` currently supplies `cav_controller(current)`
   through `step`. Extend/adapt that callback with absolute `t0_s + step_i*0.5`,
   stable IDs, ownership, session lifecycle, and report-causal simulated sensor
   data. Tick the wall supervisor/command ownership once per 1 s control interval,
   persist held commands/session state across the two 0.5 s physics steps and
   across chunk continuations. Do not mutate the shared record or model between
   arms; reset controller history per counterfactual.
3. `WmSModel.step` expects CAV next-state `[B,K,4] = s,lane,v,a`. Supply the
   wall/native controller's **issued** longitudinal command through the same
   safety/plant response semantics used by the modeled physics to obtain that
   next state, not a target-speed teleport. Track issued versus modeled/measured
   responses separately. The present callback alone replaces all CAV next states
   and is not a verified wall ownership/plant bridge.
4. Wall owns only its roster, with heterogeneous nonmember CAV actions; sole
   command ownership, assembly/hold/taper/abort/release and identical-instance
   handback must persist. Any acceptance uses feasibility and incremental benefit
   for the same proposal/report evidence. Runtime inputs are timestamped CAV and
   sensed-neighbor reports with age/missingness/coverage, not full-state truth.
5. Fail explicitly on unsupported action/callback or missing wall metadata;
   add rollout-level replay, cadence, ownership, handback, plant-delivery and
   no-future-leakage tests before enabling this mode. Blackbox-ID must retain its
   learned action-feature contract; a teacher-matched ID comparison needs an
   independently specified model/step capability because its current `step`
   intentionally ignores the CAV teacher/callback in black-box mode.

## LF comparator chain (authored only)

`wall_bb_chain.sh` accepts `GPU=gpuA|gpuB` as a host label, never contacts either
host and does not choose a CUDA index. The caller supplies absolute staged-host
paths via `PY`, `CODE`, `OUT`, `TRAIN_WINDOWS`, `VAL_WINDOWS`, `EXCLUDED_WINDOWS`,
`BASE_PACK`, `BASE_IDENTITIES`, `TRAIN_OUTCOMES`, `DSPEED_PACK`,
`DSPEED_IDENTITIES`, `DSPEED_OVERLAY`, `DAMPING_PACK`, `DAMPING_IDENTITIES`,
`DAMPING_OVERLAY`. Lists are the LPWO train/val/excluded artifacts produced by
the separate builder; the chain does not read external receipts or regenerate
splits. `PY` is required, not guessed from a host path. `OUT` must be new with
an existing parent. No arbitrary extra-flag string can replace model/list flags.

Based on local `wms_bb_chain.sh`, trainer CLI and model factories:

- `black_box_traj` uses **`--blackbox-traj`** (teacher CAVs, learned HDV base).
- `black_box_id` uses **`--black-box`** (learned CAVs plus ID; HDV physics remains).
- Both use the exact same three packs, identities, overlays, outcomes, LPWO
  lists, batch size 8, SUMO-fidelity *model flag*, start-shift 0.75,
  truncation 20, and 10/30/60/120 s horizons. The model flag does not invoke SUMO.
- Full curriculum is **2500/2500/1875/1250** optimizer steps, 8125 per model.
- Each smoke exercises all four horizons with 10 steps/stage (40/model).
  Both smokes must pass before either full job starts. `gate_stages` checks
  stage coverage/finite fields/dominance/growth; an additional inline Python
  gate checks exact per-stage counts, every row's required finite losses,
  valid targets, unroll lengths, monotone finite gate curves and g(0)=0.
  The checked fields match the local trainer's `_forward_backward` loss/gate
  record, `train_batch` finite/seconds fields, and `fit` stage/step fields. The
  gate's required terms are only position/speed/lane because auxiliary losses
  are disabled in this chain. ID `valid_targets` counts learned CAVs as well as
  HDVs; it is not mislabeled as an HDV-only target count. Synthetic records with
  that schema pass both gates; missing stages/terms, NaNs and empty targets fail.
- `MAX_HOURS` defaults to 8, must be finite in `(0,8]`, and caps **total chain
  elapsed time**, including both models, smokes, gates and validation. Each
  subprocess is wrapped in GNU `timeout --signal=KILL` with the remaining
  deadline, without kill grace/foreground mode; loader process groups are killed
  on timeout. Exhaustion stops the chain, never starts a successor. No estimated
  runtime substitute or automatic step resizing. A small caller-supplied cap can
  therefore leave a partial first or second full model; receipts/logs disclose it.
- `set -euo pipefail`, fresh output paths, allocation/start log entries, per-job
  logs and EXIT receipt preserve failure status; no retries or resume behavior.
- Successful training captures `TRAIN_RECEIPT.json`'s exact physics as a sibling
  `physics.json` for scoring, under the same remaining-time cap.
- CPU controls: `CPU_THREADS` defaults to 4 and is restricted to `[1,4]`, passed
  to trainer `--threads` and OMP/MKL/OpenBLAS/NumExpr environments. The local
  trainer sets `torch.set_num_threads(config.threads)` and PyTorch loader workers
  use their own single-thread setting. `NUM_WORKERS` defaults to **0**, with an
  explicit override restricted to `[0,4]`; only one training runs at a time.
  This intentionally supersedes the earlier inherited 10-worker setting: local
  trainer prefetch_factor=4 would queue up to 40 batches with 10 workers plus
  process/dataset copies. Default zero avoids loader child/process queue overhead.
  An override requires a separately allocated aggregate host-RAM budget. Neither
  these bounds nor per-process sizes prove a 6 GB aggregate RSS bound; RAM use,
  pinned/cache overhead and CUDA-host allocations remain unmeasured, with that
  limitation recorded in chain logs. No training was run to measure them.

Before real scoring, stage checkpoints and their exact training physics
configuration locally. No outside receipt
confirmation, LF training, finite smoke outcome, wall-state materialization,
trained checkpoint identity or scientific score is claimed by this task.

## Verification record

Synthetic scoring and static chain tests live under `lpwo/tests`. Run from the
workspace root with the direct environment Python:

```powershell
& C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -B -m pytest -q -p no:cacheprovider `
  output/wd01/wm_n1/lpwo/tests/test_wall_score.py `
  output/wd01/wm_n1/lpwo/tests/test_wall_chains.py
```

Targeted acceptance follow-up completed 2026-10-09:

- The pytest command above: **59 passed**, 1 warning, 9.42 s. The warning is
  `Unknown config option: cache_dir` because cacheprovider is deliberately
  disabled; no pytest cache is written.
- `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -B -m ruff check`
  on `score_wall.py` and the two test files: **All checks passed**.
- Direct Python `-B -m lpwo.score_wall --help`, with working directory
  `output/wd01/wm_n1`: **exit 0**, including the three required guard arguments.
- CLI orchestration is tested against synthetic predictors and an in-memory
  filesystem; three model kinds receive explicit physics in adapter-contract
  tests without loading a checkpoint. Scoring tests independently check the
  delay sign, HDV/CAV separation, wrong-selection regret, ties, shared-state
  bootstrap, excluded/error denominators, duplicate/label refusals, complete
  grids, streaming reads, missing physics and unsupported-wall behavior. Follow-up
  coverage includes wall-only and native/base/arm0 own-control scoring,
  nonidentical/native-unavailable pairing, explicit nonpair preservation,
  source-aware shared/bootstrap identity, checkpoint/physics/set preflight denials,
  required/absent guard refusals, record guard ordering, and final audits without
  modifying pre-existing outputs. An initial follow-up test run was 58 passed /
  1 failed because its synthetic native frame aliased the initial-state fixture;
  the fixture was independently copied, then the full suite passed above.
- Static chain tests inspect local trainer argument/log declarations via AST,
  parse embedded Python gates, and run the pure checker against **in-memory
  synthetic log rows only**, without importing/running training. Gate stages
  also accept the synthetic local schema; missing terms/stages, NaNs and zero
  targets are rejected by the inline checker. Thread/worker bounds are checked.
  Bash/chain
  execution, host availability, CUDA, timeout behavior on a remote host, real
  data/checkpoint scoring and external receipt identity remain unverified.

Only the five owned scoring/chain/test/document files were authored. Shared
records and READY files were not edited; no scientific conclusion changed and
no experiment allocation was spent.

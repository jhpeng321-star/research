# WM-S second integration — engineering ready

- Verification ID: **WMS-MAIN2-INTEGRATION-20261009**.
- Branch: `wms/main2`; starting HEAD: `ec6f948` (`wms/main`). Clean starting worktree.
- Integration/code receipt: `a5f27fe`. All three requested branch tips are ancestors.
- CPU only, direct interpreter `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`.
- No GPU, SUMO, remote execution or real-data training/scoring was performed.
  No CAL, HOLDOUT, sealed, chunk-5, J4B or INC1 data was opened. All task outputs
  and edits are inside this workspace. Two narrow helpers handled the stage-gate
  conflict and chain implementation/fixture tests.

## Merges and conflict resolutions

1. `wms/brakefix` (`7a924b5`) → **`0b81d0d`**. Conflicts in `wms/train.py`
   and `wms_eval/wms_plugin.py`: retained switch sampling, all rollout/model kinds,
   receipt projection/symmetric caps, and the new braking losses/asymmetric cap.
2. `wms/trainsetup` (`56a7d61`) → **`7f73874`**. Conflicts in `.gitattributes`,
   README/INDEX/ROUTE, and `wms/{cli,gate_stages,model,plan_steps,train}.py`.
   Preserved both documentation records; combined CLI/config contracts; retained
   five-stage gate default with explicit four-stage selection; retained finite
   timing guards. Callback dispatch occurs once per physics tick with the existing
   action/episode-clock fields. Entrant padding preserves original timeline columns.
   Default loss/log calculations retain the earlier horizon-normalized/probe path;
   opt-in closed-loop diagnostics capture the training pass, avoiding a second
   stateful-controller replay. Mixed sampling, zero-shot exclusions, timeline
   loading and leave-PI-and-wall-out runtime filtering remain enabled by their flags.
3. `wms/wallc1` (`f49e4fa`) → **`1e1d808`**. Conflicts in attributes/navigation,
   `docs/RESULTS_SUMMARY.md` and `protocols/experiment_history.json`: retained both
   additive engineering records and regenerated the mechanical summary.
4. The supplied branch tips advanced during integration. Clean follow-up merges:
   `wms/brakefix` **`8cca2f3` → `7b4dabf`** (mmap audit dtype-token compatibility)
   and `wms/wallc1` **`9636c49` → `ffce12d`** (inherited TRAIN split/binding receipts).
   Inherited real-data reports were preserved, not rerun or used for model selection.
   Six Ruff issues in the three incoming staging/analysis scripts were repaired
   through unused-import removal and variable renaming; missing binding keys still fail.

No reset, rebase, stash, clean or destructive checkout was used.

## Unified launch chain and compatibility

`_out/train_main_chain.sh` supports the same original and switching packs, original
LPWO lists, switching train/monitor lists, exclusions and zero-shot metadata for
all three model kinds. Each invocation uses a new `W`; smoke and full run use the
same contract. Four stages remain 10/30/60/120 s, full steps 2500/2500/1875/1250,
smoke steps 6/6/6/6. Existing disk, timing, fresh-output and failure-preservation
checks remain; `NUM_WORKERS=0` is mandatory and training subprocess RSS is guarded.

- `MODEL_KIND=structured` default; `blackbox_traj` adds `--blackbox-traj`,
  `blackbox_id` adds `--black-box`. The inherited `lpwo/wall_bb_chain.sh` is retained.
- `ROLLOUT_MODE=residual` default, `hybrid` optional; `LANE_HEAD=off` default,
  `history` requires hybrid in this chain. BB-traj requires residual/off because
  its model factory does not implement hybrid/history.
- Braking env defaults: `ASYM_SPEED_LOSS=1`, `DECEL_WEIGHT=1`, `DECEL_THRESHOLD=3`,
  `PROPAGATION_LOSS=0`, `PROP_WINDOW=3`, optional `PROP_MAX_GAP_M` and
  `RESIDUAL_CAP_NEG_MPS2` empty. New braking terms/cap default OFF; their default
  receipt fields are omitted. Enabled braking requires structured WM.
- Existing `FLAGS` and `BRAKING_FLAGS` remain supported. Effective precedence is
  env values, then `FLAGS`, then `BRAKING_FLAGS`; contract/model/mode/head
  overrides are rejected. Smoke shares and every smoke step require the enabled
  braking terms, in addition to the existing losses.
- Frozen LPWO lists declare `TRAIN` per window rather than at the root. The
  chain accepts their exact schema/list names only with nonempty, explicitly TRAIN
  rows and exact counts, preserving receipt-pinned list bytes. Wrong splits,
  missing row splits and ambiguous timeline vocabularies fail closed.
- Exact loss, deterministic log and gradient parity is tested in residual,
  driver_params and hybrid against both `6ebcf95` and **`ec6f948`**. The historical
  baseline's absent switch-sampler receipt field is handled explicitly. Existing
  live-action/checkpoint compatibility tests remain in the full panel.

## Final verification (exact)

- Required MAIN_MERGE panel: **612 passed, 6 skipped, 1 warning**, 332.07 s;
  peak OS process working set **891,092,992 bytes**.
- Entire LPWO suite: **136 passed**, 18.86 s; peak **212,439,040 bytes**.
- Standalone chain guards: **28 tests passed**, 83.187 s; peak **27,295,744 bytes**.
- Ruff: **All checks passed** across WMS, WMS-eval, LPWO, selected main evaluation
  regressions, chain tests and task tools. Both launch scripts pass Git Bash `-n`.
- `git diff --check` and staged whitespace check pass. Raw failure logs/JUnit XML
  retain exact traceback whitespace through narrowly scoped `-whitespace` attributes.
- `*.sh text eol=lf` is committed. `git ls-files --eol -- "*.sh"`: **14/14 `i/lf`**;
  some inherited working copies are CRLF, but all committed/archive scripts are LF.

The warning is the existing read-only NumPy-view-to-Torch conversion in
`train/idea_a.py`. Six optional skips are the two absent TRAIN chunk-4 real-state
fixtures and four absent staged wall-source reference cases, as in MAIN_MERGE.
No real packs were loaded. Production pack readers retain `mmap_mode='r'`; audit
and mixed-window readers process bounded windows/runs, and archives stream files.
All measured Python processes stayed below 6 GB.

Exact final invocations, from the workspace root:

```powershell
$env:TMP="E:/rd-wt/ws-main2/_out"
$env:TEMP=$env:TMP
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:CUDA_VISIBLE_DEVICES=""
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/brakefix_verify.py --tag main2_final_tips -- -c "import glob,pytest,psutil; code=pytest.main(['output/wd01/wm_n1/wms/tests','output/wd01/wm_n1/hdv/tests',*sorted(glob.glob('tests/test_wms_eval_*.py')),'-q','--basetemp=_out/pytest_main2_final_tips','--junitxml=_out/MAIN2_TESTS_FINAL.xml']); print('Peak process working set bytes:',psutil.Process().memory_info().peak_wset); raise SystemExit(code)"
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/brakefix_verify.py --tag main2_lpwo_final -- -m pytest -q output/wd01/wm_n1/lpwo/tests --basetemp=_out/pytest_main2_lpwo_final --junitxml=_out/MAIN2_LPWO_FINAL.xml
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/brakefix_verify.py --tag main2_chain_final -- -B _out/test_train_main_chain.py
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m ruff check output/wd01/wm_n1/wms output/wd01/wm_n1/wms_eval output/wd01/wm_n1/lpwo tests/test_wms_eval_main_compat.py tests/test_wms_eval_main_switch.py tests/test_wms_eval_paths.py _out/test_train_main_chain.py tools/brakefix_verify.py tools/main2_archive.py tools/main2_records.py
```

Tags/basetemps are historical exact invocations; choose new ones for reproduction.
Logs/memory receipts are committed under `brakefix_verification/main2*/`; JUnit
receipts are `MAIN2_TESTS_FINAL.xml`, `MAIN2_LPWO_FINAL.xml` and the earlier attempts.

Earlier checks are preserved: initial full panel **6 failed, 602 passed, 6 skipped,
3 warnings** (371.81 s); focused repairs plus wall suite **285 passed, 1 warning**
(113.57 s); pre-follow-up-tip full panel **611 passed, 6 skipped, 1 warning**
(295.13 s); initial standalone chain **27 passed** (99.764 s). Repairs declared
two synthetic timeline schemas, corrected an obsolete cap-replay expectation,
and extended rather than removed exact parity tests.

## Recordkeeping limits

Both inherited engineering ledger entries remain additive and their scientific
outcomes unchanged. `summary --check` passes after regeneration.
[`MAIN2_RECORD_CHECKS.json`](MAIN2_RECORD_CHECKS.json) compares metadata validation
against `ec6f948`: **1,560 baseline and 1,560 current hard errors; zero new errors**.
Those errors are absent historical evidence paths, not this integration's code.
Scoped `verify_sources` for the two merged engineering IDs is **not applicable**:
zero registered outcome sources, zero performed source checks, no historical
outcome sources opened. It is not represented as successful scientific revalidation.

Metadata-only ledger regressions: **2 failed, 18 passed, 7 deselected**, 1.94 s.
Failures are `test_all_referenced_paths_exist` and
`test_validator_reports_no_hard_errors`, matching the pre-existing missing paths.
Source-opening tests were deselected to preserve this task's narrow data scope.
Exact command is committed in
`brakefix_verification/main2_ledger_metadata/verification.json`.

## HEAD code archives and SHA256

Artifacts: `_out/main2_archives_release/` (tar files local; manifests committed).

- `wm_n1_code.tar`: **11,939,840 bytes**, **762 files**, SHA256
  `b465f58fa62d89df54ddadd664e05adbe4bf44ad7f3e6b858cec3eca78b46b08`.
- `main2_chains.tar`: **102,400 bytes**, **14 files**, SHA256
  `171f1668e5aeefe56ae57cba269ebb06ae393a351121be85692ef2506a790a92`.
- Every member matches its HEAD Git blob. All **9** code-tar `.sh` members and
  **14** chain-tar `.sh` members contain **zero CR bytes**.

The builder uses `git -c core.autocrlf=false -c tar.umask=0022 archive --format=tar
--mtime=2026-10-09T00:00:00Z --output=... HEAD^{tree} ...`. HEAD's root tree and a
fixed mtime deliberately avoid commit-ID PAX self-reference: this READY/hash
documentation commit does not change archive bytes. Final-HEAD verification
checks the archived files against HEAD again. Code tree is
`579d753cf038183a91605120005c091e835a0af4`.

The first archive allocation `_out/main2_archives/` failed blob verification
because Windows autocrlf converted non-attributed text. It is preserved with
`FAILED.json`; corrected archives use a fresh directory and a per-command config
override, without changing Git configuration or frozen source bytes.

```powershell
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" tools/main2_archive.py --out _out/main2_archives_release --verify
git ls-files --eol -- "*.sh"
git branch --merged HEAD --list wms/brakefix wms/trainsetup wms/wallc1
```

## Four exact training launch commands — prepared, not executed

Run from a host with the baseline repository/dependencies and permitted TRAIN
packs already staged. Extract both archives over that repository at `CODE_ROOT`;
the scoped archives supplement the existing baseline checkout. The named switching
monitor file must be TRAIN-scoped and world-disjoint from its train list. The same
original/switching train/monitor/exclusion inputs and seed are used for all four
jobs. Pack/identity/overlay defaults are those in `train_main_chain.sh`; provide
identity-equivalent overrides if the staged host layout differs. `merge_params.json`
and TRAIN outcomes/identity sidecars must already be present. Every `W` below must
be absent and `/root/autodl-tmp/wms_train_main` must exist.

```bash
export BOX=gpuA ORIGINAL_ROOT=/root/autodl-tmp
export PY=/root/autodl-tmp/gpu1/env/rd_gpu/bin/python
export CODE_ROOT=/root/autodl-tmp/wms_main2
export SW_ROOT=/root/autodl-tmp/wmn1_switching_train
export WINDOWS_TRAIN="$CODE_ROOT/output/wd01/wm_n1/lpwo/real_20261009/windows_train_noPI_noWall.json"
export WINDOWS_VAL="$CODE_ROOT/output/wd01/wm_n1/lpwo/real_20261009/windows_val_noPI_noWall.json"
export EXCLUDE="$CODE_ROOT/output/wd01/wm_n1/lpwo/real_20261009/excluded_runs_PI_wall.json"
export SW_WINDOWS_TRAIN="$SW_ROOT/windows_train_hetdata.json"
export SW_WINDOWS_VAL="$SW_ROOT/windows_monitor_hetdata.json"
export OMP=2 NUM_WORKERS=0 MAX_HOURS=10 CAV_DRIVE=teacher
export ROLLOUT_MODE=residual LANE_HEAD=off FLAGS="" BRAKING_FLAGS=""
export ASYM_SPEED_LOSS=1 DECEL_WEIGHT=1 DECEL_THRESHOLD=3
export PROPAGATION_LOSS=0 PROP_WINDOW=3 PROP_MAX_GAP_M="" RESIDUAL_CAP_NEG_MPS2=""

# 1. Structured WM, braking fix OFF.
MODEL_KIND=structured W=/root/autodl-tmp/wms_train_main/main2_structured_no_brake \
  bash "$CODE_ROOT/_out/train_main_chain.sh" run

# 2. Same structured WM/split, braking fix ON (untuned initial weights).
MODEL_KIND=structured ASYM_SPEED_LOSS=2 DECEL_WEIGHT=5 DECEL_THRESHOLD=3 \
  PROPAGATION_LOSS=1 PROP_WINDOW=3 RESIDUAL_CAP_NEG_MPS2=3 \
  W=/root/autodl-tmp/wms_train_main/main2_structured_brake \
  bash "$CODE_ROOT/_out/train_main_chain.sh" run

# 3. Teacher-matched learned-HDVs black box, same split.
MODEL_KIND=blackbox_traj W=/root/autodl-tmp/wms_train_main/main2_bb_traj \
  bash "$CODE_ROOT/_out/train_main_chain.sh" run

# 4. Controller-ID black box (learned CAVs), same split.
MODEL_KIND=blackbox_id W=/root/autodl-tmp/wms_train_main/main2_bb_id \
  bash "$CODE_ROOT/_out/train_main_chain.sh" run
```

Each `run` executes preflight, four-stage smoke, loss-share/finite/growth gates,
timing projection and only then the fixed full budget. For a metadata/help-only
preflight substitute `preflight` for `run` with the same env values. To select the
optional hybrid/history structured pair, set `ROLLOUT_MODE=hybrid LANE_HEAD=history`
for both structured jobs; leave BB-traj at residual/off. Suggested braking weights
are launch settings, not measured efficacy or tuned acceptance thresholds.

# WM-S main integration close-out

- Engineering verification ID: **WMS-MAIN-MERGE-20261009**.
- Branch: `wms/main`; starting base: `wms/hybrid` (`6ebcf95`).
- Status: all seven requested branches integrated in order; CPU implementation
  acceptance complete. No scientific conclusion or inherited result was
  recomputed. No SUMO, GPU, remote execution or real-data scoring was performed.
- Interpreter: `C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe`, Python
  3.10.20, Torch 2.12.1+cpu. This resumed execution used two narrowly scoped
  helper agents for implementation and synthetic compatibility tests.

## Resume audit and merge sequence

The cancelled execution had already committed the first three merges and was
mid-merge of `wms/evalshift`. Four files remained unmerged in the index, although
their worktree conflict markers had been removed. The extra unstaged
`blackbox_traj.py` compatibility work was reviewed, retained and verified.
No reset, clean, stash, rebase or destructive checkout was used.

1. **`wms/switch` — `8d2370f` (already present on resume).**
   Conflict-resolution paths: `wms/model.py`, `wms/train.py`, `wms/cli.py`,
   `wms_eval/wms_plugin.py` (under `output/wd01/wm_n1/`). The existing merge
   retained residual / driver_params / hybrid, history lane head, the run-3
   projection and entry-speed fix, action timeline, indexed identity binding
   and live native/damping/PI execution. Reviewed and repaired integration
   issues below. The earlier executor's per-merge test count was not recovered;
   this session revalidated these features in the 364-pass acceptance panel and
   final full suite.
2. **`wms/wallexec` — `dc35a81` (already present).**
   New wall executor and tests; no implementation merge conflict. Existing
   follow-up `e03cfb4` made the reference source configurable. This session
   separated the four reference-dependent cases from four standalone synthetic
   executor cases, so missing source no longer skips the entire module.
   Revalidated by the final suite; optional reference limitations are below.
3. **`wms/eval2` — `9d5c5a1` (already present).**
   Plugin conflict resolved with receipt physics, bounded residual/lane caps,
   projection, entries-v2, per-worker thread cap, all rollout modes and lane head
   retained. `c1v2` labelled models, per-horizon overlap counts and run-3 horizon
   report retained. Revalidated by checkpoint/receipt tests and final suite;
   no earlier executor test count is claimed.
4. **`wms/evalshift` — `2545c3d` (resumed and completed).**
   Conflicts: `wms_eval/c1v2.py` (add/add), `cli.py`, `models.py`,
   `wms_plugin.py`. Kept the eval2 `c1v2` implementation and its overlap/receipt
   features; unified entries flags in CLI; retained all model specs. Plugin
   combines ID-keyed shifted history with slot-indexed lane/hybrid history,
   live controllers, receipt projection/caps and legacy checkpoint loading.
   Shift materialization/runbook and inherited reports merged as provided.
   Acceptance: **364 passed, 6 skipped** after repairs.
5. **`wms/ranking` — `da3e6fb`.**
   CLI dispatch conflict resolved by keeping both horizon and ranking commands
   together with prior materialize/driver/run dispatch and CAV/entries flags.
   Paired-delta, regret and magnitude/traffic helpers retained.
   Post-merge tests: **36 passed**. Removed three unused symbols caught by Ruff.
6. **`wms/ctm` — `f979709`.**
   Clean merge of standalone CTM, optional calibration/evaluation modules and
   tests. No default rollout/controller dependency introduced.
   Post-merge tests: **18 passed**.
7. **`wms/diag120` — `4efcbab`.**
   Confirmed the remaining branch delta was entirely under `_out/`; clean
   merge of scripts and their frozen analysis artifacts. Syntax check:
   `python -m compileall -q _out/diag120` passed. Scripts were not executed and
   their scientific outcomes were not re-evaluated. Final suites ran afterwards.

All seven branch tips are ancestors of the integrated branch (checked with
`git branch --merged HEAD --list` for the seven requested names).

## Integration repairs and parity evidence

- Hybrid uses one fixed inferred driver embedding even when the lane head is
  off, including detached/chunked training continuation.
- Entry padding extends all action fields; original timeline columns are
  preserved. Entrant CAV identity and vehicle length survive BPTT boundaries.
- Timeline decoding honors explicit compact/legacy schemas and vocabulary;
  ambiguous unlabeled numeric IDs fail closed. Direct actions require both
  command fields. The existing compact schedule fixture now declares its schema.
- Live dispatch receives the action timeline; sorted t0 identity alignment and
  entrant defaults are preserved. PI invalidates on handback and resets on
  re-entry; the same damping instance resumes. Direct commands execute at their
  declared control boundary.
- Held longitudinal commands apply on both 0.5 s physics steps. Mixed initial
  and entrant CAVs call the dispatcher once per physics tick. Delayed entrants
  use the episode physics clock rather than a callback-local arrival clock.
- Switch oversampling also inspects lazy indexed windows one at a time and
  retains only boolean membership. It no longer silently ignores the flag.
- Legacy pre-switch structured and black-box trajectory checkpoints reconstruct
  their original architecture/equations with new conditioning off. Exact output
  parity is tested synthetically. Hybrid/lane checkpoints still require their
  encoder weights; receipt caps/projection and entries-v2 replay are covered.
- Constant live damping timelines are **byte-identical** to fixed-controller
  rollouts for residual, driver_params and hybrid, with/without history lane
  heads. Nonzero-command half-step holding and PI/damping lifecycle are checked.
- The path-confinement regression now works with workspace-confined pytest
  temporary directories: it checks a real internal link, then models external
  realpath resolution without accessing an external fixture.

New focused regression files:
`wms/tests/test_main_resume_integration.py`,
`tests/test_wms_eval_main_compat.py`, and
`tests/test_wms_eval_main_switch.py` (the first path is under
`output/wd01/wm_n1/`). Existing integration/action tests remain enabled.

## Final verification (exact)

**438 passed, 6 skipped, 0 failed; 444 collected**, in **199.09 s**.
Peak process working set: **820224000 bytes** (approximately 0.82 GB), below the
6 GB per-process rule. Pack-backed production readers retain `mmap_mode='r'`;
the new lazy membership scan does not retain loaded windows. No real packs were
loaded during this task.

JUnit receipt: [`MAIN_MERGE_TESTS_RELEASE.xml`](MAIN_MERGE_TESTS_RELEASE.xml).

Exact Windows invocation (explicit glob expansion is required in PowerShell):

```powershell
$env:TMP="E:/rd-wt/ws-main/_out"
$env:TEMP=$env:TMP
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:CUDA_VISIBLE_DEVICES=""
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -c "import glob,pytest,psutil; code=pytest.main(['output/wd01/wm_n1/wms/tests','output/wd01/wm_n1/hdv/tests',*sorted(glob.glob('tests/test_wms_eval_*.py')),'-q','--basetemp=_out/pytest_main_full_release','--junitxml=_out/MAIN_MERGE_TESTS_RELEASE.xml']); print('Peak process working set bytes:',psutil.Process().memory_info().peak_wset); raise SystemExit(code)"
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m ruff check output/wd01/wm_n1/wms output/wd01/wm_n1/wms_eval tests/test_wms_eval_main_compat.py tests/test_wms_eval_main_switch.py tests/test_wms_eval_paths.py
```

Ruff: **All checks passed**. `git diff --check`: passed at close-out.

Other post-merge commands:

```powershell
# evalshift acceptance: 364 passed, 6 skipped
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest output/wd01/wm_n1/wms/tests tests/test_wms_eval_main_switch.py tests/test_wms_eval_main_compat.py tests/test_wms_eval_wms_plugin.py tests/test_wms_eval_materialize.py tests/test_wms_eval_hetdata_shift.py tests/test_wms_eval_c1sets.py tests/test_wms_eval_horizon_report.py -q --basetemp=_out/pytest_evalshift_acceptance_fixed
# ranking: 36 passed
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest tests/test_wms_eval_ranking.py tests/test_wms_eval_metrics.py tests/test_wms_eval_horizon_report.py tests/test_wms_eval_wms_plugin.py -q --basetemp=_out/pytest_ranking_merge
# CTM: 18 passed
& "C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe" -m pytest tests/test_wms_eval_ctm.py tests/test_wms_eval_metrics.py tests/test_wms_eval_ranking.py -q --basetemp=_out/pytest_ctm_merge
```

### Failed/interrupted checks preserved

- Literal `tests/test_wms_eval_*.py` passed to `python -m pytest` on Windows:
  collection error, **no tests ran**. Corrected by explicit Python glob expansion.
- First expanded full-suite attempt hit the tool's **120 s timeout**; incomplete
  counts are not represented as a completed suite. Later runs used 600 s timeout.
- Initial evalshift acceptance: **1 failed, 360 passed, 3 skipped** (new mixed-CAV
  fixture omitted `a_mps2`). Fixture corrected; wall standalone coverage enabled.
- First completed full suite: **1 failed, 435 passed, 6 skipped**, in 173.50 s;
  peak 817016832 bytes. Failure was the existing confinement fixture's assumption
  that `tmp_path` was external. Receipt: `MAIN_MERGE_TESTS.xml`.
- After confinement/clock repair: **437 passed, 6 skipped**, in 195.69 s; peak
  808128512 bytes. Receipt: `MAIN_MERGE_TESTS_FINAL.xml`.
- The final one-window-at-a-time lazy switch-sampler regression independently
  passed (**1 passed**) before the 438-pass release run.
- Ruff initially found duplicate/unused `os` in the resumed plugin and three
  unused symbols in incoming ranking code; repaired. Initial whitespace check
  found an extra EOF blank line in incoming black-box code; repaired.

## Remaining evidence limits

No requested code feature was omitted and no branch was skipped. Six optional
checks remain skipped because their workspace fixtures are absent:

- Two torch/numpy real-state comparisons need TRAIN chunk-4 runs
  `w8676m2700p20t360__base` and `w8677m2700p20t360__base` under
  `_ext/campaign_chunk4/`.
- Four source-parity cases need `wall_group.py` and `wall_group_v22.py` staged
  together in `_in/traffic_shepherd/`. The four standalone wall cases pass.

Exact coordinator commands and staging contract are in
[`MAIN_READY.md`](MAIN_READY.md). No CAL, HOLDOUT, sealed, chunk-5, J4B or INC1
data was opened. Inherited branch reports are preserved, not independently
endorsed by this engineering close-out. Research ledger entries and scientific
summaries are unchanged: this record contains implementation/test verification,
not a new experiment or a change in scientific conclusions.

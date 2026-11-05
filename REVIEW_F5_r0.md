VERDICT: PASS

## Numbered blockers

1. None attributable to F5. The reporting change satisfies `E:/agent-work/_sol/tracks_spec.md:67`. The additional compatibility-suite failure below is inherited from the recorded launch base, occurs while constructing the test's expected baseline, and does not demonstrate an F5 regression. This verdict approves F5 reporting; it does not assert that the entire integrated branch's test suite is green.

## Test counts

- F5 readiness selection, independently executed: **120 passed, 6 deselected, 0 failed**, 79.98 s. Includes all **103 monitor cases**, 15 selected policy-synergy cases, and 2 launch-integration cases.
- Additional default-off compatibility suite: **13 passed, 1 failed**, 35.51 s. Failure and follow-up are detailed below.
- Combined pytest executions: **133 passed, 1 failed, 6 deselected**. Each execution emitted one pytest configuration warning because disabling the cache plugin leaves the repository's `cache_dir` option unrecognized.
- Independent in-memory comparisons against committed baseline code: **10 passed**: six numerical/safety/verdict panels, two byte-equivalent serialized default-off monitor receipts, and two byte-equivalent serialized default-off runtime configurations. These are supplementary checks, not included in the pytest counts.
- `git diff --check`: passed.

### Commands and environment

Both pytest runs used `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tools/coopmwm/test_track2_monitor_cpu.py tools/coopmwm/test_policy_synergy_cpu.py tools/coopmwm/test_policy_synergy_isolated_cpu.py tools/coopmwm/test_launch_integration_cpu.py::test_production_config_keeps_canonical_registration_and_prior_budget tools/coopmwm/test_launch_integration_cpu.py::test_production_entrypoint_imports_do_not_construct_or_dispatch_workers -k "not complete_branch_set_precharged and not isolated_complete_precharge" --basetemp=E:/agent-work/c4-fix-reporting/_out/f5_astra_r0_cpu -p no:cacheprovider

C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tools/coopmwm/test_t2c_compat_cpu.py --basetemp=E:/agent-work/c4-fix-reporting/_out/f5_astra_r0_compat -p no:cacheprovider
```

The six deselections match `F5_READY.json:80`: two `complete_branch_set_precharged` cases and four `isolated_complete_precharge` cases. They require Linux `/proc` through `tools/coopmwm/replay_start.py:35-36`. Their previous Windows failures are documented at `P3_READY.json:112-118`. I inspected those tests and confirmed that their source, `policy_synergy.py`, and `replay_start.py` are unchanged against `c4/track-base`. I did not count these excluded cases as passing or independently reproduce their known failures.

### Inherited compatibility-test failure and follow-up

`tools/coopmwm/test_t2c_compat_cpu.py:23-29` sets component entries in `sys.modules` to `None`, including `practice_weighting`, then calls the in-memory track-base implementation to construct `expected`. That baseline imports the current `v2v_runner`, whose unconditional import at `tools/coopmwm/v2v_runner.py:120` raises `ModuleNotFoundError`. The test fails before comparing F5/current output to its expected output.

Both files are unchanged against F5's recorded launch base `8bf7f53395c7a53a3e8d847c5a7f3229b79ab102`. Follow-up for the compatibility/integration owner: resolve the inherited optional-dependency import contract in `v2v_runner.py:120` and related practice-option calls, and make the baseline test's dependency isolation explicit; rerun this test without simply suppressing its assertion. The independent checks with installed dependencies confirmed identical serialized absent/false Track-2 runtime configurations against the track-base implementation. This is an existing integration issue, not an F5-required fix.

## Review scope and evidence

- Reviewer: independent verifier **astra**, 2026-11-05.
- Worktree/branch: `E:/agent-work/c4-fix-reporting`, `c4/fix-reporting`.
- Requested comparison base: `c4/track-base`, `13f184b5a7e12be04aebf246ef8e4dbb81555545`.
- Worktree HEAD and F5 readiness baseline: `8bf7f53395c7a53a3e8d847c5a7f3229b79ab102`.
- Read the shared specification and `F5_READY.json`, inspected the requested base-diff inventory and F5 monitor/test diff, and examined the metric, registration, runtime-hook and compatibility dependencies. The requested comparison includes 144 tracked files from integrated launch work. Against the fixed launch baseline, F5 changes only `tools/coopmwm/track2_monitor.py` and `tools/coopmwm/test_track2_monitor_cpu.py`, plus its untracked readiness manifest. This is a task-F5 review, not a requalification of all inherited launch work.

## Correctness against F5

- **Both quantities are explicit.** `track2_monitor.py:78-87` names total benefit `delta_all_new_veh_s = J_reference - J_all` and group-over-best-single synergy `policy_cooperation_gain_veh_s = (J_reference - J_all) - max_i(J_reference - J_solo_i)`. Units, direction and endpoint flags are explicit. The best-solo benefit is not floored at zero, matching the unchanged algebra in `policy_synergy.py:137-145`.
- **Measured reporting is real.** `track2_monitor.py:593-602` obtains both values from the existing branch-grade metric function and averages states within each world. `track2_monitor.py:513-546` then averages worlds equally and bootstraps paired world replicates, not pooled states or branches. Both metrics have separate comparisons and CIs; the old top-level comparison keys are explicitly marked as synergy aliases at lines 547-555.
- **Registered u24 definition is preserved.** `rerun_cpus.py:116-154` registers synergy, N=5, 10,000 percentile draws, 95% level, fixed seed and the no-safety-regression requirement. `track2_monitor.py:550-557` retains that decision, and safety aggregation at lines 597-599 is unchanged. Total benefit cannot substitute for synergy. Exact comparisons, safety dictionaries, safety decisions and verdicts matched the original launch-base monitor on six varying-data panels spanning u6/u12/u24 and safe/unsafe cases.
- **Probes, dry runs and legacy evidence are distinguished.** `track2_monitor.py:703-710` gives probes descriptive metric reports without inferential CIs. Dry reports contain definitions but no measured means. Status text at lines 401-418 shows both quantities and the unchanged criterion. Legacy total benefit is derived from recorded state entries at lines 501-510; missing evidence is not zero-imputed. Legacy receipt bytes remain unchanged, covered at `test_track2_monitor_cpu.py:698-725`.
- **Tests have substantive oracles.** The fake runner replaces physics/accounting, while real CPU tensors, checkpoint hashing, branch specifications, masks, algebra, persistence and bootstrap execute. Tests at `test_track2_monitor_cpu.py:631-695` cover positive total benefit with zero/negative synergy, negative total benefit with positive synergy, B/R-identical outcomes, and unequal state counts. Existing probe tests assert current-A versus common-base checkpoint routing. Synthetic outcomes establish reporting correctness, not actual learned performance.

## Rule compliance

- **Observables/no hidden truth:** F5 consumes existing diagnostic grades and safety counters; it adds no policy/planner input or truth reader. Metric additions are reporting-only. No restricted datasets or truth files were opened during this review.
- **Full look-ahead:** the expansion at `track2_monitor.py:452-460` retains the registered full window and refuses changed/truncated windows. F5 does not alter expansion or dispatch. Tests assert the original window/end time and full quoted branch cost.
- **B/R fairness:** one shared extraction/aggregation/bootstrap path serves both arms; pairing validates world/state/window identity at `track2_monitor.py:528-545`. Identical synthetic outcomes produce identical means and exact-zero B-R contrasts for both metrics.
- **Default-off byte identity:** F5 changes reporting content intentionally but does not alter policy actions, rewards, registration, runtime hooks or rollout configuration. `rerun_cpus.py`, `track2_backend.py`, `track2_reward.py`, `launch_training.py`, and `launch_production.py` have no diff against the fixed launch baseline. Supplementary baseline comparisons establish exact serialized default-off monitor-row and runtime-config equality. The inherited import-isolation failure is disclosed above.
- **No new behavioral thresholds:** F5 adds metric definitions and report metadata, not intervention thresholds. Bootstrap settings, decision boundary, safety checks and diagnostic schedules are unchanged.
- **Execution constraints:** CPU only; no SSH, SUMO or GPU execution; no commits or source edits. Test-generated scratch artifacts are confined to the worktree `_out` paths above. This review file is the only reviewer-authored deliverable.

## Fresh-session resume verification — 2026-11-06

- The initial status was clean on `c4/fix-reporting`. Current HEAD is `254c3ba06f6263eab9e27c739f8d1e7122383c22`, which preserves the F5 implementation, tests, readiness manifest, this completed review, and `F5_ACCEPTED.json`. The HEAD listed in the original review above describes its pre-commit review state.
- Read the git status/log, `F5_READY.json`, existing review and acceptance receipt, and shared specification. Inspected the requested `c4/track-base` diff inventory and the complete F5 monitor/test patch against the recorded launch baseline. The current base-diff inventory has 147 files; the preserving commit has exactly the five files listed above. No later source changes or uncommitted implementation changes require requalification.
- Preserve **VERDICT: PASS** for F5 and the original numbered-blocker assessment, including the disclosed inherited compatibility failure. The patch retains the synergy endpoint and safety decision, shared B/R aggregation, full-window execution, and reporting-only scope described above.
- Per the resume instruction not to redo finished work, the completed pytest executions and supplementary checks above were **not rerun**. Their counts are prior-session evidence, not fresh-session executions. Fresh-session `git diff --check c4/track-base --` passed.
- Only this resume note was written. No commits, jobs, dispatches, SSH, SUMO, GPU execution, or restricted-path/truth-file reads occurred in this resumed session.

## Resume verification (2026-11-06)

- The fresh session found this completed review and `F5_ACCEPTED.json` already committed together with the F5 implementation/tests and readiness receipt in `254c3ba06f6263eab9e27c739f8d1e7122383c22`. Initial `git status --short --branch` was clean on `c4/fix-reporting`; that commit is the current HEAD. The earlier HEAD/untracked descriptions above describe the original review session, before preservation in this commit.
- Read git status/history, the existing review and acceptance/readiness receipts, and the shared specification. Rechecked the permitted-path diff inventory against `c4/track-base` and the F5 implementation/test diff against the recorded launch baseline `8bf7f53395c7a53a3e8d847c5a7f3229b79ab102`.
- Preserved the completed verdict and test evidence under the explicit resume instruction not to redo finished work. All test counts above are the original session's recorded executions, not fresh executions in this session; additional pytest executions on resume: **0**. The inherited compatibility failure remains disclosed and is not represented as passing.
- This resume adds only this provenance note. No implementation changes, commits, dispatches, SSH, SUMO, GPU execution, or restricted-file reads were performed.

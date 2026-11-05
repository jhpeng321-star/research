VERDICT: BLOCK

1. **Active multi-CAV moves abort production episodes when a partner leaves the observable neighbourhood between decisions.** Locations: `tools/coopmwm/rerun_tracks_backend.py:182-202`, `:208-211`. The hook removes departed *actors* from `self.actors`, but does not invalidate surviving actors' active moves when those moves contain a departed or newly out-of-range partner. Until the next registered decision tick, it computes offsets for the old move and then requires every offset key to belong to the actor's *current* observable CAV set. Ordinary traffic turnover therefore raises `ValueError('REFUSED: HDV/nonobservable production command')`, rather than continuing safely with the remaining observable vehicles. The same hook is used in training and isolated/final evaluation. An admitted, charged episode can fail merely because one previously valid move member exits; this is not a malformed adapter output or a safety violation.

   **Independent CPU reproduction:** install the existing test-only Move API, choose a two-member `gap_open` move at t=300, use a zero residual on currently visible CAVs, and invoke the actual production `EpisodeHook.apply` wrapper. With `c1` at 0 m and `c2` at 20 m, both base commands 0.2 and move offsets 0.2, t=300 succeeds with both requested accelerations 0.4. At t=301, keep `c1` and remove `c2` from the sensor/command rows. The call fails at `rerun_tracks_backend.py:211` with the exception above. This was an in-memory local check: no files, physics, dispatch, or implementation edits. `moves.offsets(move, elapsed)` receives no updated observation, so it cannot itself discover that a member departed.

   **Required fix:** handle changing membership explicitly before executing a retained move. Safely retire/cancel the affected commitment or its unavailable members according to the registered move semantics, retain base control where appropriate, and preserve observable-only commands, single ownership, and the registered decision cadence without inventing a new behavioural threshold. Continue to reject genuinely invalid fresh proposals/residuals. Add production-hook CPU regressions with changing sensor frames: partner leaves sensing range while still on-road; partner exits; actor exits/reappears; overlapping groups; and isolated all/solo masks. Assert that valid traffic turnover does not abort the complete episode, no unavailable member is commanded, and reward/learning receipts remain consistent. Current coverage keeps membership static: `tools/coopmwm/test_t2c_backend_cpu.py:73-85` reuses one sensor frame through all 900 controlled ticks, and `tools/coopmwm/test_t2c_multicav_cpu.py:30-39` exercises only one tick per case.

## Test counts and execution evidence

Independent verifier: **astra**, 2026-11-05. Reviewed branch `c4/rerun-tracks`, HEAD **`70911f7aad20e5b4674d7dc7efb162064260737e`**, against `c4/track-base` **`13f184b5a7e12be04aebf246ef8e4dbb81555545`**. Read status/log first, then readiness/recovery/previous-review receipts and the complete `E:/agent-work/_sol/tracks_spec.md` through MAIN 02:30 (line 59), followed by the scoped diff and relevant implementations/tests. The owner-shortcut acceptance receipt was read; it does not replace the independent correctness check requested in this session.

- **Focused T2C suite: 161 passed, 1 skipped, 0 pytest failures; 162 distinct tests.** All 11 focused files were executed, with the runtime file partitioned into disjoint selections.
- **Applicable regressions: 55 passed, 0 failed, 5 deselected; 60 distinct tests including exclusions.**
- **Combined: 216 passed, 1 skipped, 5 deselected; 222 distinct tests including exclusions.** These are fresh independent execution counts, not copied readiness claims. Intermediate runtime-selection deselections are not added again to the distinct totals.
- **Additional correctness check:** the in-memory production-hook reproduction above failed as described, outside pytest; it is not included in the passing-test count. Passing existing suites therefore do not clear the blocker.

Every invocation used:

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q --tb=short -p no:cacheprovider
```

Environment: `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. The following are the exact arguments appended to that command; invocations were short and sequential, and each completed with exit code 0:

1. **43 passed, 1 skipped**, 54.20 s:
   ```text
   tools/coopmwm/test_rerun_cpus_cpu.py --basetemp=E:/Temp/opencode/astra-t2c-resume-prep-1105
   ```
2. **44 passed**, 30.29 s:
   ```text
   tools/coopmwm/test_t2c_bank_cpu.py tools/coopmwm/test_t2c_monitor_cpu.py tools/coopmwm/test_t2c_compat_cpu.py --basetemp=E:/Temp/opencode/astra-t2c-resume-contracts-1105
   ```
3. **32 passed**, 56.59 s:
   ```text
   tools/coopmwm/test_t2c_staging_cpu.py tools/coopmwm/test_t2c_augmentation_cpu.py tools/coopmwm/test_t2c_multicav_cpu.py tools/coopmwm/test_t2c_redesign_cpu.py --basetemp=E:/Temp/opencode/astra-t2c-resume-new-1105
   ```
4. **10 passed**, 38.22 s:
   ```text
   tools/coopmwm/test_t2c_backend_cpu.py --basetemp=E:/Temp/opencode/astra-t2c-resume-backend-1105
   ```
5. **17 passed, 12 deselected**, 13.74 s:
   ```text
   tools/coopmwm/test_t2c_runtime_cpu.py -k "not smoke and not executes_complete and not shared_final and not full_run_reaches" --basetemp=E:/Temp/opencode/astra-t2c-resume-runtime1-1105
   ```
6. **10 passed, 19 deselected**, 21.58 s:
   ```text
   tools/coopmwm/test_t2c_runtime_cpu.py -k "shared_final or full_run_reaches or executes_complete" --basetemp=E:/Temp/opencode/astra-t2c-resume-runtime2-1105
   ```
7. **5 passed, 27 deselected**, 41.35 s:
   ```text
   tools/coopmwm/test_t2c_lifecycle_cpu.py tools/coopmwm/test_t2c_runtime_cpu.py -k "lifecycle or smoke_executes or smoke_and_one_command" --basetemp=E:/Temp/opencode/astra-t2c-resume-lifecycle-1105
   ```
8. **48 passed, 1 deselected**, 61.19 s:
   ```text
   tools/coopmwm/test_launch_final_cpu.py tools/coopmwm/test_effimit_cpu.py tools/coopmwm/test_rerun_clean_cpu.py tools/coopmwm/test_v2v_runner_cpu.py tools/coopmwm/test_v2v_runtime_cpu.py -k "not test_gate_seals_commit_and_runtime_and_the_launcher_reads_it" --basetemp=E:/Temp/opencode/astra-t2c-resume-regression1-1105
   ```
9. **7 passed, 4 deselected**, 4.32 s:
   ```text
   tools/coopmwm/test_policy_synergy_isolated_cpu.py -k "not test_isolated_complete_precharge_adopts_old_reference_not_all_new_lesson" --basetemp=E:/Temp/opencode/astra-t2c-resume-regression2-1105
   ```

The skip is the documented Windows file-symlink privilege case (`test_rerun_cpus_cpu.py:356`); hard-link and junction checks ran. The single launch-test exclusion creates a Git commit in its fixture (`test_launch_final_cpu.py:296-297`), contrary to this review's no-commits rule. The four isolated-test exclusions are the inherited Linux `/proc`-dependent parameterizations documented by the previous independent review; they were not rerun or counted as passes. Each invocation emitted the inherited unknown-`cache_dir` warning because pytest's cache provider was disabled.

## Reassessment of the previous five blockers

- **Final redesign:** addressed at the T2C boundary. Dense teaching and WM credit are pinned OFF, factories are not constructed, verification allocations are zero, and layer learning receives realised observable-neighbourhood rewards. The native A PPO path remains separate. Tests demonstrate changed realised outcomes change layer updates, frozen-base identity, local HDV participation, and refusal of remote reward IDs. See `rerun_cpus.py:59-94`, `rerun_tracks_reward.py:11-37`, `rerun_tracks_runtime.py:534-549`, and `test_t2c_redesign_cpu.py`.
- **Augmentation admission:** addressed in the reviewed native integration. Verification invokes the actual both-seed all8 confirmation machinery, including later-harm/cooperation checks, through coordinator-owned full-wave reservations. Only admitted cooperative positive own-seed executions reach learning/practice; materialisation uses modified execution tapes and weights. Tests exercise negative/partial admission, missing/copied/changed evidence, exhausted allowance, and full windows. See `rerun_tracks_augment.py:13-74`, `:77-150`, `rerun_tracks_bank.py:15-46`, and `rerun_tracks_runtime.py:417-475`.
- **Per-CAV coverage:** static disconnected-group, row-order, ownership and isolated all/solo coverage now pass. The new blocker concerns the unhandled membership changes between those decisions; the earlier single-persistent-ego implementation has been replaced.
- **Smoke-to-run lifecycle:** addressed. Tests complete smoke and then all three full update-1 arms from fresh common-u0 models, preserving smoke outputs and once-only augmentation, with disjoint job IDs and cumulative charges. Duplicate/interrupted attempts and mismatched receipt bindings are refused. See `rerun_tracks_runtime.py:175-221` and `test_t2c_lifecycle_cpu.py`.
- **Default-off preparation:** addressed for the reviewed public quote/budget/wall-clock/CLI, configuration, runtime delegation, root/ledger, approval, action/RNG and checkpoint boundaries. Committed-base comparisons pass. See `rerun_cpus.py:174-182`, `:233-269`, `rerun_cpus_compat.py`, and `test_t2c_compat_cpu.py`.

## Rule compliance and test quality

- The production policy-facing projection whitelists local observable fields, derives each actor's neighbourhood from its sensing range, and forwards only causal t-1 partner reports (`rerun_tracks_backend.py:128-152`). Reward inputs are locally projected before P2 and cannot supply arbitrary global/remote reward IDs. No direct hidden-truth-file read was found in the changed policy-facing code. The membership blocker is an availability/correctness failure of this otherwise conservative observable-command check.
- No look-ahead reduction was found in the reviewed wiring. Episodes remain 1,200 seconds; verification rejects shortened windows; native augmentation can extend, not shorten, the source window; isolated state/window identities and shared-final full horizons are checked. Actual sibling planner behaviour remains a deployment integration dependency.
- B/R use the same enabled component configuration, reward formula/weight, augmentation procedure and admission limits, practice fraction/repeat cap, and paired diagnostic worlds, with their own banks. Disabled teacher/credit paths cannot introduce asymmetric launch-time labels. The quote is **4,310,220 sim-s including smoke**, targets **cpuV**, and asserts the 5,000,000 ceiling; augmentation is capped at 143,280, below 200,000. Replicated checks are B/R u6/u12/u24, N=5; quick probes remain u2/u4 on three states.
- No additional unregistered hand-set driving threshold was identified in the reviewed T2C changes. New experiment/reward/monitor values are explicitly proposed and approval-bound; those registrations are not proof that live owner approval has already been obtained. Existing command bounds and the native safety writer remain in use.
- Tests are substantive: native V2V controller/PPO execution, real autograd and effective/Tobit bank losses, state-conditioned targets, negative evidence checks, native all8 confirmation against CPU tapes, full diagnostic-grid orchestration, real reader-driven staging closure, and lifecycle/approval refusal cases. They use test-only sibling adapters and replace the SUMO boundary. In particular, their static traffic frames do not establish the dynamic-membership property in blocker 1. Some routing tests deliberately replace repeated update work; they are not evidence of 24 live learning updates.
- The backend contains native episode/branch/launch3 wiring, rather than only a synthetic coordinator. Sibling implementations and the T1D adapter remain deployment dependencies as documented in `RERUN_TRACKS_READY.md:154-159`. No synthetic smoke result or self-declared release flag is treated as live qualification. The blocker above is within T2C's own production hook and is reproducible without those dependencies or SUMO.
- Initial tracked worktree state was clean; all existing untracked receipts/test artifacts were preserved. An already-running integration pytest process was inspected and left untouched; its selections did not overlap this review's suites. No runtime/dispatch jobs were launched or duplicated. Review execution was local and CPU-only: no SUMO, GPU, SSH, commits, protected-path opens, or truth-data-file reads. Guard tests use forbidden path strings to assert refusal before opening; no protected file was opened. Test output was confined to the explicit temporary roots. The sole worktree edit is this requested review report. `git diff --check c4/track-base` passed before report replacement.

## Fresh-session recovery confirmation

The subsequent astra resume on 2026-11-05 found this completed independent review already present as an uncommitted modification at the same HEAD, `70911f7aad20e5b4674d7dc7efb162064260737e`. It preserved that work. Status/log were read first, followed by the readiness, acceptance, fix and review receipts and the complete shared specification. The base-diff inventory and production-hook diff were inspected again. A scoped `git diff --exit-code 70911f7aad20e5b4674d7dc7efb162064260737e -- tools/coopmwm RERUN_TRACKS_READY.md T2C_READY.json T2C_CPU_SMOKE.json T2C_FIX_CURRENT_R1.md` returned exit 0: implementation, tests and those receipts are unchanged from the reviewed revision. The retained-move validation at `rerun_tracks_backend.py:208-211` still establishes blocker 1.

The **216 passed, 1 skipped and 5 deselected** above are preserved completed independent-verifier execution results from the preceding session, not newly executed tests in this recovery and not copied readiness totals. In accordance with the instruction not to redo finished work, no completed suite or reproduction was repeated; this recovery adds **0 test executions**. The local Python-process inventory showed no active pytest invocation, and existing processes were left untouched. `git diff --check c4/track-base` passed. Only this review file was updated; no commit, dispatch, SUMO, SSH, GPU operation, protected-path open or truth-file read was performed. Verdict remains **BLOCK**.

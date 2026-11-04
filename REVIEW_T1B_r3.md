VERDICT: PASS

## Numbered blockers

None. No required fixes identified in this revision's T1B CPU build/contract scope.

## Test counts

1. T1B focused suites: **114 passed, 0 failed, 0 skipped**, 364.86 s.
2. Variety-collection compatibility suite: **48 passed, 0 failed, 0 skipped**, 24.79 s.
3. Runtime/ownership/dynamic-policy/intent/V2V-observation regression selection: **26 passed, 0 failed, 0 skipped; 9 subtests passed**, 21.97 s.
4. **Combined: 188 passed, plus 9 passing subtests.** All counts are from this independent review's completed runs, without double-counting.
5. `git diff --check c4/track-base --` for the seven changed implementation/test files: passed.

Each pytest invocation emitted one `Unknown config option: cache_dir` warning because the cache provider was disabled for read-only review. No implementation or test files were edited.

## Review scope

- **Verifier:** astra.
- **Worktree / branch:** `E:/agent-work/c4-planner-online`, `c4/planner-online`.
- **Reviewed HEAD:** `1f15a9ef0d249bebbbafd43efccd7c3f677778e7`.
- **Base:** `c4/track-base`, registered as `13f184b5a7e12be04aebf246ef8e4dbb81555545`.
- Read the shared `E:/agent-work/_sol/tracks_spec.md`, current `T1B_READY.json`, r2 review and fix receipt, changed implementation/test sources versus base, and relevant permitted dependency sources. Initial worktree status was clean.
- This PASS establishes the reviewed component's CPU build/contract readiness. T1A `moves.py` and T1C `planner_calibration.py` integration remains unverified, as disclosed in the readiness record; the shared spec permits test-local stubs while those dependencies are absent. No trained-WM or physical-plant performance claim is established by these synthetic tests.

## Correctness and rule checks

1. **Previous r2 blocker resolved: shared joint-commitment lifetime.** `tools/coopmwm/planner_online.py:159-170` preserves ego/member/profile metadata for both the candidate and shared backgrounds. `tools/coopmwm/planner_online.py:251-264` refuses incomplete background visibility instead of exposing unseen members or flattening a partial joint move. Live cancellation at `tools/coopmwm/planner_online.py:284-287` and forecast cancellation at `tools/coopmwm/planner_commitments.py:7-15,28-51` use the same observable contact predicate. Cancellation is per action and per commitment, persists through recontact, is checked on whole-second decisions, and resets at a new future's tick zero. `tools/coopmwm/plan_wm.py:103-115` applies the resulting lifetime mask without stopping the base-policy rollout. The 16 numeric cases at `tools/coopmwm/test_planner_policy_cpu.py:307-397` pass: candidate/background, range/member/ego/CAV-membership loss, immediate/delayed onset, staggered branch-local cancellation, unaffected branches, recontact, half-step behavior, expiry, 600-second continuation, and future reset. They capture actual scoring actions from the wrapped controller and exercise the production decoder; they are substantive regressions for the previous defect.

2. **Observable-only inputs / no hidden truth.** `tools/coopmwm/planner_online.py:45-107` validates sensor fields, ego-local geometry/history, CAV membership, finite command values and disjoint observable commitments. Runtime construction at `tools/coopmwm/planner_online.py:245-249,298-320` restricts records to the acting ego's view. `tools/coopmwm/planner_policy.py:34-106` filters radio state and validates timestamped publications, local sampling state and current residuals. Forecast continuation derives features from observable/predicted geometry and policy/radio state. No new hidden-driver-type input, truth-file reader, or unfiltered global observation record was found in the changed implementation. Negative privacy/causality tests pass.

3. **Full look-ahead and paired scoring.** `tools/coopmwm/planner_online.py:27-38` requires the registered 600-second horizon and at least two WM samples. `tools/coopmwm/planner_online.py:173-191` verifies complete finite paired results, computes strictly-positive paired-gain frequency and expected gain, and retains the full horizon regardless of move duration. `tools/coopmwm/planner_online.py:215-219` invokes the existing evaluator with `trim=False`, `final_chain=True` and required policy-radio continuation. The full-horizon adapter/chunk/sample tests at `tools/coopmwm/test_planner_policy_cpu.py:253-304` pass. Their external futures/dynamics producer is synthetic; the adapter, evaluator, decoder and continuation are real.

4. **Calibration and registered gate.** Calibration is mandatory with no raw-probability fallback (`tools/coopmwm/planner_online.py:147-151,184-193`). The confidence threshold is fixed at 0.8 (`tools/coopmwm/planner_online.py:35-38`). Selection requires positive expected gain and calibrated probability at least 0.8, then chooses greatest expected gain with stable ties. Tests exercise both a calibration-closed unanimous raw vote and calibration-open lower raw probability, exact threshold boundaries, invalid probabilities, missing dependencies and malformed rollout results. Runtime scoring/dependency errors fail closed to base-policy operation (`tools/coopmwm/planner_online.py:334-338`).

5. **Base-policy continuation and sampling.** `tools/coopmwm/planner_policy.py:120-199` reuses the current live sample at tick zero and advances branch-local actor/noise/RNG, causal radio frames, issued-command feedback and intent publications thereafter. Plain, V2V and intent deterministic/stochastic numeric parity tests pass, including the prior plain-policy seed-7 reproduction. Actor loading preserves global torch RNG in the online path (`tools/coopmwm/plan_wm.py:142-149`).

6. **Single writer, CAV-only commands and feasibility.** The opt-in hook at `tools/coopmwm/v2v_runtime.py:89-95,133-143` refuses competing configured guide/replay/branch producers and modifies only pre-safety requests. `tools/coopmwm/planner_online.py:233-237,342-357` enforces CAV bounds, applies offsets to base requests, and preserves the sampled `own_action`. No additional TraCI writer is introduced. Actual `CommandOwnership` execution behind a fake vehicle API passes the CAV-only, acceleration-cap and speedMode-27 assertions; ownership-chain regressions also pass.

7. **Default-off byte identity and legacy compatibility.** Omitted/`None` planner configuration leaves the runtime hook inactive. `tools/coopmwm/test_planner_online_cpu.py:381-401` executes the pinned track-base controller source and compares serialized event bytes and policy RNG state against this revision for both deterministic and stochastic policies. It passes, as do omitted-versus-explicit-None checks, legacy profile-decoding assertions, dynamic-policy transfer-table regressions and the compatibility suite. New commitment masking is activated only for metadata-bearing offset commitments.

8. **B/R fairness.** The new scorer and runtime contain no B/R-specific branch, bank selection, training allocation or arm-specific budget. Shared background commitments are included in both candidate and base scoring actions. No new differential B/R treatment was found. The compatibility suite's existing B/R contracts pass; end-to-end dense-teacher/evaluation fairness remains a caller-integration responsibility.

9. **No new hand-set behavioral thresholds found.** Confidence 0.8, default cadence 60 seconds, full horizon 600 seconds, group size at most four, acceleration cap 1.2 and inherited sensing range match the shared contract. Candidate/sample defaults 32/8 match the existing `mppi_search.REGISTERED` values. Move parameter generation is delegated to the specified move-library API; no production fallback sampler is introduced.

10. **Tests are real component tests.** The new suites include numerical live/imagined comparisons, adversarial rejection cases, source-based default-off identity checks, actual controller/ownership execution, and full-horizon decoder runs. Cross-track dependencies, WM dynamics, safety/external executor shells and vehicle APIs are synthetic where stated. Passing tests support the scoped verdict rather than demonstrating trained-model calibration quality or SUMO outcomes.

## Commands and review restrictions

The following commands used the requested interpreter, with `CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, and `TEMP`/`TMP` set to `E:/Temp/opencode`. Test temporary directories were outside the worktree.

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q -p no:cacheprovider --basetemp=E:/Temp/opencode/t1b-r3-focus tools/coopmwm/test_planner_policy_cpu.py tools/coopmwm/test_planner_online_cpu.py

C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q -p no:cacheprovider --basetemp=E:/Temp/opencode/t1b-r3-compat tools/coopmwm/test_varcollect_cpu.py

C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q -p no:cacheprovider --basetemp=E:/Temp/opencode/t1b-r3-regression tools/coopmwm/test_v2v_runtime_cpu.py tools/coopmwm/test_command_chain_cpu.py tools/coopmwm/test_mppi_runtime_cpu.py::LongProfileTests::test_transfer_table_exact_for_long_profiles_and_dynamic_policy tools/coopmwm/test_mppi_runtime_cpu.py::LongProfileTests::test_imagined_nonzero_actor_matches_observable_plant_actor tools/coopmwm/test_v2v_intent_cpu.py::test_partner_intent_has_one_step_latency tools/coopmwm/test_v2v_intent_cpu.py::test_runtime_reset_and_despawn tools/coopmwm/test_v2v_intent_cpu.py::test_actions_identical_to_current_v2v_policy_at_load tests/unit/test_c4_v2v_obs.py
```

Review was CPU-only, with no SSH, remote jobs, SUMO or GPU execution. No forbidden-path contents or prohibited truth artifacts were opened. No implementation edits or commits were made. The only review-authored worktree file is `REVIEW_T1B_r3.md`.

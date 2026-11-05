VERDICT: PASS

## Numbered blockers (current review)

None. Prior blocker 1 is resolved by `3be81d33`; no required T2H fixes remain at reviewed HEAD `a5a9bc22`.

## Test counts (current review)

- Fresh independent CPU run at `a5a9bc22`: **96 passed, 0 failed, 0 skipped, 0 deselected**, in **45.92 seconds**: 61 T2H cases and 35 practice-start regression cases.
- One configuration warning: `Unknown config option: cache_dir`, caused by disabling pytest's cache provider.
- Preserved prior independent broader run: **209 passed, 0 failed, 1 skipped, 1 deselected**, in 275.76 seconds. This is historical on-disk evidence, not a fresh run at this HEAD. The affected suites were rerun above; the unchanged broader suites were not duplicated. These counts overlap and must not be added.
- The prior skip was the Windows symlink-privilege case; the prior deselection avoided a synthetic fixture that opens a prohibited path. Neither is part of the fresh 96-case run.
- `git diff --check c4/track-base` passed before the report update.

Working directory: `E:/agent-work/c4-practice-weight`. Environment: `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q -p no:cacheprovider --basetemp=E:/Temp/opencode/t2h-astra-r0-a5a9bc22-20261105 tools/coopmwm/test_practice_weighting_cpu.py tools/coopmwm/test_practice_starts_cpu.py --tb=short
```

## Current independent review (astra, 2026-11-05)

Started with a clean worktree on `c4/practice-weight`, HEAD `a5a9bc22`. Read git status/log and `T2H_READY.json` first, then the current shared spec, existing review, PROGRESS/ACCEPTED receipts, and the changes against `c4/track-base` (`13f184b5`). Preserved the existing implementation and completed review evidence. Checked local process command lines before testing: no matching pytest/T2H review job was running.

### Correctness and rule compliance

1. **Own all8 admission: prior blocker resolved.** `tools/coopmwm/practice_weighting.py:47-99` now checks a sealed own-arm all8 confirmation, its schema/mode/bank rule and admitted status, an exact admitted cooperative example, matching seed, candidate index and plan digest, the manifest's confirmation digest, and a sealed first-seed SUMO receipt bound to that confirmation, arm, candidate plan and original snapshot. A rejected first-seed receipt is allowed only with the separately admitted all8 example, supporting seed-2 winners. `practice_weighting.py:128-161` additionally checks the own admitted parent, decision link, manifest identity and replicated snapshot hash before plan sealing. The WM flag is proposal provenance, not admission evidence. This satisfies the revised spec at `tracks_spec.md:54-57`.
2. **Frozen replay and capacity.** `tools/coopmwm/practice_starts.py:198-219` revalidates augmented all8 proof and snapshot hash at replay, so obsolete WM-only frozen rows refuse. It does not reread the manifest or resample a frozen plan. `practice_weighting.py:41-44,123-161` and `practice_starts.py:137-145` deduplicate cooperative restart state/seed/CRN keys, including augmentations already visible in the ordinary bank reader.
3. **Observable-only inputs and hidden truth.** The additions select own-bank restart records and carry provenance. They add no policy/planner observation features, global-state inputs, hidden driver-type inputs, or truth-file readers. Snapshots remain replay machinery inputs; they are not added to the policy observation vector.
4. **Look-ahead, horizon, and safety.** No planner look-ahead or scoring horizon is shortened. Existing full 1200-second episode execution, CRN restoration, matched A references, ledger charging, prefix checks, and safety comparisons are retained. `tools/coopmwm/test_practice_starts_cpu.py:933-972` executes the real episode path with a fake plant and checks the tape through time 1199 and prefix/CRN rejection. The current run also includes matched-reference and collision-guard tests.
5. **B/R fairness.** `tools/coopmwm/v2v_launch3.py:355-392` validates identical options for B/R, including per-arm declarations. Both arms use the same augmentation validator and cooperative sampling logic. `v2v_runner.py:264-289` preserves R's wait for B's realised practice count; the existing documented own-bank fallback remains, after cooperative capacity is exhausted. Bank shortages can reduce realised counts, as documented. A does not receive practice starts.
6. **Default-off byte identity.** `tools/coopmwm/test_practice_weighting_cpu.py:98-109,353-383` checks no augmentation read when disabled and compares default configuration JSON and sealed B/R plan bytes against executable shared-base source with timestamps fixed. Those tests passed with legacy practice starts both off and on. Explicit false retains the caller's supplied key but preserves legacy behavior; absent keys emit no new fields. The executable-source qualification list includes the new imported module.
7. **Registered values and budget hooks.** `tools/coopmwm/practice_weighting.py:17-37` accepts only the documented prospective 0.50 sampling share; the legacy 0.25 share and repeat cap remain. No new driving-behavior thresholds were introduced. Training replaces normal starts rather than adding training episodes. Allocation, dispatch, matched-reference lookup and resume accounting use the effective share; the full-capacity matched-reference upper bound is 691200 sim-s. Existing transfer/ceiling validation remains.
8. **Tests are substantive CPU checks.** `tools/coopmwm/test_practice_weighting_cpu.py:159-214` rejects 14 defective proof variants in each arm before plan sealing. Lines 217-269 cover seed-2 restoration, real `verify_perseed.confirm_all8` producer output in both arms with a deterministic CPU `SeedPlant`, and frozen replay rejection. Lines 272-350 exercise actual runner scheduling, update receipts, fairness, fallback, and quotes with execution seams mocked. The validation, bank readers, plan selection and receipts are real; passing these tests is not a claim of simulator execution or demonstrated learning.

### Integration boundary

`tools/coopmwm/PRACTICE_WEIGHTING.md:25-63` now documents the verified producer contract correctly. T2G/T2C own production augmentation, the <=200000 sim-s pre-launch verification budget, end-to-end rerun wiring/smoke, and the total <=5000000 sim-s assertion. This PASS covers T2H's adapter, runtime hooks and CPU contracts; it does not certify those separate integration deliverables or authorize execution.

### Review constraints and receipt reconciliation

Only this review file was edited; no implementation edits or commits were made. Tests used the approved external temporary directory, with bytecode and pytest cache writes disabled. No SSH, remote dispatch, SUMO process, GPU operation, prohibited data path, or truth file was used by this review. Existing unrelated jobs were left untouched.

The current PASS supersedes the historical BLOCK below. `T2H_READY.json` and `T2H_ACCEPTED.json` remain unchanged historical receipts; their statements that independent rereview is pending are superseded by this report. The prior review text is retained verbatim below to preserve the resumed session's evidence; its findings and line references apply to the older implementation only.

---

## Historical review (superseded; preserved evidence)

VERDICT: BLOCK

## Numbered blockers

1. **Augmented practice states bypass the current SUMO-verification requirement.** `tools/coopmwm/practice_weighting.py:70-95` checks that the *parent* is admitted, that the augmentation has an own-arm decision link and snapshot hash, and that its manifest says `wm_cooperative=True`. It never requires the augmented example itself to have an accepted all8 verification/admission record. `tools/coopmwm/practice_starts.py:137-140` adds these rows directly to the cooperative sampling pool. A decision link is only a JSON mapping (`practice_starts.py:66-72`), not proof of verification. The current spec, `E:/agent-work/_sol/tracks_spec.md:54-57`, supersedes WM-only retention: augmented plans may enter the bank only after SUMO verification (all8), with WM used only to propose variants. The documented adapter contract still explicitly accepts WM-retained augmentations (`tools/coopmwm/PRACTICE_WEIGHTING.md:33-48`). **Required fix:** require a validated, accepted own-arm all8/admission record for each augmented example, bound to its augmentation/plan, state and seed; do not treat its parent's verification or a manifest boolean as sufficient. Consume already verified producer records without launching verification from this adapter. Update the producer contract and CPU tests to reject missing, rejected, foreign-arm or mismatched augmented verification records before sealing/dispatch, and accept correctly bound verified augmentations for both B and R. Existing tests currently assert the obsolete behavior: `tools/coopmwm/test_practice_weighting_cpu.py:23-36` creates an augmented snapshot/link/WM flag with no augmented verification receipt; lines 90-105 and 150-157 successfully select it and carry it into the training-wave/update path. The three focused cases passed again on resume, confirming the acceptance gap rather than satisfying the revised requirement.

## Test counts

- Preserved prior independent completed run: **209 passed, 0 failed, 1 skipped, 1 deselected** (211 collected), in 275.76 seconds. This is the on-disk evidence from the prior session; the completed full run was not duplicated.
- Preserved prior-resume focused spec recheck: **3 passed, 0 failed, 0 skipped**, in 31.66 seconds. These are three of the previously counted cases, not three additional unique tests. They demonstrate acceptance of unverified augmentations; passing the old assertions does not resolve blocker 1.
- This includes all **28 T2H tests**, all **35 existing practice-start tests**, and the permitted launch, budget, runtime, safety, and rerun regression cases listed below.
- Skipped: `tools/coopmwm/test_rerun_cpus_cpu.py:192`, Windows symlink privilege unavailable; the hard-link case passed.
- Deliberately deselected: `tools/coopmwm/test_rerun_cpus_cpu.py::test_staging_manifest_refuses_forbidden_and_missing_inputs`. Its lines 340–341 open a temporary filename containing a prohibited token. The review's hard rule applies even to synthetic fixtures. This accounts for the one-pass difference from the readiness receipt's 210 passed; that case was not independently executed.
- One pytest configuration warning: disabling the cache provider makes the repository's `cache_dir` option unknown. No test failures resulted.
- An initial attempt timed out after 120 seconds without a final summary. Its progress is not counted; the complete rerun above is the result used for this verdict.
- `git diff --check c4/track-base` passed.

### Preserved prior-resume focused command

Environment: `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. No matching T2H review test process was running before this invocation; unrelated jobs were left untouched.

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q -p no:cacheprovider --basetemp=E:/Temp/opencode/t2h-r0-spec-recheck-20261105 tools/coopmwm/test_practice_weighting_cpu.py::test_bank_plus_augmented_distinct_states_and_frozen_resume tools/coopmwm/test_practice_weighting_cpu.py::test_augmented_state_reaches_real_wave_and_update_receipt
```

The focused run emitted the same single `cache_dir` configuration warning. Tests used synthetic on-disk receipts and a fake episode wave; no simulator was launched.

### Reproduction command

Working directory: `E:/agent-work/c4-practice-weight`. Environment: `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Test fixtures were directed to the approved temporary directory; pytest's worktree cache was disabled.

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q -p no:cacheprovider --basetemp=E:/Temp/opencode/t2h-r0-verified tools/coopmwm/test_practice_weighting_cpu.py tools/coopmwm/test_practice_starts_cpu.py tools/coopmwm/test_v2v_launch3_cpu.py tools/coopmwm/test_v2v_launch_wiring_cpu.py tools/coopmwm/test_v2v_launch_quote_cpu.py tools/coopmwm/test_v2v_budget_quote_cpu.py tools/coopmwm/test_v2v_live_cpu.py tools/coopmwm/test_v2v_runner_cpu.py tools/coopmwm/test_composite_health_cpu.py tools/coopmwm/test_warm_health_cpu.py tools/coopmwm/test_rerun_cpus_cpu.py tools/coopmwm/test_rerun_clean_cpu.py --deselect=tools/coopmwm/test_rerun_cpus_cpu.py::test_staging_manifest_refuses_forbidden_and_missing_inputs
```

## Resume reconciliation (2026-11-05)

Latest fresh-session verification (astra): reread status/log, READY/ACCEPTED/PROGRESS receipts, this uncommitted report, the current shared spec, and all implementation/test/documentation changes against `c4/track-base`. HEAD remains `93da75f6`; blocker 1 is still directly present in the adapter and its tests. The completed pytest results above are preserved prior-session evidence, not newly executed results. No additional test run or dispatch was needed because neither the implementation nor the identified requirement changed.

Fresh-session continuation (astra): initial status was `M REVIEW_T2H_r0.md` on `c4/practice-weight`, HEAD `93da75f6`; the BLOCK review and focused-test evidence below were already on disk as uncommitted work. Read status/log and task receipts first, then the current spec and the implementation/test diff against `c4/track-base`. Confirmed blocker 1 remains present: the augmentation adapter validates a parent, decision link, WM flag and snapshot hash, but no augmented all8 admission. Preserved the existing review and completed test results. No tests or dispatches were started again, consistent with the resume instruction not to duplicate finished work. The clean-start and focused-run descriptions below refer to the preceding review session. Only this review file was updated; no commit was made.

Independent verifier: astra. Started with a clean worktree on `c4/practice-weight`, HEAD `93da75f6`. Read git status/log, `T2H_READY.json`, `T2H_PROGRESS.json`, `T2H_ACCEPTED.json`, this existing report, and the current shared spec before reviewing the diff against `c4/track-base`. HEAD only preserved the earlier review/acceptance and updated readiness metadata; the implementation remains the one reviewed at `ee3af837`. The prior PASS and test evidence are preserved below as historical evidence, but the current verdict supersedes that PASS because the latest spec requires verified augmentations. The existing READY/ACCEPTED receipts were not modified and still describe the earlier acceptance; they must not be interpreted as clearing blocker 1.

The current diff adds no policy/planner observable features or truth readers, changes no look-ahead/horizon, retains the shared B/R option validation and realized-count scheduling, and preserves absent-key default configuration/plan behavior as covered by the existing differential tests. The added 0.50 share is declared as the prospective registered alternative; no new driving-behavior thresholds were found. The unresolved issue is augmentation eligibility under the revised spec. The full-run 5,000,000 sim-s cap and production rerun wiring remain T2C integration responsibilities; the T2H reference quote is not evidence of that end-to-end budget check.

## Preserved prior review scope and evidence

Independent verifier: astra. Reviewed `E:/agent-work/_sol/tracks_spec.md`, `T2H_READY.json`, and the complete ten-file diff from `c4/track-base` (`13f184b5a7e12be04aebf246ef8e4dbb81555545`) to branch `c4/practice-weight`, HEAD `ee3af8371e69a99f12657bf735ac5a9d3a1b5b8f`. Supporting runtime and test code was inspected where necessary.

- **T2H behavior:** `practice_weighting.py:16–37` declares and validates the prospective 0.50 alternative; the legacy share remains 0.25. `v2v_runner.py:264–289` uses the effective share in the actual training-wave path. Tests exercise 12 practice plus 12 normal episodes, augmented starts reaching the wave and update receipt, and normal-start fallback for an empty bank.
- **Own-bank augmentation:** `practice_weighting.py:46–96` verifies the manifest seal/schema/arm, an own admitted cooperative parent, an own decision link, seed-index agreement, the WM-retained flag, safe state identifiers, and the snapshot hash after CRN derivation. It filters duplicate augmented restart keys against bank parents and previously accepted augmentation rows. `practice_starts.py:193–210` rechecks the hash when replaying an augmented row. This adapter consumes producer-supplied WM labels; it does not itself score or verify an augmented plan.
- **Observables and hidden truth:** the additions select restart records and carry replay/receipt metadata. They introduce no policy/planner observation features or hidden driver-type/global-state inputs. Snapshot contents remain in the existing replay controller path; selection does not introduce a truth-data reader.
- **Look-ahead and safety:** no planner scoring horizon or look-ahead code changes. The existing 1200-second episode execution and charging remain in place. Tests run the real episode function against a fake plant and assert the tape reaches time 1199. Matched A references retain the same snapshot/CRN and pre-update A checkpoint; existing collision, pairing, and failure-replacement tests pass.
- **B/R fairness:** `v2v_launch3.py:355–392` enforces equal options, including per-arm declarations, and rejects practice options for A. Both bank arms use the same augmentation adapter and weighted cooperative-pool sampling. R continues to request B's realized count and retains the pre-existing own-bank fallback rule. Capacity shortages can reduce realized starts, as documented; the implementation does not promise unavailable R examples.
- **Default-off identity:** absent new keys are not emitted. The differential tests in `test_practice_weighting_cpu.py:216–246` load the actual shared-base runner and practice planner through `git show`, compare default configuration JSON, and compare sealed B/R plan bytes with time fixed. Other tests establish that disabled weighting does not read augmentation and that legacy scheduling/receipts remain unchanged. Explicit opt-out configuration retains the caller-supplied new key, as documented; the absent-key default is the byte-comparison case. The executable-source inventory intentionally includes the new dependency.
- **Registered values:** the only added sampling share is the documented prospective 0.50 value; invalid alternatives and invalid flag types refuse. Existing fraction/repeat-cap constraints remain. No new acceleration, gap, trigger, safety, or other driving-behavior thresholds were introduced. This registration is not execution approval.
- **Budget and resume:** allocation, child dispatch quotes, fresh watcher quotes, restart credits, and per-arm remaining-reference quotes use the effective fraction. Full-capacity matched-reference cost is 691200 sim-s. Frozen plans are reused before rereading augmentation, preserving job identity across restart.
- **Tests are substantive:** they build synthetic on-disk banks with sealed receipts, use real bank readers and CRN restoration, execute actual scheduling/update-receipt paths, compare against base source, and exercise malformed-manifest refusals and quote/resume paths. SUMO execution and process dispatch are replaced at test seams. These are CPU contract/regression tests, not a simulator learning demonstration.

## Integration boundary

The base has no T2G producer. `PRACTICE_WEIGHTING.md:24–48` supplies the manifest contract; its producer is represented by CPU fixtures here. T2G/T2C must wire the additive options into their end-to-end rerun, using an updated contract that requires verified augmentations. The readiness receipt identifies the producer dependency, and the shared spec assigns rerun integration and the all-components smoke to T2C. That separate ownership does not make the current WM-only adapter contract compliant with the revised spec.

## Review constraints

No implementation edits or commits were made. No SSH, remote execution, SUMO process, or GPU work was run. No prohibited data/truth paths were opened. The latest session began with this report already modified; it remains the only review-authored worktree change. The clean-worktree and focused-testing statements above describe earlier sessions. Existing implementation, receipts, prior regression evidence and unrelated running jobs were preserved.

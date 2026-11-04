VERDICT: PASS

## Numbered blockers

1. None found in the T2H implementation diff. No required fixes.

## Test counts

- Independent completed run: **209 passed, 0 failed, 1 skipped, 1 deselected** (211 collected), in 275.76 seconds.
- This includes all **28 T2H tests**, all **35 existing practice-start tests**, and the permitted launch, budget, runtime, safety, and rerun regression cases listed below.
- Skipped: `tools/coopmwm/test_rerun_cpus_cpu.py:192`, Windows symlink privilege unavailable; the hard-link case passed.
- Deliberately deselected: `tools/coopmwm/test_rerun_cpus_cpu.py::test_staging_manifest_refuses_forbidden_and_missing_inputs`. Its lines 340–341 open a temporary filename containing a prohibited token. The review's hard rule applies even to synthetic fixtures. This accounts for the one-pass difference from the readiness receipt's 210 passed; that case was not independently executed.
- One pytest configuration warning: disabling the cache provider makes the repository's `cache_dir` option unknown. No test failures resulted.
- An initial attempt timed out after 120 seconds without a final summary. Its progress is not counted; the complete rerun above is the result used for this verdict.
- `git diff --check c4/track-base` passed.

### Reproduction command

Working directory: `E:/agent-work/c4-practice-weight`. Environment: `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Test fixtures were directed to the approved temporary directory; pytest's worktree cache was disabled.

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q -p no:cacheprovider --basetemp=E:/Temp/opencode/t2h-r0-verified tools/coopmwm/test_practice_weighting_cpu.py tools/coopmwm/test_practice_starts_cpu.py tools/coopmwm/test_v2v_launch3_cpu.py tools/coopmwm/test_v2v_launch_wiring_cpu.py tools/coopmwm/test_v2v_launch_quote_cpu.py tools/coopmwm/test_v2v_budget_quote_cpu.py tools/coopmwm/test_v2v_live_cpu.py tools/coopmwm/test_v2v_runner_cpu.py tools/coopmwm/test_composite_health_cpu.py tools/coopmwm/test_warm_health_cpu.py tools/coopmwm/test_rerun_cpus_cpu.py tools/coopmwm/test_rerun_clean_cpu.py --deselect=tools/coopmwm/test_rerun_cpus_cpu.py::test_staging_manifest_refuses_forbidden_and_missing_inputs
```

## Reviewed scope and evidence

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

The base has no T2G producer. `PRACTICE_WEIGHTING.md:24–48` supplies the manifest contract; its producer is represented by CPU fixtures here. T2G/T2C must publish that contract and wire the additive options into their end-to-end rerun. The readiness receipt explicitly identifies this dependency, and the shared spec assigns rerun integration and the all-components smoke to T2C. This PASS covers T2H's implementation and CPU tests, not completion of that separate integration or permission to launch training.

## Review constraints

No implementation edits or commits were made. No SSH, remote execution, SUMO process, or GPU work was run. No prohibited data/truth paths were opened. The worktree was clean after testing and before this review file was added; this report is the only review-authored worktree change.

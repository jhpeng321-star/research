VERDICT: PASS

## Numbered blockers

None. Review r0 blocker 1 is resolved; no remaining G1 blocker was identified.

## Test counts

- Independently reran the readiness manifest's five-file CPU suite with the required interpreter: **249 passed, 1 skipped, 0 failed**, 162.44 s. This includes the 65 G1 test cases.
- Independent, in-memory pytest comparison against pre-G1 revision `6dc0520c99f0bccd71669e66d2c4124197db990d`: **2 passed, 0 failed**, 10.56 s. B and R each produced identical default-off teaching receipts, byte-identical model and optimizer tensors, and identical RNG states.
- **Final verification total: 251 passed, 1 skipped, 0 failed (252 cases).**
- Skip: `tools/coopmwm/test_launch_integration_cpu.py:526`, the existing Windows file-symlink privilege check, deferred by that test to later qualification.
- Both successful pytest invocations emitted `PytestConfigWarning: Unknown config option: cache_dir`, because this review disabled pytest's cache provider.
- Verifier harness attempts, excluded from the final verification total: an initial inline command had a shell-quoting syntax error before pytest started; the first in-memory comparison then had **2 setup-related test failures** because the verifier omitted the suite's test-only `exploration.regime` replacement and encountered unavailable `tools.leverprobe`. Restoring that existing CPU fixture behavior yielded the two passing comparisons above. Neither attempt changed implementation or test source files.
- `git diff --check c4/track-base`: passed.

Suite command, executed from the worktree root:

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tools/coopmwm/test_executable_labels_cpu.py tools/coopmwm/test_bank_breadth_cpu.py tools/coopmwm/test_launch_integration_cpu.py tools/coopmwm/test_f1_actuation_cpu.py tools/coopmwm/test_clean2_actuation_receipts_cpu.py -p no:cacheprovider --basetemp=E:/agent-work/c4-grp-certify/_out/g1_review_r1_pytest
```

Environment: `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. The additional comparisons used an in-memory pytest plugin under the same interpreter/environment, with scratch directory `_out/g1_review_r1_identity_fixed`.

## Review scope

- Independent verifier: astra; date: 2026-11-06.
- Worktree: `E:/agent-work/c4-grp-certify`; branch: `c4/grp-certify`; reviewed HEAD: `4718f00041d71b9f81310c3c87d772975105d941`.
- Requested comparison base: `c4/track-base`, `13f184b5a7e12be04aebf246ef8e4dbb81555545`.
- Read `E:/agent-work/_sol/tracks_spec.md`, `G1_READY.json`, G1 documentation/budget, the prior review, implementation/tests, and relevant observable replay, move, trigger, safety, statistics, and actuation dependencies. Inspected the requested base comparison and G1 file diffs. That comparison includes substantial inherited integration work; G1's seven-file delta was additionally attributed against its recorded initial HEAD `6dc0520c99f0bccd71669e66d2c4124197db990d`. This verdict concerns G1 and its integration.

## Correctness and rule checks

1. **r0 causal replay fix:** `tools/coopmwm/bank_breadth.py:252-300` retains the observer at `decision_s - 1`, validates the actual historical local rows against the certificate, and rebuilds sensor and local-partner inputs from that view. The decision-time frame/pair establishes actor presence and eligibility without replacing certificate geometry. Certified availability constrains candidate generation. The replacement tests at `tools/coopmwm/test_executable_labels_cpu.py:325-465` use real `EpisodePolicy`/`IntentPolicy` observers, changing positions and speeds, stationary controls, reserved-visible actors, and enabled manifest-to-F3-to-trigger learning for both B and R. They also reject stale/mismatched observations and missing/inactive actors before learning. These tests passed.

2. **Executable binding and admission:** `tools/coopmwm/executable_labels.py:66-217` binds source receipt/example/plan, snapshot/prefix/base-policy/runtime hashes, exact macro, proposer, roles, observation time, availability, and original source window. Jobs explicitly include group, base, and every member solo on two distinct restart/RNG bindings. `admit` reconstructs the canonical job and rejects missing, duplicate, altered, misaligned, or incomplete branch evidence (`:281-342`). Group issued-command audits cross-check library/raw/safe deltas and require simultaneous physical effects from at least two actual actors on each seed (`:251-278`). Negative and incomplete outcomes remain diagnostics in manifests (`:385-403`).

3. **Synergy and safety:** statistics reselect the best full-window solo inside each paired bootstrap draw, report total benefit separately, and require strictly positive pooled and per-seed synergy CI95 lower bounds (`tools/coopmwm/executable_labels.py:224-248,332-334`). Admission applies the existing composite group-minus-base safety rule and requires zero collisions/teleports on all branches; missing counters fail closed (`:312-331`). Tests cover zero/negative/uncertain synergy, alternating best solos, unsafe outcomes, clipping, and asynchronous effects, rather than merely checking schema presence.

4. **Observables only / no hidden truth:** reviewed G1 geometry and teacher inputs use local observable rows and public CAV identity, range, roles, and availability. `EpisodePolicy.observe` whitelists public fields and filters local sensing/partner membership (`tools/coopmwm/track2_backend.py:295-338`); certified replay validates that filtered view. Trigger features use local partner payloads and zero global broadcast (`tools/coopmwm/trigger_policy.py:155-186`). No hidden-driver/global-state policy input or truth-file read was identified in the reviewed G1 paths.

5. **No shortened look-ahead:** binding requires the source receipt's original window, canonical branches preserve it for group/base/all solos, and results must provide the complete per-second series with matching prefix/CRN/horizon declarations (`tools/coopmwm/executable_labels.py:139-146,200-217,304-311`). Bank matching rechecks that window (`:420-425`). Full prefix replay is included in the quote.

6. **B/R fairness and budget:** both arms use common binding/admission and F3 integration. Concrete quotes enforce equal context counts, matching family/arity, full-prefix lengths, and verification seeds, with at most eight contexts per arm (`tools/coopmwm/executable_labels.py:345-375`). Existing F3 teaching/practice enforce matching supported families and family distributions. Prospective preference is for two-actor/native-profile fidelity without outcome ranking (`:379-382`). Preferred two-actor cost is **153,600 sim-s**; the four-member ceiling is `8 * 2 * 2 * 6 * 1200 = 230,400 sim-s`, matching `G1_BUDGET.json`. The suite constructs a full batch that hits this ceiling and checks over-cap/context and unequal-effort rejection.

7. **Default-off identity and A:** manifest reads and certificate gates are opt-in; A returns before bank/certificate reads (`tools/coopmwm/bank_breadth.py:71-78`; `tools/coopmwm/launch_production.py:84-85`). The uncertified replay path remains separate (`tools/coopmwm/bank_breadth.py:313-349`). Independent B/R comparisons against the pre-G1 implementation verified exact receipt equality and byte identity of resulting model/optimizer tensors and RNG state for nonempty teaching fixtures. This establishes fixture-level compatibility, not a simulated whole-run equivalence claim.

8. **Registered thresholds:** geometry and move ranges come from the existing move library; bootstrap draw count, block size, and RNG come from `bank_statistics`; the physical-effect tolerance matches `clean2_receipts`; safety uses `bank_safety`. The strict positive CI95 gate follows G1. No additional hand-set driving threshold was identified in the G1 delta.

## Handoff boundary and review integrity

G1 is a CPU-built job-spec/admission implementation. Its passing tests exercise real observers, model/optimizer updates, controllers, and admission logic with synthetic CPU evidence; they do not establish measured SUMO synergy. As documented in `G1_READY.json:122-126`, a later authorized adapter must execute the explicit common-background, full-window, matched-CRN and issued-command-audit contract before publishing genuine certificates. That later adapter is outside the requested G1 build scope.

Review was CPU-only, with no SSH, SUMO, GPU execution, prohibited-path content access, or truth-file access. No source changes or commits were made. The pre-existing untracked `G1_READY.blocked0_1793894069.json` and `REVIEW_G1_r0.md` were preserved. The only authored deliverable is this report; pytest generated worktree-local scratch artifacts.

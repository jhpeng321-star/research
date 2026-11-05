VERDICT: BLOCK

## Numbered blockers

1. **P1 — Certified moving contexts are silently discarded because certificate and replay geometry use different timestamps.** `tools/coopmwm/bank_breadth.py:277-290`, with the binding contract at `tools/coopmwm/executable_labels.py:72-73` and the real observer at `tools/coopmwm/track2_backend.py:295-310,336-338`.

   `bind_lesson` requires `local.observed_at_s == decision_s - 1`. However, `observable_contexts` observes both `time-1` and `time`, then compares the certificate's rows against `reader.local_states`, which the second observation has replaced with **time** geometry. The equality includes positions and speeds. An ordinary moving vehicle therefore makes a valid t-1 certificate fail the comparison and yields no teaching example. If all certified contexts move, `launch_production.py:113-114` raises `nonempty observable verified-plan trigger supervision required`. The practice admission path can still accept these same certificates, so it does not establish usable trigger supervision.

   **Independent CPU reproduction:** use the existing `lesson()` fixture (decision 301, observation 300, two wall actors at -1500/-1510 m, speed 10 m/s), the real `EpisodePolicy`/`IntentPolicy` observer, and a pair at time 301. Supply those positions at frame 300 and advance both positions by 10 m at frame 301. `observable_contexts(..., certification=bound)` returns **0** examples; the same check with unchanged geometry returns **1**. The moving-case pytest assertion fails. No simulation or external data is involved.

   The new test does not catch this: `tools/coopmwm/test_executable_labels_cpu.py:324-337` uses a `Reader.observe` that does nothing and permanently fixes `local_states` to the certificate rows.

   **Required fix:** make certificate binding, replay validation, and teacher input reconstruction use an explicit, consistent causal observation-time contract. Under the current t-1 certificate contract, retain and validate the actual t-1 view rather than comparing it with overwritten t geometry; ensure the resulting teacher features and availability checks correspond to the certified executable decision. Do not fix this by relabelling current geometry as t-1 or removing the binding check. Add a CPU regression using the real observer with changing positions/speeds, including successful certified trigger supervision and rejection of genuinely mismatched/stale evidence. Exercise the enabled path for both B and R.

## Test counts

- Reported five-file CPU suite independently rerun: **222 passed, 1 skipped, 0 failed**, 227.09 s.
- Additional in-memory pytest checks using the real observer: **1 passed, 1 failed**, 13.63 s. The failure is blocker 1; the stationary control passes.
- Additional in-memory pytest checks comparing default-off teaching against pre-G1 revision `6dc0520c`: **2 passed, 0 failed**, 17.99 s. B and R each produce identical teaching receipts and byte-identical resulting model-state tensors.
- **Combined: 225 passed, 1 failed, 1 skipped (227 cases).** The independent checks were supplied through an in-memory pytest plugin; no test source files were added or modified.
- Skip: `tools/coopmwm/test_launch_integration_cpu.py:526`, Windows file-symlink privilege check deferred by the existing test.
- Each invocation emitted one `PytestConfigWarning: Unknown config option: cache_dir` because the verifier disabled pytest's cache provider. This is unrelated to the assertion failure.
- `git diff --check c4/track-base`: passed.

Suite command, from the worktree root, with `CUDA_VISIBLE_DEVICES=-1` and `PYTHONDONTWRITEBYTECODE=1`:

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tools/coopmwm/test_executable_labels_cpu.py tools/coopmwm/test_bank_breadth_cpu.py tools/coopmwm/test_launch_integration_cpu.py tools/coopmwm/test_f1_actuation_cpu.py tools/coopmwm/test_clean2_actuation_receipts_cpu.py -p no:cacheprovider --basetemp=E:/agent-work/c4-grp-certify/_out/g1_review_r0_pytest
```

## Scope and rule checks

- Independent verifier: astra; date: 2026-11-05. Reviewed branch `c4/grp-certify`, HEAD `f5ff4781b8eefe2484c35ace27f333c14306f3f5`; requested comparison base `c4/track-base` = `13f184b5a7e12be04aebf246ef8e4dbb81555545`.
- Read `E:/agent-work/_sol/tracks_spec.md`, `G1_READY.json`, G1 documentation/budget, the G1 implementation/tests, and relevant bank, observer, actuation, safety, statistics, and teaching dependencies. The requested base comparison contains substantial inherited integration work. G1's own changes were additionally attributed against its recorded initial HEAD `6dc0520c`; this review's verdict concerns G1 and its integration dependencies.
- **Observables / hidden truth:** G1's geometry and role checks use local observable rows, visible CAV membership, availability, and existing registered move geometry. The real observer whitelists public fields and filters local sensing. No hidden-truth input or truth-file read was found in the reviewed G1 paths. The causal timestamp mismatch is the blocker above.
- **Look-ahead / common comparisons:** binding requires the original source receipt window, and bank matching rechecks it. Canonical jobs preserve that window for group, base, and every member solo on both verification seeds. Admission rejects incomplete series, mismatched windows/prefix/CRN declarations, and incomplete branch sets. No new shortened scoring window was found.
- **Synergy / safety:** the code reselects the best full-window solo inside each paired bootstrap draw, separates total benefit from synergy, requires strictly positive pooled and per-seed lower bounds, checks physical simultaneous effects, and applies the existing composite safety rule plus zero collisions/teleports on every branch. Negative and incomplete result entries remain diagnostic-only. The passing suite contains substantive negative tests and real model/controller/optimizer checks; it does not demonstrate actual SUMO certification, which G1 explicitly leaves to a later adapter.
- **B/R fairness / budget:** both arms share admission code and opt-in configuration. Concrete quotes enforce matching context counts, family/arity, simulation lengths, and verification seeds; inherited F3 teaching/practice retain matched family support/distributions. The full-prefix ceiling is `8 * 2 * 2 * (group + base + 4 solos) * 1200 = 230400` sim-s; preferred two-actor cost is 153600 sim-s. These are prospective quotes, with no execution in this review.
- **Default-off identity:** additive branches are guarded by the manifest option; A returns before certificate/bank reads. The independent B/R pre-G1 comparison verified exact receipt equality and model-state byte identity for the exercised teaching fixtures. This is a CPU fixture-level check, not a whole-run simulation equivalence claim.
- **Registered thresholds:** move ranges and geometry are imported from the existing library; bootstrap draws/block size/RNG are reused from `bank_statistics`; the `1e-6` physical-effect tolerance matches `clean2_receipts`; safety uses `bank_safety`. No additional hand-set driving threshold was identified in the G1 delta.
- Review was CPU-only. No SSH, SUMO, GPU execution, prohibited-path access, or truth-file access was performed. No source changes or commits were made. Before writing this report, `git status --short` was clean; pytest generated only its local scratch artifacts.

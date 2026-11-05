VERDICT: PASS

## Blockers

None (0). No required fixes identified for G2.

## Test counts

- Independently completed CPU pytest run: **330 passed, 0 failed, 0 errors, 0 skipped, 11 deselected; 5 subtests passed**. Exit code **0**, elapsed **138.97 s**.
- The passing selection includes all **29 G2 cases**. JUnit reports 335 successful cases including the five subtests.
- The 11 existing `forbidden_paths_refused_before_open` parameterizations were deselected to retain the submitted restriction-compatible selection; they are not counted as verified.
- One pytest configuration warning: `cache_dir` is unknown when `cacheprovider` is disabled.
- Additional independent in-memory probes: **4 passed** (default-off versus actual pre-G2 code, and enabled B/R parity, each with deterministic and stochastic sampling). Each probe compared **131 consecutive control ticks**, including approach entry and periodic decisions, with exact candidate-RNG and trigger-RNG state equality.
- An initial suite attempt exceeded the tool's 120-second limit and is not counted as a completed run. The completed run above used an extended timeout.

Command executed from `E:/agent-work/c4-grp-opportunity`:

```text
C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -m pytest -q tools/coopmwm/test_coordination_opportunities_cpu.py tools/coopmwm/test_feasible_proposals_cpu.py tools/coopmwm/test_track2_backend_cpu.py tools/coopmwm/test_trigger_policy_cpu.py tools/coopmwm/test_moves_cpu.py tools/coopmwm/test_track2_monitor_cpu.py -k "not forbidden_paths_refused_before_open" -p no:cacheprovider --basetemp=E:/Temp/opencode/astra_g2_r0_complete --junitxml=E:/Temp/opencode/astra_g2_r0_results.xml
```

Environment: `CUDA_VISIBLE_DEVICES=-1`, `PYTHONDONTWRITEBYTECODE=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; `TEMP` and `TMP` set to `E:/Temp/opencode`. Test-generated artifacts were outside the worktree.

## Review basis

- Independent verifier: astra; review date: 2026-11-05.
- Read `E:/agent-work/_sol/tracks_spec.md`, including the G2 requirement at line 73, and `G2_READY.json`.
- Reviewed branch `c4/grp-opportunity`, HEAD `c8ca54f5450a890eb7a4f229e557c69fce359736`, against requested base `c4/track-base` (`13f184b5a7e12be04aebf246ef8e4dbb81555545`). Confirmed that base is an ancestor.
- The requested base diff contains inherited integrated rerun work. Traced G2's six changed/added files and its runtime dependencies, and isolated its incremental changes against recorded parent `6dc0520c99f0bccd71669e66d2c4124197db990d`. The two G2 commits are implementation `8e0ce364` and readiness `c8ca54f5`. This verdict covers G2, not a fresh certification of every inherited rerun change.
- `git diff --check` against the requested base passed for all six G2 files.

## Verification findings

- **Observable inputs and ownership:** `tools/coopmwm/coordination_opportunities.py:33-48,74-105` copies a whitelist of observable geometry, builds each edge from the initiating ego's local view and authorized CAV slots, and excludes unavailable, out-of-zone, distant and non-CAV members. `tools/coopmwm/track2_backend.py:299-356` supplies sensing-filtered rows and locally filtered controller reservations. No new truth-file, hidden-driver, global reward or WM input enters G2. Other egos' traffic observations are not unioned into the initiating ego's features; matching and ownership serve arbitration.
- **Pair-first, disjoint options with retained alternatives:** `tools/coopmwm/coordination_opportunities.py:108-157` greedily selects disjoint role pairs and supplies pair-bound library candidates, including a two-actor `ramp_assist` and singleton `ease_off`. `tools/coopmwm/feasible_proposals.py:103-159` prioritizes matched mainline proposers, validates supplied pairs, recomputes ownership before each conditional slate, and leaves the trigger's separate none action available. Tests exercise two simultaneous disjoint acquisitions and real none/single selections.
- **Executable vocabulary:** `tools/coopmwm/moves.py:129-136,156-184` establishes the actual actors and offsets. G2's `ramp_assist` acts on both role members; `gap_open` acts on one and is correctly excluded from the physically joint numerator. G2 uses the existing shared-window move vocabulary; neither this review nor the receipts infer measured positive synergy from role labels.
- **Exact conditional PPO:** `tools/coopmwm/feasible_proposals.py:143-165` retains recorded tables, descriptors, masks, sampled actions, old log probabilities and values in original context order. `tools/coopmwm/trigger_policy.py:257-266,289-332` replays those records. `tools/coopmwm/test_coordination_opportunities_cpu.py:144-161` performs a real optimizer step, verifies head changes and frozen-base preservation, and fails if candidates are regenerated. Ownership and real controller-command effects are exercised at lines 239-252 and 316-346.
- **Event timing and full horizon:** `tools/coopmwm/track2_backend.py:365-390` adds approach decisions without moving the periodic clock; `tools/coopmwm/coordination_opportunities.py:160-174` prevents repeated pair-event churn. Only onset is constrained by observable remaining approach time; hold, global look-ahead, episode windows and full-rollout reward credit are not shortened. Existing delayed-reward and 900-tick fake-physics production tests pass, including propagation from the last tick to the first decision.
- **Registered values:** Geometry, lane identities, association range, sensing range and onset/hold/depth windows come from the existing registrations (`tools/coopmwm/moves.py:63-79`, `tools/coopmwm/variety.py:61-73`). G2 introduces no new tuned distance, TTC, speed, gain or probability threshold. Endpoint canonicalization accepts only exact registered float32 encodings, not an added tolerance band.
- **B/R fairness and A isolation:** Both arms enter the same opt-in implementation with the same candidate generation, event rules and RNG mechanics. Besides the submitted tests, independent deterministic/stochastic 131-tick probes produced identical B/R samples after normalizing only the arm label, and identical RNG states. The A hook retains original policy/controller objects (`tools/coopmwm/track2_backend.py:425-434`).
- **Default-off byte identity:** Independently loaded both changed pre-G2 modules from Git into memory, wired their original proposal implementation together, restored identical policy tensors, and compared them with current code with G2 absent. Serialized samples were byte-equal at every tick in both deterministic and stochastic 131-tick runs; candidate and trigger RNG states also matched. The submitted absent-versus-false and strict checkpoint compatibility tests pass as well.
- **Offered-joint reporting:** `tools/coopmwm/feasible_proposals.py:153-155,194-197` counts actual multi-actor offers; `tools/coopmwm/coordination_opportunities.py:177-188` aggregates the ratio of sums. Tests pass through production update metrics and persisted learning-monitor receipts, including forced-none denominator rows.

## Review constraints

CPU-only review and tests; no SSH, remote execution, SUMO launch or GPU use. No prohibited dataset paths or truth files were opened. No source edits or commits were made; this review file is the only worktree addition.

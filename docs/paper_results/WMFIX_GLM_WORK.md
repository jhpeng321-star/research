# WMFIX independent GLM work packages
Initial packages were written at the design boundary (388a0cd4); updated after CPU completion/WMREQUAL3 delivery to avoid redoing finished work.
MAIN dispatched the ORIGINAL packages (data/objective/verdict/audit) against 388a0cd4; all four finished and were
reviewed by the integrator (tests rerun, none merged — superseded): see `WMFIX_GLM_REVIEW.md`. Status below is binding.
MAIN may dispatch GLM-5.3-flash as slots free. **No further worker dispatch is pending from this session.**
Base: latest committed c4/wmfixmerge; MAIN must inspect branch/status/PIDs/dispatch receipts before creating each worktree.
Every worker: separate branch below, CPU only, no SUMO/GPU training/truth/XML/restricted input/deletion/push; commit and return SHA/tests/blockers.
Read current WMFIX_DESIGN/CPU/REVIEW_REQUEST first; the earlier proposal alone is superseded by matched-only FIT and inferred admission.
Integrator retains all model/core/data-prep/trainer/verdict-metric files and central verdict_plan.json; no worker may edit those.

1. **c4/wmfix-g1 — corrected source-clock contract and exact runtime binding.** STATUS: OPEN, MAIN-side only.
   May touch tools/wmfixmerge/clock_contract.py and tests/unit/test_wmfix_clock.py only.
   Default-OFF `--execute`: compare recorded left rectangles, inherited after-insertion rectangles and the training objective on recorded TRAIN pairs.
   Use the12 FIT seeds/matched arms only. Independently check Q0/due/inserted identity ledgers and corrected scorer source hashes.
   Do not emit source_clock_match merely because WMREQUAL3 exists; quantify any half-tick endpoint difference and amend semantics explicitly.
   Test: `python -m pytest tests/unit/test_wmfix_clock.py -q`; wrong clocks/roles/Q0/counter-as-demand reject; exact matched clocks pass.
   Return HARNESS_CONTRACT evidence or an explicit blocker. MAIN supplies exact frozen corrected runtime closure; no out-of-scope searches.

2. **c4/wmfix-g2 — real TRAIN CPU feasibility and long-horizon teacher ceiling.** STATUS: OPEN, MAIN-side only.
   May touch tools/wmfixmerge/feasibility.py and tests/unit/test_wmfix_real_feasibility.py only.
   Default-OFF real probe via explicit TRAIN input CLI; reuse existing runtime/head/objective without changing losses or thresholds.
   Run registered preparation first through integrator-owned scripts, then native+one-arm60s free rollout/finite differences/optimizer movement.
   Report teacher upstream ceilings by60/120/300s, ledger-only vs inferred-admission/zero-head ablation, separate J/crossing/source gradients.
   No GPU launch based on synthetic evidence, missing data, common-mode loss decrease or near-zero/saturated paired gradients.
   Test: `python -m pytest tests/unit/test_wmfix_real_feasibility.py -q`; synthetic/absent-data/total-loss-only improvements cannot pass.
   Return exact prepared source hashes/paths and CPU receipt, or missing artifacts. Never fit validation, old MAPPO FIT, or SPEEDHEAD.

3. **c4/wmfix-g3 — exact SPEEDHEAD closed-loop producer.** STATUS: TAKEN by integrator (speedhead_panel.py + tests, this
   branch). Do NOT dispatch: duplicate work would be discarded.
   Originally may touch tools/wmfixmerge/speedhead_panel.py, speedhead_binding.json, tests/unit/test_wmfix_speedhead.py only.
   Default-OFF `--execute`: all8 worlds963880–963887, exact t300 observable histories/Q0, K8/mix ON common futures,300/600s prefixes.
   Native lanes for CTRL0/final speed-only MAPPO(d9042cc5)/PI external-U31.29/CIRCLES; all CAVs run the frozen live feedback code.
   Bind frozen source/checkpoint closure from MAIN. No update190 substitution, timeline-only original-slot replay or hand-coded approximation.
   Emit README's wmfix_speedhead/1 schema, complete absolute4-arm and arm−CTRL0 draws, road/source additivity and producer/model bindings.
   Test: `python -m pytest tests/unit/test_wmfix_speedhead.py -q`; wrong export/controller, missing world/draw, Q0 mismatch and zero-call reject.

4. **c4/wmfix-g4 — unchanged C1 zero-shot producer and canonical safety-panel binding.** STATUS: TAKEN by integrator
   (c1_panel.py + tests, this branch). Do NOT dispatch: duplicate work would be discarded.
   Originally may touch tools/wmfixmerge/c1_panel.py, c1_binding.json, tests/unit/test_wmfix_c1.py only.
   Default-OFF `--execute`: exact original C1 PI/unseen-wall cohort/estimand, hash-bound reference/cohort; candidate includes deployed inferred admission.
   Emit README's wmfix_c1_regression/1 schema, per-world errors/blowups, complete canonical world lists and model/config bindings.
   Return existing canonical accept.py TRAIN roots/panel hashes for integrator plan binding. Do not alter acceptance thresholds or canonical accept.py.
   Test: `python -m pytest tests/unit/test_wmfix_c1.py -q`; missing cohort, changed reference/estimand and increased blowups cannot pass.
   If original cohort paths violate the task's path restrictions, stop and ask MAIN for an already-authorized safe artifact; do not hide them behind aliases.

No integration is pre-approved: integrator reads each diff, reruns its exact tests, checks default-OFF/source/ownership parity and commits each merge.

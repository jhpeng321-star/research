# WMFIX independent GLM work packages
Prepared at design boundary; MAIN may dispatch GLM-5.3-flash as slots free. No worker dispatched by this receipt.
Base: c4/wmfixmerge at the design commit (resolve with git log -- docs/paper_results/WMFIX_DESIGN.md).
Each worker uses a separate worktree/branch; no GPU/SUMO/truth/XML/restricted paths, deletion or push. Commit changes and return SHA/tests/blockers.
Integrator owns src/wms/model/merge_response.py, wms.py, spec.py and exact queue plumbing; workers must not edit these.
1. c4/wmfix-g1: data contracts / preparation. May touch tools/wmfixmerge/prepare.py, data.py, tests/unit/test_wmfix_data.py only.
   Default OFF: preparation only via explicit CLI; fit allowlist/design is binding. Hash exact paths and validate identity, clocks, due/inserted/Q0.
   Test: python -m pytest tests/unit/test_wmfix_data.py -q; adversarial wrong-role/seed/identity/Q0/counter-source cases must reject before tape open.
2. c4/wmfix-g2: paired objective and CPU feasibility. May touch src/wms/train/merge_response.py, tools/wmfixmerge/feasibility.py, tests/unit/test_wmfix_objective.py only.
   Default OFF: called only by --enable-merge-response trainer. Full-fleet J incl. source, correct component contrasts, exact+smooth ID crossings.
   Test: python -m pytest tests/unit/test_wmfix_objective.py -q; finite differences, sign-correct optimizer movement, unknown identity rejection.
   Synthetic proof must be labelled; real TRAIN feasibility uses exact manifest only; no claimed pass when unavailable.
3. c4/wmfix-g3: verdict orchestrator. May touch tools/wmfixmerge/verdict.py, verdict_plan.json, tests/unit/test_wmfix_verdict.py only.
   Default OFF: explicit --execute; reuse WMREQUAL3 native-lane K8+mix scorer; SPEEDHEAD4arms; C1+accept; hash-bound per-stage receipts.
   Test: python -m pytest tests/unit/test_wmfix_verdict.py -q; dry-run/duplicate-running/missing-result/zero-call/failed-safety cases block adoption.
   Do not launch or duplicate WMREQUAL3's active scoring job. Concrete missing adapters must be surfaced, never filled with placeholder PASS.
No worker merge is pre-approved: integrator reads diff, reruns exact tests and checks compatibility before cherry-picking/merging, commits each integration.

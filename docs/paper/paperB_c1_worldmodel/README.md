# Paper B — owner drafting package

Task: **S-PAPERB**. Worktree `E:/agent-work/c4-paperb`, branch `c4/paperb`.
Target: IEEE IV, six pages including references; owner deadline November 15, 2026. These are owner-supplied constraints, not independently verified conference rules. <!-- Source: owner S-PAPERB brief, 2026-10-31. -->

## Start here

- `OUTLINE.md`: paragraph claims and page budget, ready for the owner's prose.
- `CLAIMS.md`: supported statements, explicit contradictions, and missing receipts.
- `FIGURES.md`: manuscript slots, captions, data provenance, and exact rendering commands.
- `RELATED_WORK.md`, `refs.bib`: verified literature and defensible positioning.
- `THREATS.md`: reviewer objections and evidence-bounded answers.
- `REFRESH.md`: final source refresh and remaining agent dependencies.

## Evidence rules for this package

All empirical numbers in Markdown have adjacent HTML source comments. Source-relative paths are resolved explicitly; do not resolve sibling reports against this older worktree. CSV rows retain source paths and selectors. Figures visualize existing exports or transcribed published report cells; no new prediction, uncertainty estimation, fitting, or simulation is required.

`SUPPORTED` means supported within the named report's scope, not independently replicated. `PENDING` means the statement cannot yet be asserted. `CONTRADICTED` identifies a claim that must be removed or replaced. A completed package may contain scientific PENDING fields; it is not a completed manuscript or a qualification receipt.

## Initial source audit

- Read `E:/research-dense/docs/paper/{CLAIM_EVIDENCE_CHECKLIST,IEEE_IV_2027_DRAFT}.md`, model plans/change log, named E1/E2 exports and their README.
- Read `E:/agent-work/c4-wmrequal3/docs/paper_results/{WMREQUAL2,WMREQUAL3}.md` and `E:/agent-work/c4-wmfixmerge/docs/paper_results/{WMFIX_PRIOR,WMFIX_RETARGET,WMFIX_PILOT_FAIL,WMFIX_V2_FEAS}.md`.
- Read `E:/agent-work/c4-speedhead/docs/paper_results/SPEEDHEAD.md` and `E:/agent-work/c4-wmcalib/docs/paper_results/{WMCALIB_HANDOFF,WMCALIB_RESUME}.md`.
- Requested `E:/agent-work/c4-c1gap/C1_GAPS.md` is absent. A recovered original wall report exists at `E:/agent-work/c4-c1gap/docs/paper_results/c1gap/WALL_C1_RESULT.recovered.md`, with `recovery.json`; this supersedes the older checklist's secondary-source-only restriction for that report, subject to recovery provenance.
- Requested `E:/agent-work/c4-c1closed/C1CLOSED.md` and final `E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md` are absent at initial inspection. They remain named refresh dependencies.
- Local `src/wms/model/{factory,spec}.py` identifies the in-house black-box contracts. External forecasting papers are context references, not implementations of those baselines.
- No applicable worktree-root or `docs/AGENTS.md` was present. The main repository's scientific constraints were consulted for the shared registration record.

## Central editorial decision

Lead with executable-controller/environment factorization and the question of unseen-controller prediction. Keep historical teacher-forced evidence, strict-history feedback selection, frozen direction accuracy, and corrected timeline qualification visibly distinct. Full closed-loop zero-shot superiority to black-box prediction remains conditional on the C1 agents' receipts. Do not connect results from different checkpoints into a single validated-model narrative.

This task reads existing permitted reports and CSVs only. Its rendering scripts have explicit input allowlists. Process operations, SUMO, GPU inference/training, new runs, and restricted input access are outside this package's reproduction commands.

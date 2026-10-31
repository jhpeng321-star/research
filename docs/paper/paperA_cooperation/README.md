# Paper A — owner drafting package

**Working title:** Do CAVs really cooperate? Measuring and learning cooperation in mixed-traffic merging with exact counterfactual replays.

**Target:** IEEE IV 2027, six pages including references; owner deadline 2026-11-15. These are owner-specified constraints; official submission rules and rendered pagination remain to be checked. <!-- Source: owner S-PAPERA task, 2026-10-31. -->

This is an evidence-linked drafting package. Start with `OUTLINE.md`, resolve claims through `CLAIMS.md`, and use only figures marked GENERATED in `FIGURES.md`. `RELATED_WORK.md` records reference verification and positioning; `refs.bib` contains verified identities. `THREATS.md` gives reviewer objections and bounded answers. `EVIDENCE_SOURCES.md` records the source cut; `REFRESH.md` records the final refresh.

## Current scientific boundary

- The interaction estimand is defined and a historical exact-branch implementation has completed. The new speed-only controller audit and warm-started counterfactual learner have no completed local outcome report at initial inspection. Their efficacy is pending.
- WM-S is a candidate deployable substitute for the expensive oracle. A historical direction panel supports a bounded horizon, but the corrected native-lane contract is NOT QUALIFIED. Avoid an unconditional deployment or short-horizon accuracy claim.
- Keep native release, the forced-command CTRL0 adapter, lane-on historical MAPPO, and the speed-only MAPPO export distinct.

<!-- Sources: E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md (binding amendment); E:/research-dense/docs/paper_results/COOPMEAS2.md; E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md; E:/research-dense/docs/paper_results/E2_direction_horizon.csv. -->

## Package conventions

- Claim status is exactly SUPPORTED, PENDING, or CONTRADICTED. SUPPORTED may describe a scoped negative result or a design contract; it never silently means efficacy.
- Every empirical/design quantity in narrative carries an adjacent HTML source-path comment. Equations, citation identifiers, paragraph labels and editorial page allocations are not measured quantities.
- Missing inserts use `[[PENDING: result / responsible agent]]`. Plans, test passes, allocations and canaries never fill efficacy slots.
- Analysis here uses existing permitted reports/exports only. No simulator, GPU, model inference, or new experimental jobs are required.
- Source reports are external to this worktree. Frozen input copies, plotted-value CSVs and hashes accompany generated plots so the package remains inspectable if a producer later updates its report.

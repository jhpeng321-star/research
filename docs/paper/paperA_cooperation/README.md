# Paper A — owner drafting package

**Working title:** Do CAVs really cooperate? Measuring and learning cooperation in mixed-traffic merging with exact counterfactual replays.

**Target:** IEEE IV 2027, six pages including references; owner deadline 2026-11-15. These are owner-specified constraints; official submission rules and rendered pagination remain to be checked. <!-- Source: owner S-PAPERA task, 2026-10-31. -->

This is an evidence-linked drafting package. Start with `OUTLINE.md`, resolve claims through `CLAIMS.md`, and use only figures marked GENERATED in `FIGURES.md`. `RELATED_WORK.md` records reference verification and positioning; `refs.bib` contains verified identities. `THREATS.md` gives reviewer objections and bounded answers. `EVIDENCE_SOURCES.md` records the source cut; `REFRESH.md` records the final refresh.

## Current scientific boundary

- **Primary method:** WM-trained cooperative policy using short imagination plus a terminal cost-to-go trained on real SUMO outcomes. The WMVALUE frozen offline design is present; [[PENDING: decisive offline verdict / S-WMVALUE]], then [[PENDING: separately registered policy run and exact cooperation/efficacy verification / MAIN WM policy producer, S-COOPMEASURE, S-COOPEVAL]].
- **Backup/reference:** SUMO-oracle cooperative learner. Later zero-noise saved-state branches failed despite identical commands, invalidating the original SNR/interaction values. No optimizer update completed. Corrected full-prefix rerun gates/probe and the audit remain on artifact HOLD. Earlier passing canaries apply only to their checked states.
- The interaction estimand and historical measurement are supported. Track 3B is now complete with no accepted improvement. Descriptive behavior counts have landed; probe-based positive-credit discovery cannot validate cooperation after the artifact failure.
- Historical WM direction fidelity was bounded by horizon; corrected native-lane qualification still fails. The new value-tail route is a hypothesis to test, not a validation already supplied by those older results.
- Keep native release, the forced-command CTRL0 adapter, lane-on historical MAPPO, and the speed-only MAPPO export distinct.

<!-- Sources: E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md (binding amendment); E:/research-dense/docs/paper_results/COOPMEAS2.md; E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md; E:/research-dense/docs/paper_results/E2_direction_horizon.csv. -->
<!-- Refresh sources: E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md; E:/agent-work/c4-sumocf/tools/sumocf/evidence/CANARY_V2.json. -->
<!-- Current authority/evidence: owner MAIN ADDENDUM; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, OWNER FRAMING; E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md; E:/agent-work/c4-sumoes/docs/paper_results/SUMOES_RUN.md; E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md. -->

## Package conventions

- Claim status is exactly SUPPORTED, PENDING, or CONTRADICTED. SUPPORTED may describe a scoped negative result or a design contract; it never silently means efficacy.
- Every empirical/design quantity in narrative carries an adjacent HTML source-path comment. Equations, citation identifiers, paragraph labels and editorial page allocations are not measured quantities.
- Missing inserts use `[[PENDING: result / responsible agent]]`. Plans, test passes, allocations and canaries never fill efficacy slots.
- Analysis here uses existing permitted reports/exports only. No simulator, GPU, model inference, or new experimental jobs are required.
- Source reports are external to this worktree. Frozen input copies, plotted-value CSVs and hashes accompany generated plots so the package remains inspectable if a producer later updates its report.

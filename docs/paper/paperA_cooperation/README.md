# Paper A — owner drafting package

**Working title:** Do CAVs really cooperate? Measuring and learning cooperation in mixed-traffic merging with exact counterfactual replays.

**Target:** IEEE IV 2027, six pages including references; owner deadline 2026-11-15. These are owner-specified constraints; official submission rules and rendered pagination remain to be checked. <!-- Source: owner S-PAPERA task, 2026-10-31. -->

This is an evidence-linked drafting package. Start with `OUTLINE.md`, resolve claims through `CLAIMS.md`, and use only figures marked GENERATED in `FIGURES.md`. `RELATED_WORK.md` records reference verification and positioning; `refs.bib` contains verified identities. `THREATS.md` gives reviewer objections and bounded answers. `EVIDENCE_SOURCES.md` records the source cut; `REFRESH.md` records the final refresh.

## Current scientific boundary

- **Primary method:** WM-trained cooperative policy using short imagination plus a real-SUMO-trained terminal value. WMVALUE's registered causal reconstruction, real fit and adapter gates are complete; frozen decisive inference is **RUNNING**, all predictive metrics NA. [[PENDING: complete once-only offline comparison / S-WMVALUE existing finalizer]], then [[PENDING: separately registered policy and exact verification / MAIN WM policy producer, S-COOPMEASURE, S-COOPEVAL]]. <!-- Source: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md. -->
- **Backup/reference: FINAL STOPPED without useful deployed cooperation or a primary MAPPO win.** Corrected full-prefix reruns fixed the later saved-state/held-action artifacts; original credits stay invalid. Completed backup DEV checks worsen versus MAPPO at the primary clock, with horizon ranking flips. This is not a cooperative teacher/upper-bound gain; gain-retention ratio is **NOT ESTIMABLE**. The audit's own HOLD/release migration remains unresolved. <!-- Sources: E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_RUN.md:3-88; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md; E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md. -->
- The interaction definition and historical measurement are supported. Track 3B has no accepted improvement; WMCALIB is NOT USABLE, WMQUEUE KILLED. Corrected paired behavior reports have landed without establishing a beneficial deployed mechanism; original-probe positive-credit discovery stays withdrawn. <!-- Sources: E:/agent-work/c4-sumoes/docs/paper_results/SUMOES_RUN.md; E:/agent-work/c4-wmcalib/docs/paper_results/WMCALIB.md; E:/agent-work/c4-wmqueue/docs/paper_results/WMQUEUE.md; E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md. -->
- Historical WM direction fidelity was bounded by horizon; corrected native-lane qualification still fails. The new value-tail route is a hypothesis to test, not a validation already supplied by those older results.
- Keep native release, the forced-command CTRL0 adapter, lane-on historical MAPPO, and the speed-only MAPPO export distinct.
- Paper B's recovered wall CIs, conditional full-history PI advantage and controller-winner headroom remain scoped prediction/context evidence. Tiny-pulse sensitivity does not explain WM error as inevitable chaos. C1CLOSED is incomplete; no partial performance headline. See `CLAIMS.md` B-TRANSFER/B-FLOOR/B-SWITCH/B-CLOSED and `FIGURES.md` routing. <!-- Sources: E:/agent-work/c4-c1gap/docs/paper_results/C1_GAPS.md; E:/agent-work/c4-c1pidiag/docs/paper_results/C1PIDIAG.md; E:/agent-work/c4-c1floor/docs/paper_results/C1FLOOR.md; E:/agent-work/c4-c1switch/docs/paper_results/C1SWITCH_WINNERS.md; E:/agent-work/c4-c1closed/docs/paper_results/C1CLOSED.md. -->

<!-- Sources: E:/agent-work/c4-coopmeasure/AUDIT_DESIGN.md; E:/agent-work/c4-sumocf/docs/paper_results/SUMOCF_DESIGN.md (binding amendment); E:/research-dense/docs/paper_results/COOPMEAS2.md; E:/agent-work/c4-wmrequal3/docs/paper_results/WMREQUAL3.md; E:/research-dense/docs/paper_results/E2_direction_horizon.csv. -->
<!-- Refresh sources: E:/agent-work/c4-coopmeasure/docs/paper_results/COOPAUDIT.md; E:/agent-work/c4-sumocf/tools/sumocf/evidence/CANARY_V2.json. -->
<!-- Current authority/evidence: owner MAIN ADDENDUM; E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md, OWNER FRAMING; E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE_DESIGN.md; E:/agent-work/c4-sumocf/docs/paper_results/ARTIFACT_CHECK.md; E:/agent-work/c4-sumoes/docs/paper_results/SUMOES_RUN.md; E:/agent-work/c4-coopbehav/docs/paper_results/COOPBEHAV.md. -->
<!-- WMVALUE verdict: E:/agent-work/c4-wmvalue/docs/paper_results/WMVALUE.md. -->

## Package conventions

- Claim status is exactly SUPPORTED, PENDING, or CONTRADICTED. SUPPORTED may describe a scoped negative result or a design contract; it never silently means efficacy.
- Every empirical/design quantity in narrative carries an adjacent HTML source-path comment. Equations, citation identifiers, paragraph labels and editorial page allocations are not measured quantities.
- Missing inserts use `[[PENDING: result / responsible agent]]`. Plans, test passes, allocations and canaries never fill efficacy slots.
- Analysis here uses existing permitted reports/exports only. No simulator, GPU, model inference, or new experimental jobs are required.
- Source reports are external to this worktree. Frozen input copies, plotted-value CSVs and hashes accompany generated plots so the package remains inspectable if a producer later updates its report.

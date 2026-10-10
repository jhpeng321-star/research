# Research archive before CIRCLES adoption

This archive preserves historical implementations, all raw research results, checkpoints, reports and the dirty worktree as they existed at the adoption boundary. It is evidence, not instructions to resume training or open sealed partitions.

The active implementation is `traffic_shepherd/` with the permanent frozen baseline in `baselines/circles/`. Consult the repository README for supported commands.

## Layout and identity

- `code/`: original package layout, historical tests, root launchers, and original setup/configuration files.
- `docs/`: byte-identical historical reports and superseded directions.
- `configs/scenarios/`: historical local SUMO scenarios; shared assets outside this workspace were untouched.
- `results/runs/`: raw evaluation/probe/training/log/checkpoint families, moved once without duplicating large data.
- `reports/`: manuscript, slides, plots, PDFs, and user-facing historical outputs.
- [manifest.json](manifest.json): every original path, current physical path, size, SHA256, family classification and relocation status.
- [path_dependencies.json](path_dependencies.json): concrete frozen/source/document references that motivated compatibility aliases.
- [git_status_before_moves.txt](git_status_before_moves.txt) and [dirty tracked patch](dirty_tracked_changes_before_moves.patch): pre-move tracked and untracked state.

Inventory: **59,846 files**, **21,446,210,494 bytes**. Acceptance/locked/ambiguous outcome files were only hashed as opaque bytes; none were parsed to classify performance.

## Historical questions and conclusions

### Released CIRCLES model provenance and transfer

**Promising** — Constrained adapter failed; direct p20 reduces braking with service/TTC trade-offs. Working adoption is not scientific acceptance.

Controller/configuration: Externally released frozen ONNX; raw SI singleton absolute-acceleration adapter. 3 indexed top-level artifact paths.

Reports: [PUBLISHED_DAMPING_REUSE_AUDIT_2026-09-06.md](docs/PUBLISHED_DAMPING_REUSE_AUDIT_2026-09-06.md), [PUBLISHED_IDM_FIT_CHECK_2026-09-06.md](docs/PUBLISHED_IDM_FIT_CHECK_2026-09-06.md).

### Recovery-reference mechanism after native-selectable failure

**Failed** — Rebound removed, pulse effort +18.94%; 480-second budget exhausted; no successor authorized.

Controller/configuration: v2.1 fixed prototype; no training. 1 indexed top-level artifact paths.

Reports: [NATIVE_DAMPING_RECOVERY_V21_RESULTS_2026-09-06.md](docs/NATIVE_DAMPING_RECOVERY_V21_RESULTS_2026-09-06.md).

### Explicit native action, bounded intervention preference and learned damping

**Failed** — Release/re-engagement and reward ordering verified; learned corridor efficacy failed. Handback speed-equality statistic was corrected, with a later paired physical check.

Controller/configuration: v2 native-selectable policy and SUMO actuator. 4 indexed top-level artifact paths.

Reports: [NATIVE_SELECTABLE_DAMPING_V2_RESULTS.md](docs/NATIVE_SELECTABLE_DAMPING_V2_RESULTS.md).

### Train standalone local IDM residual damping and repair team credit

**Failed** — Campaign checkpoint-paused at 36/8/8. Episode-20 efficacy fails and microunit corrections still engage a distinct control law. No accepted checkpoint.

Controller/configuration: PyTorch PPO native damping; analytical IDM/residual command path. 11 indexed top-level artifact paths.

Reports: [NATIVE_DAMPING_O71_E020_DEVELOPMENT_RESULTS_2026-09-05.md](docs/NATIVE_DAMPING_O71_E020_DEVELOPMENT_RESULTS_2026-09-05.md), [NATIVE_DAMPING_MAGNITUDE_ABLATION_RESULTS_2026-09-05.md](docs/NATIVE_DAMPING_MAGNITUDE_ABLATION_RESULTS_2026-09-05.md).

### Frozen sustained-wall escalation generation

**Quarantined** — Stopped historical direction; controller/observation/horizon assumptions do not establish adoption-era results. Preserve all rows and seed histories.

Controller/configuration: Historical stage-2 damping/wall/predictor contract. 2 indexed top-level artifact paths.

Reports: [STAGE2_GENERATION_RESULTS.md](docs/STAGE2_GENERATION_RESULTS.md), [STAGE2_NEXT_AGENT_PROMPT.md](docs/STAGE2_NEXT_AGENT_PROMPT.md).

### Wall release, hold duration, onset timing and fair threshold comparisons

**Inconclusive** — Development release/timing evidence; no current accepted wall or predictor follows from these arms.

Controller/configuration: Historical sustained wall and release variants. 23 indexed top-level artifact paths.

Reports: [STAGE1C2_RESULTS.md](docs/STAGE1C2_RESULTS.md), [SUSTAINED_WALL_V1_STAGE1C_RELEASE_RESULTS.md](docs/SUSTAINED_WALL_V1_STAGE1C_RELEASE_RESULTS.md).

### Map transition band and fit CAV-report gate

**Inconclusive** — Useful generation/reporting infrastructure and development fit; not current baseline validation or a prospective escalation claim.

Controller/configuration: Historical sustained-wall preparation/generation and gate. 39 indexed top-level artifact paths.

Reports: [SUSTAINED_WALL_V1_STAGE1B_GENERATION_RESULTS.md](docs/SUSTAINED_WALL_V1_STAGE1B_GENERATION_RESULTS.md), [SUSTAINED_WALL_V1_GATE_V1_FIT_RESULTS.md](docs/SUSTAINED_WALL_V1_GATE_V1_FIT_RESULTS.md).

### Establish sustained-wall mechanics and development outcomes

**Inconclusive** — Pre-adoption treatment development only; use matching report before quoting any result.

Controller/configuration: Historical sustained wall. 1 indexed top-level artifact paths.

Reports: [SUSTAINED_WALL_V1_STAGE1A_RESULTS.md](docs/SUSTAINED_WALL_V1_STAGE1A_RESULTS.md).

### Wall feasibility, delivered dose, safety, headroom and observability

**Failed** — Several local mechanics/availability gates passed, but accepted safe effective escalation was not established. Do not pool incompatible treatment variants or resume compact-wall search.

Controller/configuration: Historical IDM wall v5.x variants and two-stage/shadow gates. 35 indexed top-level artifact paths.

Reports: [IDM_V5_4_SHADOW_OBSERVATION_V2_RESULTS.md](docs/IDM_V5_4_SHADOW_OBSERVATION_V2_RESULTS.md), [IDM_V5_5_T880_GATE_F_AND_WALL_PILOT_RESULTS.md](docs/IDM_V5_5_T880_GATE_F_AND_WALL_PILOT_RESULTS.md), [IDM_V5_6_2_PHASE_SPLIT_AUTHORITY_NOGO_DIAGNOSIS.md](docs/IDM_V5_6_2_PHASE_SPLIT_AUTHORITY_NOGO_DIAGNOSIS.md).

### Separate wall feasibility from incremental benefit and train historical gates

**Quarantined** — Older dose/branch identity and observation contracts differ. Locked outcomes remain sealed during inventory. No pooling into current predictor work.

Controller/configuration: Paired no-wall/wall counterfactual rows; multiple historical labels and policies. 100 indexed top-level artifact paths.

Reports: [COUNTERFACTUAL_TWO_HEAD_LOCAL_V2_FINAL_RESULTS.md](docs/COUNTERFACTUAL_TWO_HEAD_LOCAL_V2_FINAL_RESULTS.md), [LEGACY_RESEARCH_RECORD.md](docs/LEGACY_RESEARCH_RECORD.md).

### Local activation observation sufficiency, authority and P30/P50 efficacy

**Failed** — Useful observation/authority/accounting diagnostics; fresh efficacy and/or safety/headroom gates failed. Analytical activation is not a trained damping checkpoint.

Controller/configuration: Fixed analytical P30/P50 and modified FollowerStopper/feedback variants. 138 indexed top-level artifact paths.

Reports: [IDM_DECENTRALIZED_P30_FRESH_VALIDATION_REPORT.md](docs/IDM_DECENTRALIZED_P30_FRESH_VALIDATION_REPORT.md), [IDM_DECENTRALIZED_P50_DEVELOPMENT_SCREEN_RESULTS.md](docs/IDM_DECENTRALIZED_P50_DEVELOPMENT_SCREEN_RESULTS.md), [IDM_MODIFIED_FOLLOWERSTOPPER_P30_VALIDATION_RESULTS.md](docs/IDM_MODIFIED_FOLLOWERSTOPPER_P30_VALIDATION_RESULTS.md).

### Option-level reward, local context and treatment timing

**Inconclusive** — Option-credit primitives and selected audits are useful, but no accepted timing/controller transfer is implied. Follow each report's explicit go/no-go.

Controller/configuration: Historical binary/paired options and timing candidates. 27 indexed top-level artifact paths.

Reports: [IDM_BINARY_OPTION_REWARD_RESULTS.md](docs/IDM_BINARY_OPTION_REWARD_RESULTS.md), [IDM_RL_TIMING_CANDIDATE_NOGO_DIAGNOSIS.md](docs/IDM_RL_TIMING_CANDIDATE_NOGO_DIAGNOSIS.md), [IDM_TIMING_OBSERVATION_VALUE_SCREEN_RESULTS.md](docs/IDM_TIMING_OBSERVATION_VALUE_SCREEN_RESULTS.md).

### IDM analytical/residual foundations, reward, action and observation gates

**Inconclusive** — Useful infrastructure mixed with failed training/readiness gates. No accepted trained IDM checkpoint; exact family outcome requires its linked development report.

Controller/configuration: Historical IDM analytical/RLlib treatments and wall pilots. 114 indexed top-level artifact paths.

Reports: [IDM_PHASE5_ENDPOINT_ALIGNED_READINESS.md](docs/IDM_PHASE5_ENDPOINT_ALIGNED_READINESS.md), [IDM_DECENTRALIZED_CLEAN_SMOKE_V3_DIAGNOSIS.md](docs/IDM_DECENTRALIZED_CLEAN_SMOKE_V3_DIAGNOSIS.md), [ARTIFACT_MANIFEST.md](docs/ARTIFACT_MANIFEST.md).

### Earlier discrete damping and wall-supervisor optimization

**Quarantined** — Historical checkpoint/performance evidence only. Do not substitute this pipeline for the adopted direct-output baseline or resume medium optimization.

Controller/configuration: Legacy Krauss/medium or mixed IDM transfer configurations; inspect frozen config for exact domain. 193 indexed top-level artifact paths.

Reports: [LEGACY_RESEARCH_RECORD.md](docs/LEGACY_RESEARCH_RECORD.md), [RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md).

### Historical diagnostics, references, autopsies and campaign utilities

**Inconclusive** — Preserved for provenance. No current-paper validity inferred; ambiguous per-family scope is explicitly unresolved rather than invented.

Controller/configuration: Exact retained artifact/configuration; not inferred from filename. 1078 indexed top-level artifact paths.

Reports: [LEGACY_RESEARCH_RECORD.md](docs/LEGACY_RESEARCH_RECORD.md), [RESEARCH_OVERVIEW.md](docs/RESEARCH_OVERVIEW.md).

## Discover a particular experiment

Every `runs/<category>/<family>` has its own manifest entry, including raw/checkpoint paths, principal question, conservative outcome classification, report links, filename-derived seed/penetration/demand tokens and exact dependency references. Filename tokens are navigation aids, not a complete or validated exclusion registry. Consult the frozen protocol/configuration before using a scientific scope; do not infer a missing outcome from a directory name.

The per-family outcome is deliberately conservative: mixed or insufficiently reconciled families are `inconclusive`; legacy or incompatible datasets are `quarantined`. These labels do not erase specific positive findings in the unchanged reports. Negative bespoke v1/v2/v2.1 results remain explicit.

## Frozen-reader exceptions

Historical original `runs/` and scenario locations may be NTFS junctions to the one physical archive copy. The manifest enumerates all 19 aliases and targets, including the local policy-runner input fixture. These exist solely because frozen records and readers store original paths; they are not active output directories. New active work uses `runs/active/`. Archive-local `code/runs`, `code/docs`, `code/scenarios`, and `code/output` restore source-relative historical reads; archive-local `runs`/`output` preserve report links. No active controller imports archived learners.

## Historical regression suite

Original tests were archived with their implementation, without weakening assertions. The pre-change full suite passed **1146 tests**. The supported historical discovery/run path is:

```powershell
Set-Location D:\research-dense
python archive/pre_circles/run_regressions.py
```

This command copies the small historical source/configuration trees to a disposable workspace under `output/adoption_cleanup/archived_regression_workspaces/`, isolates writable scenarios, caches, locks and output, and leaves the large raw data in their single archived copy. It runs regression tests only; historical training/experiment launchers are preserved for provenance and are not authorized to execute. Final archive-suite verification and any residual failures are reported in the active cleanup handoff. Original absolute-path assumptions remain in frozen source; exact byte preservation is intentional. Do not silently rewrite frozen manifests to repair them.

## Verified regression and preservation result

The unchanged archived suite passed **1146 tests** after resolving a relative-path fixture issue. The first direct archived run's three startup failures remain in `output/adoption_cleanup/archived_pytest.log`; the successful rerun is `archived_pytest_relocated.log`. Five exact local shared-input copies (5,053 bytes) let the three legacy policy-runner tests write locally instead of into shared assets. SUMO statistics record 37 seconds each, including 7 seconds of warmup, or 111 simulated seconds total for those three tests. Other existing SUMO regressions also ran; this was not a training or penetration campaign.

The original extended-zone builder test rewrote one generated scenario manifest during the direct suite. Its altered output is preserved, and the original was restored only after its SHA256 exactly matched the pre-move inventory (`archive_fixture_restoration.json`). **Do not run pytest directly against the frozen archive.** The supported disposable-workspace runner isolates that write; its two unchanged builder tests passed with no archived input changes. No legacy assertions or source files were weakened. Full-file identity verification plus the exact one-file restoration are recorded in `output/adoption_cleanup/archive_verification.json`; the initial full-scan failure is preserved separately.

Orphaned legacy root bytecode is preserved separately under `transient/root_pycache/`, with its own [cache identity manifest](transient/cache_manifest.json). These reproducible cache files are outside the 59,846-file scientific inventory; they were moved rather than deleted.

# Proposed markdown deletions — 2026-10-09

Owner request (2026-10-09): delete markdown that is marked outdated and no longer
useful. **Nothing was deleted, moved or renamed by this task.** This is a proposal
only. Each entry below is a `.md` file that is clearly outdated, has **no inbound
references anywhere in scope**, is **not evidence** (no run receipt, registration,
ledger-referenced report, frozen manifest or handoff/morning summary), and carries a
low-risk rationale.

Context and per-file status: [DOCS_AUDIT_20261009.md](DOCS_AUDIT_20261009.md).
Current direction: [ROUTE.md](ROUTE.md), [WM_S2_ROADMAP.md](WM_S2_ROADMAP.md).

## Proposed for deletion

### 1. `docs/IMPROVEMENT_OPTIONS_20261006.md`
- **Git-tracked:** no (untracked working-tree file).
- **Why outdated:** a one-time "owner picks which to try" menu dated 2026-10-06; its
  items are now resolved or superseded by the roadmap's revised sequence and by
  `docs/DESIGN_DISCUSSION_20261008.md`. `docs/WM_S2_ROADMAP.md` already states the
  current plan ("Phase A - WM long-horizon fix", "direct output as the MAIN training
  data") and the DONE/READY/BUILD/DATA statuses in this menu are stale.
- **Inbound links that would break:** none found (no in-scope file references it).
  Fix applied in step 2: none needed; it is listed only under INDEX's superseded
  section, which does not require the file to exist.
- **Risk:** low. It is a proposal menu, not a registration or result.
- **Superseded by:** `docs/WM_S2_ROADMAP.md` (UPDATE 2026-10-09),
  `docs/DESIGN_DISCUSSION_20261008.md`.

### 2. `docs/REDESIGN_V3_ZCODE_PLAN.md`
- **Git-tracked:** no (untracked).
- **Why outdated:** an execution plan for a one-night **ZCode GLM-5.3** refactor into
  a `src/rd/` package. The current direction is "Executors: **OpenCode only** (no
  ZCode)" (owner 2026-10-01/2026-10-09), and the refactor is not the active route. The
  plan also embeds now-retired systems (FollowerStopper; "don't resurrect them").
- **Inbound links that would break:** none found (only a self-reference at line 13).
- **Risk:** low. The refactor proposal branch (if any) is a git worktree, not this
  file; no ledger or manifest cites it.
- **Superseded by:** current OpenCode-only execution routing and the WM-S route.

### 3. `docs/REORG_PLAN.md`
- **Git-tracked:** no (untracked).
- **Why outdated:** the "one root, organised tree, deletion candidates" plan's phase 3
  is recorded as already executed in its own "Executed 2026-10-01" table, and phase 2
  (moves) remains gated and unexecuted. The owner's current storage rule is "D:
  staging retired (E: only)"; this historical migration plan no longer governs work.
- **Inbound links that would break:** none found.
- **Risk:** low-to-medium. The plan is a record of an executed cleanup; if the owner
  wants to retain that operational history, keep it. The move map itself lives in
  `E:/rd-wt/reorg2_architect/MOVE_MAP.json` and the phase-3 manifest in
  `E:/research-dense-backups/reorg_phase3_20261001/MANIFEST.json`, so deletion loses
  only the prose plan.
- **Superseded by:** completed execution + the current E:-only storage policy.

### 4. `docs/WORLD_MODEL_DECISION_PROTOCOL.md`
- **Git-tracked:** yes (tracked).
- **Why outdated:** a "stage-4 rehearsal (v2 design contract)" dated 2026-09-21 that
  binds the old WM-N1 ensemble/outcome-head benefit-evaluation design (Frames A/B/C,
  five-cost vector, ensemble spread band). `docs/WM_STATE_PLAN.md` states WM-S
  "Supersedes the effect-head (delay-head) direction of WM-N1 A8-A13 as the primary
  approach", and the owner's direction retires the WM-N1 delay head and makes the
  WM a pure next-state predictor.
- **Inbound links that would break:** none found.
- **Risk:** low-to-medium. It is a design contract, not a registration or result; if
  any future historical replay needs its frame definitions, keep it. No ledger entry,
  freeze file or manifest cites it.
- **Superseded by:** `docs/WM_STATE_PLAN.md`, `docs/WM_S2_ROADMAP.md`.

### 5. `docs/presentation_20261005/OUTLINE_v1.md`
- **Git-tracked:** no (untracked).
- **Why outdated:** the first draft of the 2026-10-05 talk outline; superseded by
  `docs/presentation_20261005/OUTLINE.md` (later revision) and the short variants.
- **Inbound links that would break:** none found (the `.py` deck builders reference
  `OUTLINE.md` and the short outline, not `OUTLINE_v1.md`).
- **Risk:** low. A superseded draft with no references.

## Considered but deliberately NOT proposed

| Path | Why held |
|---|---|
| `docs/STATUS.md` | "Current status — 2026-09-08" is stale, but the frozen `baselines/circles/adoption_manifest.json` and `docs/PROJECT_LOG.md` reference it; deleting would break a frozen manifest entry. |
| `docs/EXPERIMENT_LEDGER.md` | Secondary pre-WM index, superseded by the master ledger, but `protocols/experiment_history.json` names it. |
| `docs/NEXT_EXPERIMENT_PLAN.md` | Wall-only plan, superseded, but `protocols/experiment_history.json` names it. |
| `docs/OBSERVATION_DECISION_CONTRACT.md` | Closed wall entry/continuation contract, but `protocols/experiment_history.json` names it. |
| `docs/GATE_FEATURE_CONTRACT.md` | v1 predecessor of the v2 contract; explicitly retained as v1, and INDEX/OBSERVATION_DECISION_CONTRACT link to it. |
| `docs/CLOUD_OFFLOAD_PLAN.md`, `docs/GITHUB_PLAN.md` | Superseded plans, but `docs/HANDOFF_20261002.md` (a kept handoff) links to them and is outside this task's edit scope. |
| `docs/presentation_20261005/OUTLINE_progress.md`, `OUTLINE_short.md` | Superseded drafts, but referenced by non-`.md` deck-build scripts this task may not edit. |
| `docs/HANDOFF_20261002.md`, `docs/MORNING_SUMMARY_20261005.md` | Handoff/morning-summary class the owner asked to keep. |
| `docs/ARCHITECTURE.md`, `docs/WORLD_MODEL_RESEARCH_GOAL.md`, `docs/RESULTS_SUMMARY.md` | Living/reference docs needing a WM-S refresh, not deletion. |

## Files modified by this task (working tree; nothing committed)

- `docs/DOCS_AUDIT_20261009.md` (new)
- `docs/DOCS_DELETE_PROPOSAL.md` (new)
- `README.md` (status section + header block refreshed for WM-S)
- `docs/INDEX.md` (current-documents and superseded sections added)
- `docs/ROUTE.md` (current WM-S route block added)
- `docs/OPEN_QUESTIONS.md` (WM-S open questions added)
- `docs/WM_S2_ROADMAP.md` (2026-10-09 status lines)
- `AGENTS.md` (active-owner-goal header only)

No file was deleted or moved; `git status` still shows every proposed file present.

## Executed 2026-10-09

Owner approved all 5. Backed up first to `gdrive:research-archive/docs-deleted-20261009` (rclone check: 5 matching, 0 differences; local copy `tmp/docs_deleted_20261009/`), then deleted (`git rm` for the tracked WORLD_MODEL_DECISION_PROTOCOL.md; plain removal for the 4 untracked files). INDEX rows updated.

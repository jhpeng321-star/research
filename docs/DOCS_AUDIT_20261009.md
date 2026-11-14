# Markdown documentation audit — 2026-10-09

Owner request (2026-10-09): make all markdown files reflect the changes we tried and
the direction we are working, and identify the ones that are outdated and no longer
useful. This is a docs-only audit. No file was deleted, moved or renamed; deletion
candidates are listed in [DOCS_DELETE_PROPOSAL.md](DOCS_DELETE_PROPOSAL.md).

## Current direction (source of truth for this audit)

The project is **WM-S**: a physics-informed per-step traffic world model (IDM/MOBIL
core + gated learned residual) used to pick controllers per situation
([WM_STATE_PLAN.md](WM_STATE_PLAN.md), [WM_S2_ROADMAP.md](WM_S2_ROADMAP.md)).

- **C1** zero-shot prediction of unseen controllers: verified on the PI arm, and on
  2026-10-09 on the unseen **WALL** (structured 8.71 m vs bb_traj 25.41 / bb_id
  40.07 m HDV median |ds| at 60 s on wall_same; verdict SUPPORTED with caveats;
  `E:/rd-wt/ws-wallc1/_out/WALL_C1_RESULT.md`).
- **C2** per-driver identification; **C3** fragility (deferred); **C4** learned
  cooperative multi-CAV control trained Dreamer-style inside the WM (later).
- This week: make the WM ready and the code set up for the next steps; direct output
  / per-CAV switching data is the main training data; RL later.
- Recent work: DIAG120 found under-braking / missed jams cause the 120 s blow-ups;
  braking-fix and no-braking models trained 2026-10-09
  ([REGISTRATION.md](../runs/active/WM-S-MAIN2-20261009/REGISTRATION.md)), scoring in
  progress (`E:/rd-wt/ws-main2score`); pack_v3 contributed **0** training windows
  (fix in progress `E:/rd-wt/ws-packv3fix`); entries-ON rescoring in progress
  (`E:/rd-wt/ws-entrieseval`); black-box retraining dropped per owner rule (compare
  new models only with the previous C1 models); both GPUs shut down.
- Executors: OpenCode only (no ZCode).
- **Retired:** FollowerStopper, per-lane mix layer, wall-only planning, WM-N1 delay
  head, D: staging (E: only), self-recovery/pushforward training (owner vetoed).
- Next: one GPU run with loss rebalancing; walls are one action; **regret** is the
  primary planner score.

## Scope and method

- Inventoried every `.md` under the repo root and `docs/` (including
  `docs/presentation_20261005/`). `protocols/`, `tools/` and `tests/` contain no
  `.md` files. Root agent-config markdown (`.claude/`, `.cursor/`) is out of scope.
- Excluded: `archive/`, `sealed/`, `runs/`, `output/`, `assets/`, `tmp/`,
  `node_modules/`, `.git/`, and any `CAL` / `HOLDOUT` / `chunk-5` / `J4B` / `INC1`
  path. None of those contents were read.
- Dates are the file modification times; git status is from
  `git ls-files --error-unmatch`.
- Inbound links are in-scope textual references to the file (by basename) in the
  root/docs/protocols/tools/tests trees; ledger/manifest/tool references are noted
  because they constrain deletion.
- Status: **CURRENT** | **NEEDS-UPDATE** | **SUPERSEDED (by …)** | **OUTDATED-MARKED**
  (the file itself, or INDEX, says outdated/superseded/retired) | **DELETE-PROPOSED**.

## Inventory

| Path | Purpose | Last-updated | Git | Inbound links (in scope) | Status |
|---|---|---|---|---|---|
| `AGENTS.md` | Repo working rules, constraints, close-out requirements, active owner goal header | 2026-09-25 | tracked | README, HANDOFF, EXPERIMENT_RECORD_TEMPLATE, REORG_PLAN, REDESIGN_V3, ledger, code/tests | NEEDS-UPDATE (goal header edited; rest retained) |
| `README.md` | Project entry point | 2026-10-04 | tracked | AGENTS, INDEX, ROUTE, PROJECT_LOG, GITHUB_PLAN, REDESIGN_V3, ledger, code/tests | NEEDS-UPDATE (status updated) |
| `docs/ARCHITECTURE.md` | Controller/ownership/wall architecture | 2026-09-25 | tracked | INDEX, PROJECT_LOG, REDESIGN_V3, adoption manifest | NEEDS-UPDATE (no WM-S layer; out of this task's edit scope) |
| `docs/CLOUD_OFFLOAD_PLAN.md` | Google-Drive offload plan + receipts | 2026-10-02 | untracked | HANDOFF_20261002 | SUPERSEDED (offload executed; D: staging retired) — not proposed (HANDOFF inbound) |
| `docs/DESIGN_DISCUSSION_20261008.md` | This week's design/number brief (EFFH-02, DIAG120, CTM, RANKING) | 2026-10-08 | untracked | WM_S2_ROADMAP | CURRENT |
| `docs/EXPERIMENT_HISTORY.md` | Curated experiment history | 2026-10-04 | tracked | AGENTS, INDEX, OPEN_QUESTIONS, RESULTS_SUMMARY, PROJECT_LOG, ROUTE, template, tools | CURRENT |
| `docs/EXPERIMENT_LEDGER.md` | "Secondary index" of pre-WM blocks | 2026-09-08 | tracked | EXPERIMENT_HISTORY, INDEX, ROUTE, ledger | SUPERSEDED (by `protocols/experiment_history.json` + EXPERIMENT_HISTORY) — not proposed (ledger-referenced) |
| `docs/EXPERIMENT_RECORD_TEMPLATE.md` | Registration/close-out template | 2026-09-07 | tracked | AGENTS, EXPERIMENT_HISTORY, INDEX, OPEN_QUESTIONS, PROJECT_LOG | CURRENT |
| `docs/GATE_FEATURE_CONTRACT.md` | v1 entry feature/decision contract | 2026-09-06 | tracked | INDEX, OBSERVATION_DECISION_CONTRACT | SUPERSEDED (v2) but retained as v1 — not proposed (inbound) |
| `docs/GITHUB_PLAN.md` | Proposed private GitHub export plan | 2026-10-02 | untracked | HANDOFF_20261002 | SUPERSEDED/OUTDATED (never executed; references WM-N1/wd01 branches) — not proposed (HANDOFF inbound) |
| `docs/HANDOFF_20261002.md` | Coordinator handoff, WM-N1 recovery | 2026-10-02 | untracked | README, INDEX, ROUTE, ledger | SUPERSEDED but KEPT (handoff referenced by INDEX/memory) |
| `docs/IMPROVEMENT_OPTIONS_20261006.md` | WM-S option menu (A–F) | 2026-10-06 | untracked | — | SUPERSEDED (by WM_S2_ROADMAP + DESIGN_DISCUSSION) — DELETE-PROPOSED |
| `docs/INDEX.md` | Master status index (append log) | 2026-10-04 | tracked | AGENTS, README, ROUTE, PROJECT_LOG, ledger, tests | NEEDS-UPDATE (current + superseded sections added) |
| `docs/LITCHECK_MICRO_MACRO_20261008.md` | Lit check: micro–macro coupling | 2026-10-08 | untracked | DESIGN_DISCUSSION, WM_S2_ROADMAP | CURRENT |
| `docs/LITCHECK_MULTICAV_WM_RL_20261009.md` | Lit check: multi-CAV WM-RL (C4) | 2026-10-09 | untracked | WM_S2_ROADMAP | CURRENT |
| `docs/MORNING_SUMMARY_20261005.md` | Morning summary, WM-N1 A8–A11 | 2026-10-05 | untracked | presentation numbers.json | SUPERSEDED but KEPT (morning summary; presentation-referenced) |
| `docs/NEXT_EXPERIMENT_PLAN.md` | Proposed experiment sequence after brake/HOLD audit | 2026-09-06 | tracked | INDEX, ROUTE, ledger | SUPERSEDED (wall-only planning retired) — not proposed (ledger-referenced) |
| `docs/OBSERVATION_DECISION_CONTRACT.md` | Observation/decision contract v2 | 2026-09-25 | tracked | INDEX, ledger | SUPERSEDED (closed wall entry/continuation era) — not proposed (ledger-referenced) |
| `docs/OPEN_QUESTIONS.md` | Open-question decision guide (append log) | 2026-10-04 | tracked | AGENTS, README, INDEX, ROUTE, PROJECT_LOG, ledger, tools | NEEDS-UPDATE (WM-S questions added) |
| `docs/PROJECT_LOG.md` | Historical status log (moved from README by MIG-03) | 2026-09-24 | tracked | AGENTS, README, INDEX, ledger | CURRENT (historical record; keep) |
| `docs/REDESIGN_V3_ZCODE_PLAN.md` | One-shot ZCode GLM refactor plan | 2026-10-01 | untracked | — | SUPERSEDED (OpenCode-only, no ZCode; refactor not active) — DELETE-PROPOSED |
| `docs/REORG_PLAN.md` | One-root reorganization + deletion plan | 2026-10-02 | untracked | — | SUPERSEDED (phase 3 executed; phase 2 gated) — DELETE-PROPOSED |
| `docs/RESEARCH_ASSET_MIGRATION.md` | MIG-01/02 asset isolation record | 2026-09-17 | tracked | AGENTS, ROUTE, INDEX, PROJECT_LOG, OPEN_QUESTIONS, ledger | CURRENT |
| `docs/RESULTS_SUMMARY.md` | Generated results summary from ledger | 2026-10-04 | tracked | README, INDEX, ROUTE, tools, tests | NEEDS-UPDATE (generated; predates WM-S ledger entries; out of this task's edit scope) |
| `docs/ROUTE.md` | Current route (append log) | 2026-10-04 | tracked | AGENTS, INDEX, PROJECT_LOG, NEXT_EXPERIMENT_PLAN, OPEN_QUESTIONS, EXPERIMENT_HISTORY, template, ledger, tests | NEEDS-UPDATE (WM-S route block added) |
| `docs/STATUS.md` | "Current status — 2026-09-08" (PPO era) | 2026-09-24 | tracked | PROJECT_LOG, INDEX, adoption manifest, ledger, GITHUB_PLAN, EXPERIMENT_LEDGER | OUTDATED-MARKED (self-dated 2026-09-08) — not proposed (frozen adoption manifest + PROJECT_LOG inbound) |
| `docs/WM_S2_ROADMAP.md` | WM-S contributions + weekly plan | 2026-10-09 | untracked | WM_STATE_PLAN | CURRENT (status lines updated) |
| `docs/WM_STATE_PLAN.md` | WM-S physics-informed WM plan | 2026-10-06 | untracked | — | CURRENT (status lines updated by WM_S2_ROADMAP) |
| `docs/WORLD_MODEL_DECISION_PROTOCOL.md` | Stage-4 rehearsal benefit-evaluation contract (2026-09-21) | 2026-09-25 | tracked | — | SUPERSEDED (WM-N1 delay-head/stage-4 direction retired) — DELETE-PROPOSED |
| `docs/WORLD_MODEL_RESEARCH_GOAL.md` | Active owner goal + ordered work | 2026-09-25 | tracked | AGENTS, INDEX, PROJECT_LOG, ROUTE, OPEN_QUESTIONS, ledger | NEEDS-UPDATE (controller-first ordering superseded by WM-S; out of this task's edit scope) |
| `docs/presentation_20261005/OUTLINE.md` | Presentation outline (current) | 2026-10-05 | untracked | build_deck*.py | CURRENT |
| `docs/presentation_20261005/OUTLINE_progress.md` | Earlier progress outline | 2026-10-05 | untracked | build_progress.py | SUPERSEDED (by OUTLINE.md) — listed as candidate, held (script inbound) |
| `docs/presentation_20261005/OUTLINE_short.md` | Short outline variant | 2026-10-05 | untracked | build_short*.py | SUPERSEDED (by OUTLINE.md) but script-referenced — not proposed |
| `docs/presentation_20261005/OUTLINE_v1.md` | Earliest outline draft | 2026-10-05 | untracked | — | SUPERSEDED (by OUTLINE.md) — DELETE-PROPOSED |

## Notes

- `docs/EXPERIMENT_LEDGER.md`, `docs/NEXT_EXPERIMENT_PLAN.md` and
  `docs/OBSERVATION_DECISION_CONTRACT.md` are named inside
  `protocols/experiment_history.json`; `docs/STATUS.md` and `docs/ARCHITECTURE.md`
  are named inside the frozen `baselines/circles/adoption_manifest.json`. They were
  therefore classified SUPERSEDED but **not** proposed for deletion, to avoid
  breaking ledger/manifest references that this task must not edit.
- `docs/CLOUD_OFFLOAD_PLAN.md` and `docs/GITHUB_PLAN.md` are linked from the
  historical `docs/HANDOFF_20261002.md`, which this task must not edit, so they are
  not proposed for deletion either.
- `docs/MORNING_SUMMARY_20261005.md` and `docs/HANDOFF_20261002.md` are the
  handoff/morning-summary class the owner asked to keep.
- Five files have **zero** inbound references and no evidence role, and are proposed
  for deletion in [DOCS_DELETE_PROPOSAL.md](DOCS_DELETE_PROPOSAL.md):
  `IMPROVEMENT_OPTIONS_20261006.md`, `REDESIGN_V3_ZCODE_PLAN.md`, `REORG_PLAN.md`,
  `WORLD_MODEL_DECISION_PROTOCOL.md`, `presentation_20261005/OUTLINE_v1.md`.

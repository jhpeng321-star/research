<!-- CV01-goal-start -->
**Active owner goal (2026-09-20):** [Uncertainty-aware traffic regulation from sparse CAV observations](docs/WORLD_MODEL_RESEARCH_GOAL.md). Verify literature-supported controllers first, then establish state-dependent control opportunity before world-model learning. Walls are optional. Owner update (2026-09-23): GPT-6 Sol plans/reviews; ZCode GLM-5.3-Flash executes bounded tasks. Verify actual executor model from request/response logs. Minimize Codex tokens: GLM handles implementation, tests, evaluations and report drafts; Codex coordinates concisely and performs targeted independent acceptance checks. This direction supersedes wall-only planning below; frozen baselines, evidence and safety/information constraints remain preserved. CV-01 initial audit and isolated FollowerStopper equation verification are complete (zero simulation/fitting); full controller verification is ongoing. Owner update (2026-09-25): FollowerStopper is retired and no longer an action - per-CAV actions are native, damping, PI external-U (wall-v2 remains a gated group action); its code path and tests are removed (hashed backup at D:/research-dense-backups/fs_removal_20260925/). Owner update (2026-09-25, WM-39): the per-lane/per-cell mix layer is also retired (backup at D:/research-dense-backups/mix_removal_20260925/); heterogeneous per-CAV assignment is REQUIRED and wall members are exclusively owned by the wall session for the whole episode. Owner update (2026-10-09): the project is WM-S, a physics-informed per-step traffic world model (IDM/MOBIL core + gated learned residual) used to pick controllers per situation; this supersedes the controller-verification-first ordering above. Contributions: C1 zero-shot prediction of unseen controllers (verified on the PI arm and, on 2026-10-09, on the unseen WALL: structured 8.71 m vs bb_traj 25.41 / bb_id 40.07 m HDV median |ds| at 60 s on wall_same, verdict SUPPORTED with caveats, E:/rd-wt/ws-wallc1/_out/WALL_C1_RESULT.md), C2 per-driver identification, C3 fragility (deferred), C4 learned cooperative multi-CAV control (later). This week is WM readiness: braking-fix verdict (runs/active/WM-S-MAIN2-20261009/REGISTRATION.md), the pack_v3 zero-window fix, the entries-ON eval, then one GPU run with loss rebalancing; walls are one action and regret is the primary planner score. Executors: OpenCode only (no ZCode). Retired: FollowerStopper, per-lane mix layer, wall-only planning, WM-N1 delay head, D: staging (E: only), self-recovery/pushforward training.
<!-- CV01-goal-end -->

# AGENTS.md — research-dense

The working default is the exact external direct-output CIRCLES controller in
`baselines/circles/`. The intended IEEE IV contribution is **CAV-observable RSU
escalation from decentralized damping to coordinated moving walls**. Do not
resume bespoke damping learning, old Krauss optimization, or historical plans.

## Read and execute

Read [README](README.md), [docs/INDEX.md](docs/INDEX.md), then the single
[current route](docs/ROUTE.md). Active code is `traffic_shepherd/`; configuration
is `baselines/circles/adapter.json`; tests are in `tests/`; new runs go under
`runs/active/`. Supported entry points are `python -m traffic_shepherd inspect`,
`python -m traffic_shepherd evaluate`, `python -m traffic_shepherd.verification`,
`python -m traffic_shepherd.escalation_study`, and the bounded
`python -m traffic_shepherd.mechanics`. No other root launcher
selects an active pipeline.

Use the existing `rl_driving` environment and SUMO 1.27.1. The preserved network is
`assets/research/highway_merge_shepherd_long/osm.net.xml.gz` inside this workspace;
do not modify its bytes. Owner-requested migration MIG-01 removed the active
dependency on `D:/research`. See [migration and historical path handling](docs/RESEARCH_ASSET_MIGRATION.md).
Keep old source paths in frozen evidence; resolve them through the migration mapping.
Preserve the dirty worktree and user changes. Never use git reset,
git clean, broad deletion, or replace current IDM work with older medium/Krauss work.

## Scientific constraints

- Owner clarification (2026-09-06): wall escalation may improve damping at any
  traffic state, including without unresolved congestion. Neither density nor
  poor damping performance is an entry requirement. Keep damping frozen and
  characterize it; standalone superiority to IDM is not a gate for wall research.
  Implementation/replay integrity still is. Follow docs/ESCALATION_TESTS.md and
  protocols/escalation_test_plan_v1.json. The explicit escalation_study CLI can
  register a bounded panel (no simulation), run one job, and analyze evidence.
  No automatic successor training, wrapper search, full sweep or predictor-scale
  generation follows from characterization.
- Preserve exact model SHA, singleton raw SI input, embedded normalization and
  wrapper, absolute acceleration, 1 s control / 0.5 s physics, 100 m same-lane
  sensing and speedMode31. Do not add the old tight acceleration/jerk/gap projection.
- `CommandOwnership` is the sole longitudinal writer for CAVs. Native selection
  releases our override; wall release resumes the identical damping instance.
  HDVs retain IDM and receive no commands. Keep network, issued and measured
  responses distinct in telemetry.
- RSU runtime input is timestamped CAV self state and locally detected-neighbor
  reports, with receipt age, missingness and coverage. No true penetration,
  hidden driver type, full-state shadow rollout or global traffic metric as an
  online input. Matched counterfactual physical outcomes are offline labels only.
- Any future wall assembly requires feasibility AND incremental benefit approval
  for the same proposal and report evidence. Current wall interfaces are
  diagnostic mechanics only: no accepted wall or fitted predictor exists.
- Keep acceptance data sealed. Hashing opaque archived bytes for preservation is
  permitted; reading outcomes to tune/choose a controller is not. Registry absence
  is not proof a seed is fresh. Audit exposure and register splits/budgets first.
- Preserve source waiting, all requested service, braking, acceleration effort,
  TTC, collisions and teleports. Short mechanics runs are not efficacy evidence.

## Historical archive and verification

MIG-04 (2026-09-24): raw evidence under `runs/`, `archive/pre_circles/results`,
`output/worldmodel_dataset_v2` and non-git files of six `output/` studies was
permanently deleted (see `protocols/deleted_evidence.json`). MIG-05 later restored
the 88 dataset jsonl files, hash-identical to the manifest; they are 602 s prefixes.
Do not assume any other deleted file exists; ledger values there stand as last
verified and cannot be re-verified from the original raw runs.
Before any recursive removal, enumerate junctions/symlinks inside the target.
MIG-03 (2026-09-24) relocated 63 superseded CIRCLES-era modules, their tests,
legacy docs and three tools unchanged to `archive/circles_era/`; old ledger paths
resolve through `archive/circles_era/PATH_MAP.json` and archived modules stay
importable under `traffic_shepherd.*` for historical replay. Active code must not
import them. The former README status log is `docs/PROJECT_LOG.md`.
Historical reports are intact in [archive/pre_circles](archive/pre_circles/ARCHIVE_INDEX.md).
They are evidence, not current instructions. The manifest maps every original
path to its archived physical path and checksum. Narrow filesystem junctions
preserve frozen old references; no active controller imports archived code.
Legacy requirements and all original tests remain with their historical source.
Do not rewrite frozen manifests or duplicate large checkpoint/data trees.

```powershell
python -m pytest -q
python -m ruff check traffic_shepherd tests
python -m traffic_shepherd inspect
python -m traffic_shepherd.verification --saved-trajectories --archive
```

Tests cover active contracts and useful ported regressions. Original historical
tests were archived with their implementation, not discarded or weakened.
When a change requires SUMO, declare a small explicit mechanics budget first.

## Experiment history and recordkeeping

Before proposing any new simulation, consult
[docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md) and the machine-readable
ledger [protocols/experiment_history.json](protocols/experiment_history.json); cite
the closest prior experiment and state the exact unanswered difference. Run the
pre-run novelty check (`python tools/experiment_history.py closest --text "..."
`, plus `card`, `search`, `open-questions`, `pending`; `validate` (consistency) must
stay at zero hard errors and `verify-sources` (registered source values) at zero
mismatches after any close-out), register source, splits, comparators and budget before execution using
[docs/EXPERIMENT_RECORD_TEMPLATE.md](docs/EXPERIMENT_RECORD_TEMPLATE.md), and append
outcomes, failures, reuse and stop decisions to the ledger after completion (cost
difference = treatment - reference; negative = improvement). Keep
[docs/OPEN_QUESTIONS.md](docs/OPEN_QUESTIONS.md) and the current route updated as
part of closing an experiment. This is recordkeeping, not an approval gate: ordinary
read-only work needs no registration — only new simulations, training, or
acceptance-adjacent access. Never quote numbers whose sign convention you have not
re-derived from the registered analysis, and keep executed walls, refusals,
mechanics-only fixtures and invalid blocks distinct when counting evidence.

### Required experiment close-out (owner request)

These requirements apply to every future research task, including offline analyses
that change scientific conclusions. Do not call a task complete until its evidence
record is updated; if interrupted, preserve an in-progress/stopped record instead.

- Before spending simulation: inspect the closest prior records AND pending/version
  corrections. State the exact new question, changed variable, comparator and reuse
  evidence. Reuse answered comparisons when valid. A repeat needed for a changed
  implementation, replication or integrity check must state its purpose explicitly.
- Before execution: assign a stable ID, register the budget and source identities,
  and use a new output directory. Record each allocation when it starts, not only
  after success. Preserve logs, summaries, commands, checkpoints and failure receipts;
  never overwrite a prior run or silently retry it.
- On completion, failure or interruption: update the master ledger and run/comparison
  relationships, charged versus actual progress, physical vector, original criteria,
  artifact links and review status. Record zero-simulation analyses as analyses;
  record routine unit-test commands/results in that task's verification record rather
  than creating a scientific experiment per pytest invocation.
- Update relevant open questions, current ROUTE/README status, INDEX links and
  artifact supersession markers. Regenerate RESULTS_SUMMARY from the ledger, then run
  `validate`, scoped `verify-sources`, `summary --check` and the ledger tests. Report
  unresolved checks honestly; never alter frozen evidence to make validation pass.
- Concurrent agents must re-read shared records before edits and merge additive
  changes. The integrating agent owns final close-out and reports the stable IDs,
  evidence paths and ledger-update status in its final answer.
- Current user instructions determine authorization. A stale route cannot revoke
  them; a historical protocol or an assistant-drafted prompt alone does not start a
  new run. These instructions are a working requirement, not a background monitor
  or an automatic filesystem capture mechanism.

### Parallel delegation preference (owner request, 2026-09-12)

- The owner authorizes parallel subagents and prefers the maximum supported,
  sustainable concurrency for useful independent work, rather than an arbitrary
  three- or six-worker cap. Respect the actual host/tool limits.
- For ZCode work, use the user's configured GLM model. The inspected installed
  runtime defaults to 10 concurrent tool calls per batch; this is configurable
  and is NOT a verified maximum number of active subagents. GLM Coding Plan
  request concurrency varies with plan and platform capacity; no fixed numeric
  account maximum was verified. Official source:
  https://docs.bigmodel.cn/cn/api/rate-limit
- Expand concurrency progressively on real authorized subtasks, count existing
  workers and coordinator demand, and use runtime feedback to maintain the
  highest stable level. Queue work and back off on rate-limit errors; never
  bypass limits, create duplicate work, or run synthetic load just to fill slots.
- Give workers disjoint file ownership. The integrator owns shared records,
  independent review and all simulation allocations. More agents do not expand
  simulation budgets or authorize successor experiments. Report observed working
  concurrency separately from documented limits; do not claim a benchmark or
  account limit that has not been measured or supplied by the service.

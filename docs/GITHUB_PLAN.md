**Coordinator read-only review (2026-10-02):** neither v3 proposal is adoption-ready. Both predate current WM-N1 GPU/A6/recovery fixes. v3-owner additionally loses snapshot chunk5 seed/exposure guards and uses older planner dispatch. Recommend continuing v3arch incremental/path-bound integration, preserving canonical frozen bytes, then installed guarded parity/export checks after training stabilization. No main merge or push is authorized by this review. See `tmp/zcode_review/COORDINATOR_CLOUD_STRUCTURE_REVIEW_20261002.json`.

# GitHub-ready structure: plan (no git writes, no push)

2026-10-01, organize session. A plan only. Pushing happens only on the owner's direct OK in the organize session.
Prior art: `E:/rd-release/traffic-shepherd-v2` (local, never pushed) already shows the pattern:
- fresh history;
- code plus small summaries;
- legacy work as one archive;
- secret and personal-path scan;
- a release-time test run in a fresh venv.

This plan extends that pattern to the current state.

## (a) Proposed public repo layout

```
traffic-shepherd/                 (new repo; fresh squashed history, see (d))
  traffic_shepherd/               active package: controllers, dispatcher (mixed_assignment/3), sensing, wall_group, planner/
  wm_n1/                          WM-N1 world-model package (from zcode/wd01: output/wd01/wm_n1)
  world_model/                    Part A / Part B code, predict contract (output/wd01/{predict, part4, partB_*}: code only)
  reconstructor/                  hybrid_state_v1, recon_v6 and lineage modules (code only)
  campaign/                       campaign_stage1_v2 registration, plan and branch-driver code (no run data)
  tools/                          experiment_history.py, holdout_guard.py, compare tools
  tests/                          active suite + module suites (fixtures ≤ 2.5 MB)
  scenarios/  assets/  baselines/ SUMO networks, routes, CIRCLES baseline (small; already tracked)
  protocols/                      experiment_history.json (ledger), registrations, holdout_exclusions.json (names only), deleted_evidence.json
  docs/                           ROUTE, INDEX, design docs (DESIGN_PARTS_A_B, PARTB_V2, PREDICT_CONTRACT_v2, PL04 closed loop, wall law), REORG/STORAGE plans
  results/                        small result summaries only (≤ 1 MB each; REPORT.md / *_summary.json), listed in results/README.md
  legacy/previous_work.tar.gz     pre-CIRCLES and CIRCLES-era code and docs (as in traffic-shepherd-v2)
  README.md  STATUS.md  pyproject.toml  requirements*.txt  .gitignore  .gitattributes (* -text)
```

Notes:
- The package currently imports research modules as `output.<name>`. Two options:
  - (i) keep an `output/` code-only namespace, as traffic-shepherd-v2 did. This is lowest risk and needs no code edits.
  - (ii) rename to the clean names above, with import shims.

  Recommendation: (i) for the first upload, (ii) as a follow-up refactor with tests.
- The release-time edits made for traffic-shepherd-v2 must be redone:
  - repo-relative paths in place of hard-coded worktree paths;
  - `sys.executable` in place of machine interpreter paths;
  - `<USER_HOME>` redaction.

## (b) NEVER in the public repo

Enforced by `.gitignore` plus a pre-push check script (`tools/prepush_check.py`, to be written). The check fails the push on any hit.

| class | rule |
|---|---|
| run and campaign data | `runs/`, `output/**/results*/raw`, `*.jsonl` over 1 MB, `sync_staging/`, `tmp/` |
| datasets, packs, shards | `output/worldmodel_dataset_*`, `**/pack_*`, `**/upload_*`, `*.npz`, `*.npy`, `*.parquet`, `*.tar.zst`, `*.zip` (except legacy/previous_work.tar.gz) |
| checkpoints | `*.pt`, `*.pth`, `*.pkl`, `*.ckpt`, and any file over 50 MB |
| HOLDOUT / CAL / sealed | every path in `protocols/holdout_exclusions.json`, `*/holdout/*`, `*/cal/*` data, sealed chunk-5 (`campaign_stage1_chunk5*`), INC1 data. The check refuses any path whose name matches the exclusion list. Only names are compared; contents are never opened |
| other drives and local paths | anything under `D:/`, `D:/rd-staging`; absolute `C:/Users/<user>` paths are redacted to `<USER_HOME>` |
| credentials and tool config | `.config/opencode/`, `.zcode/`, `.codex/`, `.claude/`, `.cursor/`, `*.env`, `*key*`, `*token*`, `*.pem`, `rented_box.pub` and other SSH material, the bridge `config/local.json` |

**Secret scan (2026-10-01).** Scope: tracked files of `checkpoint-2026-09-28` and `zcode/wd01`. Only file names and counts are reported; no values were printed.

- Provider-key patterns (GitHub `ghp_` / `github_pat_`, OpenAI-style `sk-…`, AWS `AKIA…`, Slack `xox…`, Google `AIza…`, PEM private keys): **0 files**.
- Generic `api_key|secret|token|password = "<16+ chars>"`: **0 files**.
- User-home paths (`C:/Users/<user>`): 15 files (checkpoint) and 63 files (wd01). These get redacted at export.
- `D:/` path literals: about 1,000–1,150 files. They are D:-era references in frozen records. They are harmless to publish but reveal local layout. They stay in the ledger and docs, which are already resolved by the RD_WT_ROOT mapping.

Not covered: history blobs were not secret-scanned. A fresh squashed history (see (d)) removes that exposure entirely.

## (c) Branch integration (before export)

Integration branch: create `integ/github-2026-10` from `zcode/wd01` (it contains `checkpoint-2026-09-28` plus ~414 WM commits), then merge in this order:

1. **Divergent branches with 1–2 commits not in `zcode/wd01` or the checkpoint:** zcode/pifix (2), zcode/inc1wire (2), zcode/wd01x2 (2), zcode/wd01w (2), zcode/wd01x (1), zcode/wd01pa2 (1), zcode/wallrt (1), zcode/kfixB (1), zcode/feat1run3 (1). Review each one: the wd01x/w/pa2 commits may be superseded experiments, so merge only the ones still wanted.
2. **Planner:** worktree `pl01` (branch zcode/pl01) has uncommitted changes (3), and PL-04 closed loop is being ported to mixed_assignment/3. Its owning session commits and merges first.
3. **Live worktrees with uncommitted work, each committed by its owning session:** `wmn1integC` (7 changes), `gpu1` (detached, 2), `j4bfix` (detached, 27).
4. **Everything else:** the other ~150 local branches are already contained in `zcode/wd01` or the checkpoint. No merge is needed.

After each merge, run: the full active suite, the module suites, `experiment_history.py validate` and `verify-sources`. Each owning session does its own merges; the integrator reviews.

## (d) History check

- **Size:** `.git` is 194 MB (63 MB loose, 14 MB packed, ~5,900 objects). The tracked tree is small: 3,420 files, about 8.4 MB.
- **Large blobs in history:**
  - `archive/pre_circles/manifest.json`, 52.5 MB. That is above GitHub's 50 MB warning size and below its 100 MB hard limit.
  - Several `output/wd01/partA_v2g/results/*.jsonl` files (5–14 MB each).
  - `output/wd01/partB_v04b/results/*/model_deployable_*.pkl` (10–11 MB; pickled models, which are also a code-execution risk for anyone loading them).
  - `tmp_stern_1705.pdf` (5.3 MB).
- **Recommendation: a fresh squashed history.**
  - The history mixes code with result data, pickles and machine paths.
  - It has never been secret-scanned blob by blob.
  - It includes HOLDOUT-adjacent result files on some branches.
  - The internal repo keeps its full history; the public repo starts from a single "snapshot" commit. Later releases can be follow-up commits, or new squashes.

## (e) Public vs private

| | public | private |
|---|---|---|
| visibility for the professor and collaborators | anyone can read; a citable link | invited collaborators only |
| risk | any slip (path, dataset fragment, HOLDOUT name, unpublished result) is permanent; web caches and forks | contained; can be opened later |
| HOLDOUT / unpublished results | must exclude even summaries that could leak holdout outcomes, and pre-publication numbers | summaries are fine |
| licence and third-party code | needs a licence choice and attribution checks (CIRCLES baseline: MIT, attribution required; SUMO networks: check source licence) | same, but less urgent |
| recommendation | later, after the paper | **now: private** (as for traffic-shepherd-v2); flip to public after a final review |

## Proposed execution (after owner OK)

1. Each owning session completes the integration in (c), and the suites pass.
2. Export into a new folder with a fresh `git init`:
   - code-only copy plus results summaries plus the legacy archive;
   - redaction;
   - `prepush_check.py`.
3. Release verification: fresh-venv tests, the secret scan and the size check.
4. Create a private GitHub repo and push, **only on the owner's direct OK in the organize session**.

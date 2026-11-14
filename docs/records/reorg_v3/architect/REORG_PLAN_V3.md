# REORG_PLAN_V3 — one-folder reorganisation of research data on `E:`

Generated 2026-10-09 by the reorg3 architect. **PLANNING ONLY.** Nothing was moved,
deleted, renamed or uploaded. The three scripts in `scripts/` are delivered
**unexecuted**; the owner runs them. No `rclone.conf`/credential file was opened;
`sealed/`, CAL, HOLDOUT, chunk-5, J4B and INC1 contents were never opened; no
process was killed.

Inputs read in full (read-only):

| input | what it gave us |
|---|---|
| `E:/rd-wt/reorg3/s1_E/INVENTORY_E.md` + `inventory_E.json` | E: top-level, repo subfolders, runs/output, 209 worktrees, 186 junctions, duplicate groups |
| `E:/rd-wt/reorg3/s2_D/INVENTORY_D.md` + `inventory_D.json` | `D:/rd-staging` 153.29 GiB, classes, Drive-verification record, blockers |
| `E:/rd-wt/reorg3/s3_C_G/INVENTORY_CG.md` + `inventory_CG.json` | C: research items + `%TEMP%` reclaim, G: Drive cold mount |
| `E:/rd-wt/reorg3/s4_refs/PATH_REFS.md` + `path_refs.json` | 42,674 hard-coded path hits; impact buckets; junction strategy; 14 in-repo junctions |
| `E:/research-dense/docs/REORG_PLAN.md` | prior unexecuted phase-2 moves + executed phase-3 deletions |
| `E:/rd-wt/reorg2_architect/REORG_PLAN_V2.md`, `MOVE_MAP.json` | V2 design (junctions, worktree move) |
| `E:/rd-wt/reorg2_reviewer/REVIEW.md` | VERDICT **FAIL**: 1 BLOCKER (renaming the live tree) + 6 MAJOR |
| `E:/research-dense/AGENTS.md` | never broad-delete; preserve the dirty worktree; enumerate junctions before recursive removal; frozen evidence |
| `E:/research-dense/docs/RESEARCH_ASSET_MIGRATION.md` | MIG-01/02 path handling; do not rewrite frozen manifests |

Measured now (`Get-PSDrive`, 2026-10-09):

| drive | used GiB | **free GiB** | total GiB |
|---|---:|---:|---:|
| E: | 470.33 | **6.30** | 476.63 |
| D: | 415.32 | 36.33 | 451.64 |
| C: | 330.42 | 69.58 | 400 |
| G: (Drive stream cache) | 417.13 | 34.51 | 451.64 |

Cloud target: `rclone` remote `gdrive` (Google Drive, 5 TB). G: is only the local
stream cache of one cold sub-tree; capacity for this plan is the 5 TB remote.

---

## 1. Decision — single root `E:/research`

**Chosen: `E:/research` as the umbrella root; rename `E:/research-dense` →
`E:/research/repo` with a permanent junction left at the old path.**

Reasons:

1. **The goal is one folder.** `E:/research-dense` is the git working tree
   itself, not a container. Using it as the umbrella would keep 12 other
   `E:` top-level folders beside it and would invite placing untracked cold data
   inside a git repo (history noise, `git status` pollution, accidental commits).
2. **`E:/research` does not exist yet** (verified), so there is no collision and
   no risk of overwriting. `PATH_REFS.md` already names it as the planned root.
3. **All old absolute paths stay resolvable** through a junction at
   `E:/research-dense` kept indefinitely. That covers **32,712** literals to
   `E:/research-dense`, **175** to `E:/research-dense-backups`, **2,328** to
   `E:/rd-wt`, **3** to `E:/rd-release`, and the **14 existing in-repo junctions**
   whose targets are absolute `E:\research-dense\...` (moving the repo would break
   them without the old-path junction).
4. **Zero extra space needed.** Every `E:` move is a same-volume rename.
5. **Active code already supports relocation** via env vars with `E:` defaults
   (`RESEARCH_DENSE_ROOT`, `RD_WT_ROOT`, `RD_BACKUPS_ROOT`); no literal rewrite is
   strictly required after the junction exists.

**Counter-argument and mitigation.** Keeping `E:/research-dense` as the root would
avoid reviewer BLOCKER #1 (renaming a live tree). We therefore split phase 2:
**2a** moves only idle siblings/worktrees and never touches the live tree;
**2b** (the `repo` rename) is scripted but hard-gated on a confirmed zero-writer
window and `-AllowHighRisk`, and is not proposed for tonight.

**Important structural constraint.** `runs/`, `output/`, `sealed/`, `archive/`,
`tmp/` must stay **inside** `repo/`: ~124,000 repo-relative ledger/doc cites,
375 `output.<name>` imports, and 42,088 frozen-evidence literals assume the
in-repo layout. `data/` in the target tree is therefore for **external** data
(packs/staging/datasets), not for the in-repo evidence tree.

---

## 2. Target tree

```
E:/research/                         <- single root (NEW; created by phase2_moves.ps1)
  PATH_MAP.json                      <- old -> new + junction list (MIG-03 style)
  README.md
  repo/                              <- E:/research-dense            [rename 2b + junction]
      runs/  output/  sealed/  archive/  tmp/  protocols/  assets/ ...  (UNCHANGED inside)
  worktrees/                         <- 60 idle ARCHIVE worktrees    [git worktree move 2a]
  data/
    packs/
      wm_n1/                         <- D:/rd-staging/wmn1_feat/{pack_v3,shards_v3,labels_*}  [5, gated]
    staging/                         <- E:/rented_bundle, E:/rented_bundle_dspeed, D:/rd-staging/cpu_payload
    datasets/                        <- C:/Users/jpeng/ray_results  [5, ACL-blocked]
  release/                           <- E:/rd-release ; release/github-export <- E:/research-github
  backups/                           <- E:/research-dense-backups
  archive/                           <- E:/migration_logs, E:/research-dense-c-migrated
  tools/
    glm/  glm-worker/                <- E:/glm, E:/glm-worker
  scratch/                           <- E:/tmp   (junction targets from ws-p2/_ext point here)
  incoming/                          <- small C: research artifacts (SUMO msi/video, PDFs, pptx, tar)

Stays a REAL directory (NOT junctioned, NOT moved):
  E:/rd-wt/                          <- live + KEEP-ACTIVE worktrees + per-worktree junctions
Stays off E::
  D:/rd-staging/                     <- owner staging exception (until Drive-verified)
  C:/Users/jpeng/miniconda3/envs/rl_driving
  C:/Users/jpeng/Documents/Codex/tools/AgentControlPlane
  /root/autodl-tmp/... (gpuA/gpuB)
  G:/                                <- already Drive
```

---

## 3. What goes where

| source | destination | size | method | compatibility |
|---|---|---:|---|---|
| `E:/research-dense` | `E:/research/repo` | 403.97 GiB | rename (2b, gated) | junction at old path |
| `E:/rd-wt/<idle ARCHIVE wt>` × 60 | `E:/research/worktrees/<name>` | 36.59 GiB | `git worktree move` (2a) | per-worktree junction |
| `E:/rd-release` | `E:/research/release` | 1.013 GiB | rename (2a) | junction |
| `E:/research-github` | `E:/research/release/github-export` | 0.031 GiB | rename (2a) | junction |
| `E:/research-dense-backups` | `E:/research/backups` | 2.026 GiB | rename (2a) | junction |
| `E:/migration_logs` | `E:/research/archive/migration` | 0.000 GiB | rename (2a) | junction |
| `E:/research-dense-c-migrated` | `E:/research/archive/c-migrated` | 0.055 GiB | rename (2a) | junction |
| `E:/glm`, `E:/glm-worker` | `E:/research/tools/{glm,glm-worker}` | ~0 | rename (2a) | junctions |
| `E:/tmp` | `E:/research/scratch` | 0.361 GiB | rename (2a) | junction (keeps ws-p2 `_ext` targets valid) |
| `E:/rented_bundle`, `E:/rented_bundle_dspeed` | `E:/research/data/staging/...` | 0.163 GiB | rename (2a) | junctions |
| `D:/rd-staging/wmn1_feat/{pack_v3,shards_v3,labels_*}` | `E:/research/data/packs/wm_n1/` | ~100 GiB | robocopy+verify (5) | junction `D:/rd-staging/wmn1_feat` → target |
| `D:/rd-staging/cpu_payload` | `E:/research/data/staging/cpu_payload` | 1.118 GiB | robocopy+verify (5) | junction at old path |
| small C: artifacts | `E:/research/incoming/` | ~0.21 GiB | copy+verify (5) | none |
| `C:/Users/jpeng/ray_results` | `E:/research/data/datasets/ray_results` | unknown | robocopy+verify (5) | junction; **ACL-blocked** |

Full machine map: `MOVE_MAP.json` (83 explicit moves; no wildcards).
`E:/rd-wt$w` (junction to `ws-p2/_ext`, no data) is a delete candidate, not a move.

**Live worktrees — never moved tonight** (hard-coded refusal list in the script):
`E:/rd-wt/ws-main`, `ws-main2`, `ws-brakefix`, `ws-trainsetup`, `ws-wallc1`.
Also not moved: every dirty worktree, `gpu1`, `scorer-a9full`, `feat1j4fix`,
`bundle3` (holds `sync_config_live.json`/`queue/jobs.json`), and the
KEEP-ACTIVE process-evidence worktrees (`ws-eval`, `ws-evalshift`, `ws-p5`,
`ws-run3`, `ws-swgen`, `ws-swpack`, `ws-wall`).

---

## 4. Drive-only (upload → server-side verify → owner deletes locally)

Verification is `rclone copy` then `rclone check --one-way` per item, recorded in
a manifest (`scripts/drive_offload.sh`). **No local deletion happens before a
passing check.** `sealed/` is uploaded as opaque bytes only (hashing bytes is
permitted; opening outcomes is not).

| item | size | Drive status today | disposition |
|---|---:|---|---|
| `E:/research-dense-c-migrated` | 0.055 GiB | not verified | upload, verify, delete |
| `E:/research-dense/output/partB_bundles` | 0.524 GiB | not verified | upload, keep shipped zips+sha, delete dir |
| idle ARCHIVE worktrees `_out`/ignored output (60) | ≤36.59 GiB | not verified | archive, verify, `git worktree remove` |
| `E:/research-dense/output/evidence_regeneration_v1` | 22.43 GiB | not verified | **OWNER-GATED** frozen evidence |
| `E:/research-dense/runs/campaign_stage1_chunk1` p40 superseded subset (WM-42/44/56) | 24.2 GiB | not verified | **OWNER-GATED** frozen evidence |
| `E:/research-dense/sealed` | 61.83 GiB | not verified | **OWNER-GATED** acceptance data |
| `D:/.../upload_v2` | 6.391 GiB | **VERIFIED 7/7** (2026-10-06) | delete now |
| `D:/.../upload_v2c` | 6.438 GiB | **VERIFIED 7/7** | delete now |
| `D:/.../raw_chunk1_p40` | 4.553 GiB | **VERIFIED 294/294** | delete now |
| `D:/.../shards_v2/{chunk1,chunk1_fix27,chunk2,index}` | 13.5 GiB | **VERIFIED** | delete now |
| `D:/.../shards_v2/{chunk3,chunk4,truearms_chunk2,truearms_chunk3,walls_pilot_chunk4}` | 21.0 GiB | **TRUNCATED on Drive** | re-upload, verify, then delete |
| `D:/.../{pack_v3,shards_v3,cpu_payload}` | 101.1 GiB | not on Drive | upload+verify (pack_v3 also gpuA/gpuB 18/18 sha), then delete **or** move to E: |
| G: `research-dense-cold` (48.54 GB) | — | already Drive | leave |

---

## 5. Delete candidates (why each is safe)

Executed only by the owner via `scripts/DELETE_AFTER_VERIFY.ps1`
(report-only default, `-Execute` to act, per-item manifest, hard-coded paths,
no wildcards). The script hard-blocks `sealed|CAL|HOLDOUT|chunk-5|chunk5|J4B|INC1`.

### 5a. Regenerable (no Drive check needed)
| path | size | why safe |
|---|---:|---|
| `E:/rd-release/{venv-test,venv-pptx}` | 0.916 GiB | virtualenvs recreate from requirements |
| `E:/rd-wt/reorg2_*, reorg3, reorg_inv_*` (13 dirs) | 0.093 GiB | prior-plan scout scratch |
| `E:/research-dense/tmp` scratch subset | 0.377 GiB | all `DELETE-CANDIDATE` entries; `tmp/zcode_review` + `tmp/storage_audit` are ledger-cited and KEPT |
| `E:/research-dense/runs` smoke/failed | 0.126 GiB | name-marked smoke + failed attempts |
| `C:/Users/jpeng/AppData/Local/Temp/claude` | 4.637 GiB | agent temp |
| `C:/Users/jpeng/AppData/Local/Temp/pip-unpack-*` | 0.127 GiB | pip temp |
| `C:/Users/jpeng/AppData/Local/Temp/rd_v3_*.txt`, `RUNS_*.jsonl`, `wm_run*_tail.txt` | ~0.001 GiB | transient logs |
| `C:/Users/jpeng/{Sumo,ZCodeProject}` | 0 | empty dirs |

### 5b. Superseded / duplicated (evidence recorded)
| path | size | why safe + evidence |
|---|---:|---|
| `E:/research-dense/output/partB_bundles` | 0.524 GiB | pre-staging bundles; shipped zips+sha256 kept; `REORG_PLAN_V2` SAFE |
| `E:/rd-wt$w` | 0 | junction to `ws-p2/_ext`, no data; `inventory_E.json` reparse record |
| E: duplicate groups outside `sealed/` | ≤2.15 GiB | size+basename match **not hash-verified** → hash-verify each group before delete; `inventory_E` duplicates |
| `D:/rd-staging/wmn1_feat/shards` (v1) | 0.075 GiB | aborted shard-key collisions; `FEAT1_REPORT.json` |
| `D:/rd-staging/box_push_j4b_bases` | 0.008 GiB | duplicates 16 `__base` dirs of E: `runs/campaign_stage1_chunk4` (`RUN_RECEIPT.json`) |
| `D:/rd-staging/plexp1d_v3_root` + slot tars | 0.156 GiB | code duplicated in `E:/rented_bundle/code_plexp1d` |
| `D:/rd-staging/wmn1_feat/{logs,quarantine_v2,*}` | 0.005 GiB | regenerable featurization logs |
| `D:/research-dense` | 0 | stray caches only; owner rule says it should not exist |

### 5c. Drive-verified only (D:, listed in §4)
`upload_v2`, `upload_v2c`, `raw_chunk1_p40`, verified `shards_v2` subfolders
(30.88 GiB), then re-uploaded `shards_v2` truncations (21.0 GiB), then
`pack_v3`/`shards_v3`/`cpu_payload` (101.1 GiB).

**NEVER delete:** any frozen evidence not explicitly waived (ledger-cited
`runs/`, `output/`, `protocols/`, `archive/`, registrations, manifests,
`sealed/`), `runs/active`, `campaign_walls_pilot_chunk4*`, `tmp/zcode_review`,
`tmp/storage_audit`, dirty/live worktrees, `D:/rd-staging` (until verified), and
CPU/GPU/box paths.

---

## 6. Space math, step by step

Baseline: **E: free 6.30 GiB**, D: free 36.33 GiB, C: free 69.58 GiB.

**Step 1 — E: local reclaim, no Drive (owner-run, low risk).**
```
  research-dense/tmp scratch            0.377
  runs smoke/failed                     0.126
  rd-release venvs                      0.916
  rd-wt scout scratch                   0.093
  E: duplicate groups (hash-verified)  +2.150  (optional, verify first)
  ------------------------------------------------------
  subtotal                             3.66 GiB  (1.51 GiB without duplicates)
  E: free  6.30 + 3.66              =  9.96 GiB  (~7.81 without duplicates)
```

**Step 2 — Drive-archive + delete E: cold/superseded blocks (rclone check required).**
```
  idle ARCHIVE worktrees (ignored output; keep git objects)   36.59
  output/partB_bundles                                         0.524
  ------------------------------------------------------------
  non-gated subtotal                                          37.11 GiB
  E: free  9.96 + 37.11                                     = 47.07 GiB
  + evidence_regeneration_v1 (gated)  22.43
  + chunk1 p40 superseded (gated)     24.20
  + sealed (gated)                    61.83
  gated subtotal                     108.46 GiB
  E: free if all waivers granted      ≈ 155.5 GiB
```

**Step 3 — D: Drive-verify then delete locally (frees D:, not E:).**
```
  already verified                                   30.88
  re-upload truncated shards_v2                      21.00
  pack_v3 + shards_v3 + cpu_payload                 101.10 (upload needed)
  ---------------------------------------------------------
  D: free  36.33 + 153.0                           ≈ 189.3 GiB
```

**Step 4 — move D: KEEP-ACTIVE to E: (needs E: free ≥ ~101.5 GiB).**
```
  pack_v3 48.803 + shards_v3 51.181 + cpu_payload 1.118 + labels 0.02 ≈ 101.1 GiB
  Feasible only on the gated path (E: free ≈ 155.5). Otherwise leave packs on D:
  (or Drive-only) and only move labels/cpu_payload (1.14 GiB).
```

**Why not everything on E:?** Consolidated E: content with the D: packs would be
```
  repo 403.97 + worktrees 59.26 + siblings 3.65 + D: packs 101.1 = 567.98 GiB
  on a 476.63 GiB volume  -> shortfall 91.35 GiB
```
So **full one-folder consolidation requires offloading ≥ ~91 GiB (round to ~100 GiB
for working headroom) of E: content to Drive first.** The only candidates that size
are frozen evidence (`sealed` 61.83 + `chunk1` p40 24.20 + `evidence_regeneration_v1`
22.43 = 108.46 GiB), which need explicit owner waivers. **Default recommendation:**
keep `sealed/` and live evidence on E:, offload D: packs to Drive (or keep them
Drive-verified on D:), and move to E: only `cpu_payload`/`labels` until the owner
grants waivers.

---

## 7. Phases and ordering

| phase | action | gate |
|---|---|---|
| 0 | preflight: `git worktree list` snapshot; junction enumeration; disk check | read-only |
| 1 | local deletions §5a/§5b | owner review, report-only first |
| 2a | create `E:/research`; rename idle siblings; `git worktree move` 60 idle trees; junctions | no open handles; not live |
| 2b | rename `E:/research-dense` → `E:/research/repo`; `git worktree repair`; junction | **zero-writer window + `-AllowHighRisk`** |
| 3 | Drive offload + `rclone check --one-way` per item (§4) | rclone remote `gdrive` reachable |
| 4 | owner deletes Drive-verified items | manifest shows `verified` |
| 5 | move D:/C: KEEP items to E:/research/data|incoming | E: free ≥ 102 GiB |
| 6 | move `E:/research-dense/runs/*` cold evidence (only with waiver) | owner waiver |

Ordering rule for `phase2_moves.ps1`: siblings first, then worktrees, then (gated)
repo rename. The journal is written **before** each mutation.

---

## 8. Compatibility

- **Junction at every old path, kept indefinitely** — this is the V2/V3 and
  MIG-03/MIG-04 pattern. It keeps the 35,218 `E_REORG_AFFECTED` literals and the
  14 in-repo absolute-target junctions valid.
- **Frozen evidence literals are never rewritten** (`output/`, `runs/`,
  `archive/`, `protocols/`, manifests, ledger).
- **Env vars** point active code at the new roots:
  `RESEARCH_DENSE_ROOT`, `RD_WT_ROOT`, `RD_BACKUPS_ROOT` (and optional umbrella
  `RESEARCH_ROOT`).
- **`E:/research/PATH_MAP.json`** records old→new + junction list; register a
  MIG-06 ledger entry.
- **Literal-only active spots** to fix only if they must run post-move:
  `tools/run_v22_smoke.py:109`; worktree-only planner tests that assert literal
  `E:/research-dense/...`; `D:/rd-staging` consumers (kept resolving by the
  `wmn1_feat` junction).
- **Untouched:** `/root/autodl-tmp` box paths, `C:/.../envs/rl_driving`,
  `D:/rd-staging/*` (owner exception), `G:/`.
- **Pre-existing breakage:** 4,241 `D:/research*` literals already unresolved
  today (MIG-01 maps them); not caused by V3.

---

## 9. Memory rule (applies to every follow-up script/analysis)

- Every Python process stays **under 6 GB**.
- mmap `.npy` packs with `mmap_mode='r'`; never load a whole pack.
- Process states/runs **one at a time**; never load all runs.
- The inventories and scripts here use **metadata/JSON only** — no pack or run
  content is opened.

---

## 10. Rollback

Because every `E:` move is a **same-volume rename** (no bytes rewritten), rollback
is symmetric and cheap. `MOVE_MAP.json` lists 83 explicit old→new pairs; the
journal (`scripts` → `journal/phase2_*.jsonl`) records each item immediately
before and after it changes.

**Rollback procedure (reverse order):**
1. **Stop** — do not start phase 3+.
2. Remove the compat junction at each moved old path (links only; targets are safe):
   `Remove-Item -LiteralPath <old> -Force` for reparse points.
3. Reverse each mutation using the journal's `after_move` records, newest first:
   - rename: `Move-Item -LiteralPath <new> -Destination <old>`
   - worktree: `git -C E:/research-dense worktree move <new> <old>` then
     `git -C <old> worktree repair`
   - robocopy (D:/C:): delete the partial destination, source is untouched until
     the owner's post-verify delete.
4. Restore junctions that were removed in error.
5. Optional automated reverse: `scripts/rollback_moves.ps1` is described but
   **not included** to keep the delivered surface minimal; the journal + `MOVE_MAP.json`
   are sufficient to script it in minutes. (Add it only if the owner wants one.)

**Deletions are not reversible locally** — that is why deletes are gated behind a
passing `rclone check` manifest and a report-only default. Recovery is
`rclone copy` back from `gdrive`. Sealed/evidence deletions need explicit owner
waivers and are the only irreversible high-value operations.

---

## 11. Deliverables and verification

| file | purpose | executed? |
|---|---|---|
| `REORG_PLAN_V3.md` | this plan | no |
| `MOVE_MAP.json` | 83 moves old→new + size/method/compat/order; drive-only + delete lists; never-move list | generated (read-only inputs) |
| `tools/build_move_map.py` | reproducible generator for `MOVE_MAP.json` | run (writes JSON only) |
| `scripts/phase2_moves.ps1` | journal-before-mutation; dry-run default; `-Execute`; refuses open handles + live worktrees; `-AllowHighRisk` for 2b | **not run** |
| `scripts/drive_offload.sh` | `rclone copy` + `rclone check --one-way` per item; verification manifest | **not run** |
| `scripts/DELETE_AFTER_VERIFY.ps1` | owner-run; only Drive-verified items; report-only default | **not run** |

Checks performed on the deliverables (no mutations):
`python tools/build_move_map.py` (83 moves, 0 duplicate destinations),
PowerShell parser passed for both `.ps1` files, JSON parses.

## 12. Blockers / open items

- `rclone` is not installed in this environment; Drive checks rely on the
  2026-10-06 record until the owner runs `drive_offload.sh` with `rclone` present.
- 5 `shards_v2` archives are truncated on Drive → re-upload needed.
- `pack_v3`/`shards_v3` (~100 GiB) are not on Drive and are still referenced by
  ~104 worktree paths → keep until uploaded/verified or junctioned.
- `C:/Users/jpeng/ray_results` cannot be sized (ACL Access denied) → owner must
  grant read.
- Full one-folder consolidation is ~91 GiB short unless ~100 GiB of frozen
  evidence is offloaded with owner waivers (see §6).

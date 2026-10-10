#!/usr/bin/env python
"""Generate MOVE_MAP.json for REORG_PLAN_V3.

Read-only: consumes the read-only inventories under E:/rd-wt/reorg3/s{1..4}
and writes only MOVE_MAP.json next to this repository. No filesystem mutation
(moves/deletes/renames) and no large data reads. Memory rule: only JSON
manifests are parsed; no .npy packs or run contents are opened.
"""
import json
import os
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REORG = r"E:\rd-wt\reorg3"
GiB = 1024 ** 3


def load(rel):
    with open(os.path.join(REORG, rel), encoding="utf-8") as fh:
        return json.load(fh)


inv_e = load(r"s1_E\inventory_E.json")
inv_d = load(r"s2_D\inventory_D.json")

TOP = {x["name"]: x for x in inv_e["top_level"]}


def sib(name):
    x = TOP[name]
    return x["bytes"], x["reparse_count"]


moves = []
order = 0


def add(old, new, size_bytes, method, compat, phase, preconds, klass, note=""):
    global order
    order += 1
    moves.append({
        "order": order,
        "old": old,
        "new": new,
        "size_bytes": size_bytes,
        "size_gib": round(size_bytes / GiB, 4) if size_bytes is not None else None,
        "method": method,
        "compatibility": compat,
        "phase": phase,
        "preconditions": preconds,
        "class": klass,
        "note": note,
    })


# --- Phase 2a: create root + same-volume renames of idle E: siblings ---
siblings = [
    ("E:\\rd-release", "E:\\research\\release",
     "release build workspace (ARCHIVE)"),
    ("E:\\research-dense-backups", "E:\\research\\backups",
     "backups incl. reorg_phase3 + hashed fs/mix removals (ARCHIVE)"),
    ("E:\\migration_logs", "E:\\research\\archive\\migration",
     "2026-09-28/29 E:/D: cutover logs (ARCHIVE)"),
    ("E:\\glm", "E:\\research\\tools\\glm",
     "GLM/ZCode worker launcher (ARCHIVE)"),
    ("E:\\glm-worker", "E:\\research\\tools\\glm-worker",
     "GLM worker scratch (ARCHIVE)"),
    ("E:\\tmp", "E:\\research\\scratch",
     "E: scratch; 9 junctions from live ws-p2/_ext point here (KEEP-ACTIVE)"),
    ("E:\\research-dense-c-migrated", "E:\\research\\archive\\c-migrated",
     "C:->E: migration copy of user files (DRIVE-ONLY)"),
    ("E:\\research-github", "E:\\research\\release\\github-export",
     "public GitHub export staging (KEEP-ACTIVE)"),
    ("E:\\rented_bundle", "E:\\research\\data\\staging\\rented_bundle",
     "rented-box bundle scripts/manifests (ARCHIVE)"),
    ("E:\\rented_bundle_dspeed", "E:\\research\\data\\staging\\rented_bundle_dspeed",
     "dspeed rented-box bundle variant + upload_staging (ARCHIVE)"),
]
for old, new, note in siblings:
    name = old.split("\\")[-1]
    b, rc = sib(name)
    add(old, new, b, "rename (same volume)", "junction at the old path, kept indefinitely",
        "2a", "no writer/sync/executor with an open handle under the source; "
        "enumerate junctions inside the source first; link targets kept valid by the old-path junction",
        "ARCHIVE/KEEP-ACTIVE", note)

# --- Phase 2b: the main git tree (HIGH RISK, quiesced window only) ---
rd = TOP["research-dense"]["bytes"]
add("E:\\research-dense", "E:\\research\\repo", rd, "rename (same volume)",
    "junction E:\\research-dense -> E:\\research\\repo, kept indefinitely",
    "2b",
    "ZERO-WRITER WINDOW: no SUMO/job/executor/sync/pytest writing under E:/research-dense; "
    "runs/active, campaign_walls_pilot_chunk4*, sealed chunk-5 quiescent; owner approval; "
    "snapshot `git worktree list`; after rename run `git worktree repair` from the new path",
    "KEEP-ACTIVE",
    "physically relocates live+sealed in-repo data; reviewer BLOCKER; do NOT run tonight")

# --- Phase 2a: idle ARCHIVE worktrees -> E:/research/worktrees ---
arch_wt = [w for w in inv_e["worktrees"] if w["proposed_class"] == "ARCHIVE"
           and w["registered_worktree"]]
for w in sorted(arch_wt, key=lambda x: -x["bytes"]):
    add("E:/rd-wt/" + w["name"], "E:/research/worktrees/" + w["name"], w["bytes"],
        "git worktree move + git worktree repair",
        "per-worktree junction at E:/rd-wt/<name> (E:/rd-wt stays a real directory)",
        "2a",
        "worktree registered, clean (dirty_all=0), not a live worktree, no process evidence, "
        "no open handle; E:/rd-wt/<name> must not be a live worktree",
        "ARCHIVE", "branch %s" % (w.get("branch") or "detached"))

# --- Phase 5: D: staging -> E:/research/data (after E: space freed + Drive verify) ---
d_moves = [
    ("D:\\rd-staging\\wmn1_feat\\pack_v3", "E:\\research\\data\\packs\\wm_n1\\pack_v3", 48.803,
     "robocopy /E /COPY:DAT + verify (sha256/count) then junction",
     "junction D:\\rd-staging\\wmn1_feat -> E: target keeps 1019 D:/rd-staging literals resolving",
     "KEEP-ACTIVE", "upload+server-side-verify on Drive FIRST, or pack_v3 mirrored on gpuA/gpuB (18/18 sha)"),
    ("D:\\rd-staging\\wmn1_feat\\shards_v3", "E:\\research\\data\\packs\\wm_n1\\shards_v3", 51.181,
     "robocopy /E /COPY:DAT + verify then junction",
     "junction at old path", "KEEP-ACTIVE", "upload+server-side-verify on Drive FIRST"),
    ("D:\\rd-staging\\cpu_payload", "E:\\research\\data\\staging\\cpu_payload", 1.118,
     "robocopy /E /COPY:DAT + verify then junction",
     "junction at old path", "KEEP-ACTIVE", "upload+server-side-verify on Drive FIRST"),
    ("D:\\rd-staging\\wmn1_feat\\labels_fix1_20261003T141730Z", "E:\\research\\data\\packs\\wm_n1\\labels_fix1", 0.007,
     "robocopy /E /COPY:DAT + verify then junction", "junction at old path",
     "KEEP-ACTIVE", "small; bundle with shards_v3 move"),
    ("D:\\rd-staging\\wmn1_feat\\labels_cov_20261004T072826Z", "E:\\research\\data\\packs\\wm_n1\\labels_cov", 0.013,
     "robocopy /E /COPY:DAT + verify then junction", "junction at old path",
     "KEEP-ACTIVE", "small; bundle with shards_v3 move"),
]
for old, new, gib, method, compat, klass, pre in d_moves:
    add(old, new, int(gib * GiB), method, compat, "5", pre, klass,
        "requires E: free >= ~%.1f GiB before copy; %s" % (gib, "moves off D: only after Drive verify"))

# --- Phase 5: C: small research artifacts -> E:/research/incoming ---
c_moves = [
    ("C:\\Users\\jpeng\\AppData\\Local\\Temp\\wm_n1_full.tar", "E:\\research\\incoming\\wm_n1_full.tar", 8.65, "copy+verify"),
    ("C:\\Users\\jpeng\\Downloads\\sumo-win64-1.27.1.msi", "E:\\research\\incoming\\sumo-win64-1.27.1.msi", 190.36, "copy+verify"),
    ("C:\\Users\\jpeng\\Downloads\\sumo-video.mp4", "E:\\research\\incoming\\sumo-video.mp4", 6.36, "copy+verify"),
    ("C:\\Users\\jpeng\\OneDrive\\Desktop\\WM_short_20261005.pptx", "E:\\research\\incoming\\WM_short_20261005.pptx", 0.04, "copy+verify"),
    ("C:\\Users\\jpeng\\Downloads\\Send Anywhere (2026-09-21 03-19-30)\\ResearchProposalForm_Jinghan_filled.pdf",
     "E:\\research\\incoming\\ResearchProposalForm_Jinghan_filled.pdf", 0.11, "copy+verify"),
    ("C:\\Users\\jpeng\\Documents\\Codex\\2026-09-07\\cuhk-2\\work\\research-registration\\ResearchProposalForm.pdf",
     "E:\\research\\incoming\\ResearchProposalForm.pdf", 0.02, "copy+verify"),
]
for old, new, mb, method in c_moves:
    add(old, new, int(mb * 1024 * 1024), method, "no junction (source leaves C:); archival copy only",
        "5", "no open handle under the source; verify size+sha256 after copy", "MOVE-TO-E")

# --- C: ray_results (unknown size, ACL-blocked) ---
add("C:\\Users\\jpeng\\ray_results", "E:\\research\\data\\datasets\\ray_results", None,
    "robocopy /E /COPY:DAT + verify then junction",
    "junction at old path until owner confirms",
    "5", "size UNKNOWN - enumeration returns Access denied (ACL); owner must grant read first; "
    "archive to Drive if too large for E:", "MOVE-TO-E", "blocked on ACL")

# ---- Drive-only (offload then delete locally; NOT executed here) ----
drive_only = [
    # E:
    dict(src="E:\\research-dense-c-migrated", remote="gdrive:research-dense-cold/01_Code_And_Restore/research-dense-c-migrated",
         size_gib=0.055, verify="2026-10-06 streamed-archive member hashes (existing record); re-verify rclone check --one-way",
         then="delete local after server-side check; already also moved to E:/research/archive/c-migrated in 2a"),
    dict(src="E:\\research-dense\\output\\partB_bundles", remote="gdrive:research-dense-cold/03_Experiment_Archives/partB_bundles",
         size_gib=0.524, verify="rclone check --one-way + sha256 of shipped zips",
         then="delete local dir after keeping shipped zips+sha256 (already offloaded by prior plan? re-verify)"),
    dict(src="E:\\research-dense\\output\\evidence_regeneration_v1",
         remote="gdrive:research-dense-cold/03_Experiment_Archives/evidence_regeneration_v1",
         size_gib=22.426, verify="rclone check --one-way per item + verification manifest",
         then="OWNER-GATED: frozen evidence; delete only with explicit waiver; frees ~22.4 GiB on E:"),
    dict(src="E:\\research-dense\\runs\\campaign_stage1_chunk1 (superseded p40 branches; WM-42/44/56)",
         remote="gdrive:research-dense-cold/03_Experiment_Archives/chunk1_p40_superseded",
         size_gib=24.2, verify="per-file sha256 already present for the archived subset; rclone check --one-way",
         then="OWNER-GATED: frozen evidence; delete only with explicit waiver; frees ~24.2 GiB on E:"),
    dict(src="E:\\research-dense\\sealed",
         remote="gdrive:research-dense-cold/04_Restricted_Archives_Do_Not_Open/sealed",
         size_gib=61.832, verify="hash OPAQUE archived bytes only (never open contents); rclone check --one-way",
         then="OWNER-GATED: acceptance data; local delete only with explicit waiver; frees ~61.8 GiB on E:"),
    # idle ARCHIVE worktree ignored output (archive unique bytes, then git worktree remove)
    dict(src="E:/rd-wt/<idle ARCHIVE worktree>/_out + ignored output (60 trees, up to 36.59 GiB)",
         remote="gdrive:research-dense-cold/03_Experiment_Archives/worktrees/<name>",
         size_gib=36.59, verify="tar/zip + member sha256, then rclone check --one-way",
         then="git worktree remove (no --force) only after archive verified; frees E: checkout bytes"),
    # D: already verified 2026-10-06
    dict(src="D:\\rd-staging\\wmn1_feat\\upload_v2", remote="gdrive:.../Archived_Training_Packs_V2/upload_v2.tar.zst",
         size_gib=6.391, verify="VERIFIED 7/7 (2026-10-06) + fresh rclone check --one-way",
         then="owner may delete locally now (script), frees 6.39 GiB on D:"),
    dict(src="D:\\rd-staging\\wmn1_feat\\upload_v2c", remote="gdrive:.../Archived_Training_Packs_V2/upload_v2c.tar.zst",
         size_gib=6.438, verify="VERIFIED 7/7 (2026-10-06) + fresh rclone check --one-way",
         then="owner may delete locally now, frees 6.44 GiB on D:"),
    dict(src="D:\\rd-staging\\wmn1_feat\\raw_chunk1_p40", remote="gdrive:.../Raw_And_Superseded/raw_chunk1_p40.tar.zst",
         size_gib=4.553, verify="VERIFIED 294/294 (2026-10-06) + fresh rclone check --one-way",
         then="owner may delete locally now, frees 4.55 GiB on D:"),
    dict(src="D:\\rd-staging\\wmn1_feat\\shards_v2\\{chunk1,chunk1_fix27,chunk2,index}",
         remote="gdrive:.../Derived_Shards_V2/", size_gib=13.5,
         verify="VERIFIED (2026-10-06) + fresh rclone check --one-way",
         then="owner may delete verified subfolders, frees 13.5 GiB on D:"),
    dict(src="D:\\rd-staging\\wmn1_feat\\shards_v2\\{chunk3,chunk4,truearms_chunk2,truearms_chunk3,walls_pilot_chunk4}",
         remote="gdrive:.../Derived_Shards_V2/", size_gib=21.0,
         verify="TRUNCATED on Drive -> re-upload then rclone check --one-way; keep local until pass",
         then="delete only after re-upload passes, frees ~21 GiB on D:"),
    dict(src="D:\\rd-staging\\wmn1_feat\\{pack_v3,shards_v3} + cpu_payload",
         remote="gdrive:research-dense-cold/02_World_Model_Data/wm_n1/", size_gib=101.1,
         verify="not on Drive today; upload + rclone check --one-way; pack_v3 also gpuA/gpuB 18/18 sha",
         then="if moving to E: instead, only delete from D: after the E: copy verifies"),
]

# ---- Delete candidates (owner-run DELETE_AFTER_VERIFY.ps1; report-only default) ----
deletes = [
    # E: regenerable / superseded / duplicate, no frozen evidence
    dict(path="E:\\research-dense\\tmp (non-ledger-cited scratch only)", size_gib=0.377,
         why="scratch; zcode_review + storage_audit are ledger-cited and KEPT",
         evidence="inventory_E.json research_dense_tmp_subfolders"),
    dict(path="E:\\research-dense\\runs\\campaign_walls_v22_smoke{,_r2}", size_gib=0.116,
         why="name-marked smoke runs", evidence="inventory_E runs_subfolders"),
    dict(path="E:\\research-dense\\runs\\campaign_stage1_chunk5.failed_attempt*", size_gib=0.002,
         why="failed attempts", evidence="inventory_E runs_subfolders"),
    dict(path="E:\\research-dense\\runs\\campaign_stage1_chunk1_failed_branches_attempt1.tar.gz", size_gib=0.008,
         why="already compressed superseded attempt", evidence="inventory_E runs_subfolders"),
    dict(path="E:\\research-dense\\tmp\\drive_verify_20261006 etc.", size_gib=0.002,
         why="verification scratch", evidence="inventory_E"),
    dict(path="E:\\rd-release\\venv-test (0.88 GiB) + venv-pptx", size_gib=0.916,
         why="regenerable virtualenvs", evidence="REORG_PLAN_V2 SAFE"),
    dict(path="E:\\rd-wt\\reorg2_*, reorg3, reorg_inv_* (13 non-git scout dirs)", size_gib=0.093,
         why="prior-plan scout scratch; regenerate on demand", evidence="inventory_E worktrees"),
    dict(path="E:\\rd-wt$w (junction alias to ws-p2/_ext)", size_gib=0.0,
         why="junction with no independent data; recreated on demand",
         evidence="inventory_E top_level reparse"),
    dict(path="E: duplicate (size+basename) groups outside sealed/", size_gib=2.15,
         why="duplicate _out tars / torch dll etc.; NOT hash-verified - hash-verify each group first",
         evidence="inventory_E duplicates.top_groups"),
    # D:
    dict(path="D:\\research-dense (stray .claude/.pytest_cache)", size_gib=0.0,
         why="retired leftovers; owner rule: should not exist", evidence="inventory_D"),
    dict(path="D:\\rd-staging\\wmn1_feat\\shards (v1, aborted)", size_gib=0.075,
         why="aborted shard-key collisions", evidence="inventory_D"),
    dict(path="D:\\rd-staging\\box_push_j4b_bases", size_gib=0.008,
         why="duplicate of E: runs/campaign_stage1_chunk4 __base dirs", evidence="inventory_D"),
    dict(path="D:\\rd-staging\\plexp1d_v3_root + plexp1d_v3_slot*.tar", size_gib=0.156,
         why="code duplicated in E:/rented_bundle/code_plexp1d", evidence="inventory_D"),
    dict(path="D:\\rd-staging\\wmn1_feat\\{logs,quarantine_v2,*}", size_gib=0.005,
         why="regenerable featurization logs", evidence="inventory_D"),
    # C:
    dict(path="C:\\Users\\jpeng\\AppData\\Local\\Temp\\claude", size_gib=4.637,
         why="agent temp; regenerable", evidence="inventory_CG c_items"),
    dict(path="C:\\Users\\jpeng\\AppData\\Local\\Temp\\pip-unpack-*", size_gib=0.127,
         why="pip temp", evidence="inventory_CG"),
    dict(path="C:\\Users\\jpeng\\AppData\\Local\\Temp\\rd_v3_*.txt, RUNS_*.jsonl, wm_run*_tail.txt",
         size_gib=0.001, why="transient logs", evidence="inventory_CG"),
    dict(path="C:\\Users\\jpeng\\Sumo, C:\\Users\\jpeng\\ZCodeProject", size_gib=0.0,
         why="empty dirs", evidence="inventory_CG"),
]

# ---- Never moved / never deleted ----
never = dict(
    live_worktrees=["E:/rd-wt/ws-main", "E:/rd-wt/ws-main2", "E:/rd-wt/ws-brakefix",
                    "E:/rd-wt/ws-trainsetup", "E:/rd-wt/ws-wallc1"],
    keep_in_place=[
        "C:/Users/jpeng/miniconda3/envs/rl_driving",
        "C:/Users/jpeng/Documents/Codex/tools/AgentControlPlane",
        "C:/Users/jpeng/.claude", "C:/Users/jpeng/.codex/worktrees",
        "E:/rd-wt/<KEEP-ACTIVE worktrees>", "E:/research-dense/sealed (contents never opened)",
        "E:/research-dense/runs/active", "E:/research-dense/runs/campaign_walls_pilot_chunk4*",
        "E:/research-dense/protocols/*, frozen evidence literals",
        "D:/rd-staging/* (until Drive-verified)", "G:/ (already Drive)",
        "/root/autodl-tmp GPU box paths",
    ],
)

out = {
    "plan": "REORG_PLAN_V3",
    "generated": datetime.datetime.now().isoformat(timespec="seconds"),
    "root": "E:/research",
    "root_decision": "use E:/research as the single umbrella; rename E:/research-dense -> E:/research/repo "
                     "with a junction at the old path. See REORG_PLAN_V3.md (decision + reasons).",
    "junction_policy": "Create a directory junction at every old path after its move and keep it "
                       "indefinitely; junctions make all old absolute literals (32,712 to "
                       "E:/research-dense, 2,328 to E:/rd-wt, 175 to E:/research-dense-backups) resolve.",
    "move_method_note": "All E: moves are same-volume renames (instant, no extra space). "
                        "D:/C: moves are cross-volume copies (robocopy) that need free space and verification.",
    "count": len(moves),
    "moves": moves,
    "drive_only": drive_only,
    "delete_candidates": deletes,
    "never_move_or_delete": never,
}

with open(os.path.join(REPO, "MOVE_MAP.json"), "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)

print("wrote MOVE_MAP.json with %d moves" % len(moves))
print("siblings: %d, worktrees: %d, D: %d, C: %d" % (
    len(siblings), len(arch_wt), len(d_moves), len(c_moves) + 1))

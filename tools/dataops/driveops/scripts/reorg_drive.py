#!/usr/bin/env python3
"""Server-side reorganisation of gdrive:research-dense-cold into gdrive:research-archive.

Rules:
- Never deletes remote data; only server-side moves (rclone moveto), each recorded.
- Records old->new in MOVE_WITHIN_DRIVE.json and appends to drive_progress.log.
- Idempotent: skips when source absent or destination already present.
"""
import json, os, subprocess, sys, datetime

RCLONE = r"C:\Users\jpeng\miniconda3\bin\rclone.exe"
ROOT = r"E:/rd-wt/reorg3/driveops"
LOG = os.path.join(ROOT, "drive_progress.log")
MOVE_JSON = os.path.join(ROOT, "MOVE_WITHIN_DRIVE.json")
SRC = "gdrive:research-dense-cold"
DST = "gdrive:research-archive"

# restricted container is intentionally NOT moved (owner decision, do not touch)
MAPPING = [
    ("03_Experiment_Archives/Campaigns", "repo-evidence/runs/campaigns"),
    ("03_Experiment_Archives/Raw_And_Superseded", "repo-evidence/runs/raw_and_superseded"),
    ("03_Experiment_Archives/Validation_And_Regeneration", "repo-evidence/runs/validation_and_regeneration"),
    ("03_Experiment_Archives/Rented_Box", "data/staging/rented_box"),
    ("02_World_Model_Data/Training_Pack_V3_K192", "data/packs/training_pack_v3_k192"),
    ("02_World_Model_Data/Archived_Training_Packs_V2", "data/packs/archived_training_packs_v2"),
    ("02_World_Model_Data/Derived_Shards_V2", "data/packs/derived_shards_v2"),
    ("02_World_Model_Data/Quarantine", "data/packs/quarantine"),
    ("02_World_Model_Data/Dataset_Exports", "datasets/exports"),
    ("01_Code_And_Restore", "code"),
    ("05_Recovery_Backups", "backups"),
    ("06_Models", "models"),
    ("90_Upload_Records", "records"),
    ("00_ARCHIVE_CATALOG_20261002_R2.json", "records/00_ARCHIVE_CATALOG_20261002_R2.json"),
    ("00_ARCHIVE_CATALOG_20261004_R3.json", "records/00_ARCHIVE_CATALOG_20261004_R3.json"),
    ("00_START_HERE_20261002_R2.txt", "records/00_START_HERE_20261002_R2.txt"),
    ("00_START_HERE_20261004_R3.txt", "records/00_START_HERE_20261004_R3.txt"),
]


def run(args, timeout=1800):
    p = subprocess.run([RCLONE] + args, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def exists(remote):
    rc, out, err = run(["lsf", remote, "--max-depth", "1"])
    return rc == 0 and out.strip() != ""


def log(msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    os.makedirs(ROOT, exist_ok=True)
    records = []
    if os.path.exists(MOVE_JSON):
        records = json.load(open(MOVE_JSON, encoding="utf-8"))
    done = {r["new"] for r in records if r.get("ok")}
    for src_rel, dst_rel in MAPPING:
        src = f"{SRC}/{src_rel}"
        dst = f"{DST}/{dst_rel}"
        if dst_rel in done:
            log(f"SKIP already-done {src_rel}")
            continue
        if not exists(src):
            log(f"SKIP source-absent {src_rel}")
            continue
        if exists(dst):
            log(f"SKIP dest-exists {dst_rel} (source kept)")
            records.append({"old": src, "new": dst, "ok": False,
                            "note": "dest already exists; source left in place"})
            continue
        rc, out, err = run(["moveto", src, dst])
        ok = rc == 0
        log(f"{'MOVED' if ok else 'FAIL'} {src_rel} -> {dst_rel} rc={rc}")
        if not ok:
            log("  stderr: " + err.strip().replace("\n", " | ")[:500])
        records.append({"old": src, "new": dst, "ok": ok,
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})
        json.dump(records, open(MOVE_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(records, open(MOVE_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"DONE reorg: {sum(1 for r in records if r.get('ok'))}/{len(MAPPING)} moved")


if __name__ == "__main__":
    main()

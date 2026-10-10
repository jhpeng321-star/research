# check_campaign_coverage.py -- before uploading big campaign/run trees, check
# whether gdrive:research-archive/repo-evidence/runs/campaigns/ or the
# records/Upload_Receipts/ listing already covers them (so we verify instead of
# re-uploading). Read-only: only lists remote names; never opens rclone.conf,
# never deletes. Junction-safe local sizing via upload_driver.measure.
import json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upload_driver as u  # noqa: E402

RUNS = r"E:\research-dense\runs"
CAMPAIGNS_REMOTE = "gdrive:research-archive/repo-evidence/runs/campaigns/"
RECEIPTS_REMOTE = "gdrive:research-archive/records/Upload_Receipts/"
OUT = os.path.join(u.WORK, "_out", "campaign_coverage.json")
TARGETS = [
    "campaign_stage1_chunk2", "campaign_stage1_chunk3", "campaign_stage1_chunk4",
    "campaign_walls_pilot_chunk4", "campaign_stage1_chunk2_truearms",
    "campaign_stage1_chunk3_truearms", "active",
]


def lsf(remote):
    try:
        r = subprocess.run([u.RCLONE, "lsf", remote, "--max-depth", "1"],
                           capture_output=True, text=True, timeout=300)
        return [x.strip().rstrip("/") for x in (r.stdout or "").splitlines() if x.strip()]
    except Exception as e:
        return ["__ERROR__:" + str(e)]


def main():
    camp = lsf(CAMPAIGNS_REMOTE)
    recs = lsf(RECEIPTS_REMOTE)
    rows = []
    for name in TARGETS:
        src = os.path.join(RUNS, name)
        n, b = u.measure(src) if os.path.isdir(src) else (0, 0)
        covered = any(x == name or x == name + ".tar.zst" for x in camp)
        receipt = any(x.startswith(name + ".tar.zst") for x in recs)
        rows.append({
            "tree": name, "local_files": n, "local_bytes": b,
            "in_campaigns_folder": covered, "has_upload_receipt": receipt,
            "action": "verify_or_skip" if (covered or receipt) else "upload",
        })
    doc = {"schema": "dataops_campaign_coverage/1",
           "campaigns_remote": CAMPAIGNS_REMOTE,
           "campaigns_listing": camp,
           "receipts_listing": recs,
           "targets": rows}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    for r in rows:
        print(r["action"], "|", r["tree"], "| %.2fGB" % (r["local_bytes"] / 1e9),
              "| campaigns=%s receipts=%s" % (r["in_campaigns_folder"], r["has_upload_receipt"]))
    print("report ->", OUT)


if __name__ == "__main__":
    main()

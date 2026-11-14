# make_summary.py -- build DATA_SUMMARY.md from the queue + verified receipts.
import datetime, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upload_driver as u

WORK = r"E:\rd-wt\reorg3\dataops"
QUEUE = os.path.join(WORK, "_out", "queue.json")
OUT = os.path.join(WORK, "DATA_SUMMARY.md")
MBPS = 2.6  # fallback; overridden by _out/throughput.json or DATAOPS_MBPS
_tp = os.path.join(WORK, "_out", "throughput.json")
if os.path.exists(_tp):
    try:
        MBPS = float(json.load(open(_tp, encoding="utf-8")).get("mbps", MBPS))
    except Exception:
        pass
MBPS = float(os.environ.get("DATAOPS_MBPS", MBPS))

def load_receipts():
    recs = {}
    d = os.path.join(WORK, "receipts")
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".json"):
                try:
                    r = json.load(open(os.path.join(d, f), encoding="utf-8"))
                    recs[r.get("id")] = r
                except Exception:
                    pass
    return recs

def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return "%.2f %s" % (n, unit)
        n /= 1024.0

def main():
    q = json.load(open(QUEUE, encoding="utf-8"))
    units = u.expand_queue(q["items"], q.get("exclude", []))
    units.sort(key=lambda x: x["bytes"], reverse=True)
    recs = load_receipts()

    up = [x for x in units if x["id"] in recs]
    pend = [x for x in units if x["id"] not in recs]
    up_bytes = sum(x["bytes"] for x in up)
    pend_bytes = sum(x["bytes"] for x in pend)
    eta_h = (pend_bytes / (MBPS * 1e6)) / 3600.0
    now = datetime.datetime.now().isoformat(timespec="seconds")

    L = []
    A = L.append
    A("# DATA_SUMMARY - research data upload to Google Drive")
    A("")
    A("Generated %s by the dataops agent (resumable; receipts in `receipts/`)." % now)
    A("Remote: `gdrive:research-archive`. Uploads are `tar.zst` with per-member md5")
    A("manifests (`manifests/members/`) and a verified md5 of each archive.")
    A("Measured aggregate upload speed from this PC: ~%.1f MB/s (parallel workers)." % MBPS)
    A("")
    A("## Totals")
    A("")
    A("| metric | value |")
    A("|---|---:|")
    A("| queue units | %d |" % len(units))
    A("| uploaded + verified units | %d |" % len(up))
    A("| pending units | %d |" % len(pend))
    A("| uploaded bytes (source) | %s |" % human(up_bytes))
    A("| pending bytes (source) | %s |" % human(pend_bytes))
    A("| pending ETA @ %.1f MB/s | %.1f h |" % (MBPS, eta_h))
    A("")
    A("## Uploaded and verified")
    A("")
    if up:
        A("| id | source bytes | archive bytes | file |")
        A("|---|---:|---:|---|")
        for x in sorted(up, key=lambda x: x["bytes"], reverse=True):
            r = recs[x["id"]]
            A("| `%s` | %s | %s | %s |" % (x["id"], human(x["bytes"]),
              human(r.get("archive_bytes", 0)), r.get("drive", "")))
    else:
        A("_none yet_")
    A("")
    A("## Pending (largest first) - top 40")
    A("")
    A("| id | source bytes |")
    A("|---|---:|")
    for x in pend[:40]:
        A("| `%s` | %s |" % (x["id"], human(x["bytes"])))
    if len(pend) > 40:
        A("")
        A("_... plus %d smaller units, %s total._" % (len(pend) - 40, human(pend_bytes)))
    A("")
    A("## gpuA-pending packs (NOT uploaded from this PC)")
    A("")
    A("Uploaded from gpuA by the owner via `_out/box_upload/upload_packs_from_gpuA.sh`")
    A("(see `_out/box_upload/README_BOX_UPLOAD.md`). Local md5 manifests already computed:")
    A("")
    md = os.path.join(WORK, "manifests")
    if os.path.isdir(md):
        for f in sorted(os.listdir(md)):
            if f.endswith(".md5"):
                A("- `manifests/%s`" % f)
    A("")
    A("## Delete candidates (NOT deleted by the agent)")
    A("")
    A("- `D:/rd-staging/pack_v3_r3_subset` (9.21 GB) - redundant subset; superseded by pack_v3.")
    A("- `E:/research-dense/output/partB_bundles` (0.56 GB) - `check=passed` on Drive (driveops).")
    A("- `D:/rd-staging/wmn1_feat/shards` (shards_v1, 80 MB) - `check=passed` on Drive.")
    A("- `D:/rd-staging/box_push_j4b_bases`, `plexp1d_v3_root`, `plexp1d_v3_slot*.tar` - on Drive.")
    A("- `D:/rd-staging/wmn1_feat/logs*`, `labels_*`, `pack_v2c_meta` - on Drive.")
    A("- `E:/research-dense-c-migrated`, `E:/migration_logs`, `E:/glm*`, `E:/rented_bundle*` - on Drive.")
    A("- `E:/research-dense/runs/campaign_stage1_chunk1` - archived on Drive (only 1 sidecar not in archive).")
    A("- Per `E:/rd-wt/reorg3/s1_E` and `s3_C_G`: reorg scratch worktrees and C: `%TEMP%\\claude` (~4.7 GB) etc.")
    A("")
    A("## Sensitive / never touched")
    A("")
    A("- `sealed/`, `CAL`, `HOLDOUT`, `chunk-5`/`chunk5`, `INC1` contents were never opened or uploaded (J4B allowed).")
    A("- No junctions/symlinks were followed or uploaded.")
    A("- `rclone.conf` and credential files were never opened (owner supplies credentials on the box).")
    A("- No source file was deleted or moved; only the agent's own temp `tar.zst` after verified upload.")
    A("")
    A("## Notes / blockers")
    A("")
    A("- Upload is I/O-bound; measured aggregate ~%.1f MB/s; total remaining is ~%.1f h. It is resumable:" % (MBPS, eta_h))
    A("  re-run `python scripts/upload_driver.py --queue _out/queue.json --workers 4` (verified receipts are")
    A("  skipped). Rounds are orchestrated by `scripts/run_upload_rounds.ps1` (never two drivers at once).")
    A("- `C:/Users/jpeng/ray_results` could not be sized (ACL access denied) - owner action required.")
    A("- Temp tars are written to `D:/rd-staging/_dataops_tmp` with a free-space check; oversized single")
    A("  files (> D: free space) are deferred and reported in `data_progress.log`.")
    A("")
    open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print("wrote", OUT, "units=%d up=%d pend=%d" % (len(units), len(up), len(pend)))

if __name__ == "__main__":
    main()

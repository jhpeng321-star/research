"""Build _out/verified_for_deletion.json from passed receipts. Read-only; never deletes.

A local path is listed only when its Drive archive passed (size+md5 via rclone md5sum
of the uploaded archive vs the local archive md5, or rclone check --one-way --checksum
for as_files copies). Split units are listed only when every part is verified. Directories
containing entries the driver intentionally skipped (sensitive names, junctions) are
flagged with skipped_entries>0 -- the cleanup script must NOT remove those paths wholesale.
"""
import datetime, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upload_driver as u

WORK = u.WORK
OUT = os.path.join(WORK, "_out", "verified_for_deletion.json")


def count_skipped(src):
    """Entries under src that the driver does not upload (sensitive / reparse)."""
    n = 0
    for root, dirs, files in os.walk(src):
        keep = []
        for d in dirs:
            p = os.path.join(root, d)
            try:
                rep = u.is_reparse(os.lstat(p))
            except OSError:
                rep = True
            if rep or u.sensitive(d):
                n += 1
            else:
                keep.append(d)
        dirs[:] = keep
        for f in files:
            p = os.path.join(root, f)
            try:
                rep = u.is_reparse(os.lstat(p))
            except OSError:
                rep = True
            if rep or u.sensitive(f):
                n += 1
    return n


def main():
    recs = {}
    d = os.path.join(WORK, "receipts")
    for f in os.listdir(d):
        if f.endswith(".json") and f != "COVERAGE.json":
            try:
                r = json.load(open(os.path.join(d, f), encoding="utf-8"))
                if r.get("check") == "passed" and r.get("id"):
                    recs[r["id"]] = r
            except Exception:
                pass
    q = json.load(open(os.path.join(WORK, "_out", "queue.json"), encoding="utf-8"))
    units = u.expand_queue(q["items"], q.get("exclude", []))
    by_src = {}
    for x in units:
        by_src.setdefault(x["src"], []).append(x)
    entries = []
    for src, us in by_src.items():
        if not all(x["id"] in recs for x in us):
            continue
        if not os.path.exists(src):
            continue
        r0 = recs[us[0]["id"]]
        e = {"path": src, "kind": us[0]["kind"], "bytes": sum(x["bytes"] for x in us),
             "drive": [recs[x["id"]]["drive"] for x in us],
             "archive_md5": [recs[x["id"]].get("archive_md5", "") for x in us],
             "check": r0.get("check_method", ""), "verified_at": max(recs[x["id"]].get("timestamp", "") for x in us)}
        if us[0]["kind"] == "dir":
            e["skipped_entries"] = count_skipped(src)
        else:
            e["skipped_entries"] = 0
        entries.append(e)
    entries.sort(key=lambda e: e["path"])
    json.dump({"schema": "verified_for_deletion/1",
               "generated": datetime.datetime.now().isoformat(timespec="seconds"),
               "note": "Local paths whose Drive copy passed verification. NEVER auto-deleted by dataops. "
                       "Entries with skipped_entries>0 contain sensitive/junction entries not uploaded: "
                       "do not remove them wholesale.",
               "total_bytes": sum(e["bytes"] for e in entries), "entries": entries},
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT, len(entries), "entries", sum(e["bytes"] for e in entries) / 1e9, "GB")


if __name__ == "__main__":
    main()

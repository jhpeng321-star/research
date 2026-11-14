# build_index.py -- rebuild gdrive_index/INDEX.json from receipts/*.json.
import datetime, glob, json, os

WORK = r"E:\rd-wt\reorg3\dataops"
REC = os.path.join(WORK, "receipts")
OUT = os.path.join(WORK, "gdrive_index", "INDEX.json")

def main():
    entries = []
    for f in sorted(glob.glob(os.path.join(REC, "*.json"))):
        try:
            r = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if r.get("schema") not in ("dataops_receipt/1", "driveops_receipt/1"):
            continue
        entries.append({
            "id": r.get("id"), "drive": r.get("drive"), "bytes": r.get("bytes"),
            "files": r.get("files"), "archive_bytes": r.get("archive_bytes"),
            "archive_md5": r.get("archive_md5"), "check": r.get("check"),
            "timestamp": r.get("timestamp"),
        })
    entries.sort(key=lambda e: (e.get("timestamp") or ""))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"schema": "research_archive_index/1",
               "generated": datetime.datetime.now().isoformat(timespec="seconds"),
               "entries": entries}, open(OUT, "w", encoding="utf-8"), indent=1)
    print("index entries=%d" % len(entries))

if __name__ == "__main__":
    main()

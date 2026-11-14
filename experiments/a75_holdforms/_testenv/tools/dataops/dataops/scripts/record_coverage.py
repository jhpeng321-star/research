# record_coverage.py -- emit receipts/COVERAGE.json documenting local trees that
# are already covered by a verified Drive archive (member list/count/bytes equal),
# plus opaque-preserved archives. Read-only; does not create fake upload receipts.
import json, os

WORK = r"E:\rd-wt\reorg3\dataops"
ALL = os.path.join(WORK, "_out", "coverage_all.json")
RUNS = os.path.join(WORK, "_out", "coverage_report.json")
OUT = os.path.join(WORK, "receipts", "COVERAGE.json")

def main():
    allr = json.load(open(ALL, encoding="utf-8"))
    verified, opaque, absent, changed = [], [], [], []
    for e in allr:
        st = e.get("status")
        row = {"receipt": e["receipt"], "source": e.get("source"),
               "archive": e.get("archive"), "archive_md5": e.get("archive_md5"),
               "files": e.get("local_files", e.get("receipt_n")),
               "bytes": e.get("local_bytes", e.get("receipt_bytes"))}
        if st == "verified_covered":
            verified.append(row)
        elif st in ("opaque_no_member_list",):
            opaque.append(row)
        elif st == "source_absent":
            absent.append(row)
        elif st == "mismatch":
            changed.append(row)
    doc = {
        "schema": "dataops_coverage/1",
        "check": "verified_covered",
        "note": ("Local trees covered by an existing gdrive:research-archive archive. "
                 "verified_covered = receipt member path set, file count and total bytes "
                 "equal the local tree (no re-download needed, nothing to upload). "
                 "opaque = archive preserved but receipt has no member list (cannot "
                 "member-verify; re-upload if the source must be provably covered). "
                 "source_absent = local tree already cleaned; archive-only. "
                 "mismatch = local tree differs from receipt; needs re-upload. "
                 "No source file was deleted or moved by this check."),
        "verified_covered": verified,
        "opaque_preserved": opaque,
        "source_absent": absent,
        "mismatch": changed,
    }
    json.dump(doc, open(OUT, "w", encoding="utf-8"), indent=1)
    print("wrote", OUT, "verified=%d opaque=%d absent=%d mismatch=%d" %
          (len(verified), len(opaque), len(absent), len(changed)))
    for r in verified:
        print("  COVERED", r["source"], r["bytes"], "B", "->", r["archive"])

if __name__ == "__main__":
    main()

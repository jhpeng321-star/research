# verify_covered.py -- verify Drive-archived runs/ trees against local copies
# using Upload_Receipts member lists (no re-download). Junction-safe; skips
# sealed/CAL/HOLDOUT/chunk-5/chunk5/INC1. Never deletes anything.
import json, os, re, stat as st_mod, sys

RUNS = r"E:\research-dense\runs"
RECEIPTS = r"E:\rd-wt\reorg3\dataops\_out\drive_receipts"
OUT = r"E:\rd-wt\reorg3\dataops\_out\coverage_report.json"
SENSITIVE = {"sealed", "cal", "holdout", "inc1"}

def is_reparse(s):
    return (getattr(s, "st_file_attributes", 0) & st_mod.FILE_ATTRIBUTE_REPARSE_POINT) != 0

def sensitive(rel):
    for p in rel.replace("\\", "/").split("/"):
        lp = p.lower()
        if lp in SENSITIVE or "chunk5" in lp or "chunk-5" in lp:
            return True
    return False

def walk(root):
    root = os.path.abspath(root)
    out = {}
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    s = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if is_reparse(s):
                    continue
                rel = os.path.relpath(e.path, root).replace(os.sep, "/")
                if sensitive(rel):
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    out[rel] = s.st_size
    return out

def main():
    receipts = {}
    for f in os.listdir(RECEIPTS):
        if not f.endswith(".json"):
            continue
        try:
            d = json.load(open(os.path.join(RECEIPTS, f), encoding="utf-8"))
        except Exception:
            continue
        src = d.get("source")
        if not src or "files_sha256" not in d or not d.get("files_sha256"):
            continue
        receipts[os.path.normcase(os.path.abspath(src))] = (f, d)

    report = {"runs_root": RUNS, "trees": []}
    for name in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, name)
        if not os.path.isdir(p) or is_reparse(os.stat(p, follow_symlinks=False)):
            continue
        if sensitive(name):
            report["trees"].append({"tree": name, "status": "skipped_sensitive"})
            continue
        files = walk(p)
        nbytes = sum(files.values())
        entry = {"tree": name, "local_files": len(files), "local_bytes": nbytes}
        rec = receipts.get(os.path.normcase(os.path.abspath(p)))
        if not rec:
            entry["status"] = "no_matching_receipt"
        else:
            rf, d = rec
            # receipt member keys are prefixed with the tree folder name
            prefix = name + "/"
            members = {}
            for k, v in d["files_sha256"].items():
                kk = k.replace("\\", "/")
                if kk.startswith(prefix):
                    members[kk[len(prefix):]] = v
            rbytes = d.get("original_bytes")
            entry.update({
                "receipt": rf, "receipt_n": d.get("n_files"),
                "receipt_bytes": rbytes, "archive": d.get("archive"),
                "archive_md5": d.get("archive_md5"),
                "member_paths": len(members),
                "n_match": (len(files) == len(members)),
                "bytes_match": (rbytes == nbytes),
                "pathset_equal": (set(files) == set(members)),
            })
            entry["status"] = ("verified_covered"
                               if entry["n_match"] and entry["bytes_match"]
                               and entry["pathset_equal"] else "mismatch")
        report["trees"].append(entry)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    for t in report["trees"]:
        print(t.get("status"), "|", t["tree"], "|", t.get("local_files"),
              "files", t.get("local_bytes"), "B", "|", t.get("receipt", ""))
    print("report ->", OUT)

if __name__ == "__main__":
    main()

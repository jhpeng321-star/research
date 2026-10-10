# verify_all_receipts.py -- for every Upload_Receipts entry, if its local source
# still exists, compare the receipt member list (path set, file count, bytes)
# against the local tree. Junction-safe; skips sealed/CAL/HOLDOUT/chunk-5/INC1.
# Read-only. Emits _out/coverage_all.json.
import json, os, stat as st_mod, sys

RECEIPTS = r"E:\rd-wt\reorg3\dataops\_out\drive_receipts"
OUT = r"E:\rd-wt\reorg3\dataops\_out\coverage_all.json"
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
    res = []
    for f in sorted(os.listdir(RECEIPTS)):
        if not f.endswith(".json"):
            continue
        d = json.load(open(os.path.join(RECEIPTS, f), encoding="utf-8"))
        src = d.get("source")
        members = d.get("files_sha256")
        e = {"receipt": f, "source": src, "archive": d.get("archive"),
             "archive_md5": d.get("archive_md5"),
             "receipt_n": d.get("n_files"), "receipt_bytes": d.get("original_bytes")}
        if not src:
            e["status"] = "receipt_has_no_source"; res.append(e); continue
        if not members:
            e["status"] = "opaque_no_member_list"; res.append(e); continue
        if not os.path.exists(src):
            e["status"] = "source_absent"; res.append(e); continue
        if os.path.isfile(src):
            e["status"] = "source_is_file"; res.append(e); continue
        files = walk(src)
        nbytes = sum(files.values())
        prefix = os.path.basename(os.path.abspath(src)) + "/"
        mem = {k.replace("\\", "/")[len(prefix):]: v for k, v in members.items()
               if k.replace("\\", "/").startswith(prefix)}
        e.update({"local_files": len(files), "local_bytes": nbytes,
                  "member_paths": len(mem),
                  "n_match": len(files) == len(mem),
                  "bytes_match": d.get("original_bytes") == nbytes,
                  "pathset_equal": set(files) == set(mem)})
        e["status"] = ("verified_covered"
                       if e["n_match"] and e["bytes_match"] and e["pathset_equal"]
                       else "mismatch")
        res.append(e)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for e in res:
        print(e["status"], "|", e["receipt"], "|", e.get("local_files"),
              e.get("local_bytes"), "|", e["source"])
    print("report ->", OUT)

if __name__ == "__main__":
    main()

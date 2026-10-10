# make_pack_manifest.py -- streaming md5 manifest for a pack directory.
# Usage: python make_pack_manifest.py --pack NAME --source DIR --box PATH --dest RCLONE_DEST --out FILE.md5
# Memory-friendly (4 MiB chunks), never follows junctions/symlinks (records them),
# refuses sensitive path components (sealed/CAL/HOLDOUT/chunk-5/chunk5/INC1).
import argparse, hashlib, os, sys, datetime

CHUNK = 4 * 1024 * 1024
SENSITIVE = {"sealed", "cal", "holdout", "inc1"}

def is_reparse(st):
    return (getattr(st, "st_file_attributes", 0) & 0x400) != 0

def sensitive(rel_parts):
    for p in rel_parts:
        lp = p.lower()
        if lp in SENSITIVE or "chunk5" in lp or "chunk-5" in lp:
            return True
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--box", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    src = os.path.abspath(a.source)
    if not os.path.isdir(src):
        print(f"source missing: {src}", file=sys.stderr); return 2
    rows, skipped_junction, skipped_sensitive = [], [], []
    total = 0
    for root, dirs, files in os.walk(src, topdown=True, followlinks=False):
        rel_root = os.path.relpath(root, src)
        keep_dirs = []
        for d in dirs:
            full = os.path.join(root, d)
            st = os.lstat(full)
            rel_parts = ([] if rel_root == "." else rel_root.split(os.sep)) + [d]
            if is_reparse(st):
                skipped_junction.append(full); continue
            if sensitive(rel_parts):
                skipped_sensitive.append(full); continue
            keep_dirs.append(d)
        dirs[:] = keep_dirs
        for f in sorted(files):
            full = os.path.join(root, f)
            st = os.lstat(full)
            if is_reparse(st):
                skipped_junction.append(full); continue
            rel = os.path.relpath(full, src).replace(os.sep, "/")
            if sensitive(rel.split("/")):
                skipped_sensitive.append(full); continue
            h = hashlib.md5()
            with open("\\\\?\\" + os.path.abspath(full), "rb") as fh:
                while True:
                    b = fh.read(CHUNK)
                    if not b: break
                    h.update(b)
            rows.append((h.hexdigest(), st.st_size, rel))
            total += st.st_size
    rows.sort(key=lambda r: r[2].encode("utf-8"))
    with open(a.out, "w", encoding="utf-8", newline="\n") as out:
        out.write("# pack_manifest/1\n")
        out.write(f"# pack={a.pack}\n")
        out.write(f"# local_source={src}\n")
        out.write(f"# box_path={a.box}\n")
        out.write(f"# drive_dest={a.dest}\n")
        out.write(f"# generated={datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        out.write(f"# files={len(rows)} bytes={total}\n")
        out.write("# format: md5 TAB size TAB relpath (LC_ALL=C sorted by relpath)\n")
        for md5, size, rel in rows:
            out.write(f"{md5}\t{size}\t{rel}\n")
    print(f"{a.pack}: files={len(rows)} bytes={total} junctions_skipped={len(skipped_junction)} sensitive_skipped={len(skipped_sensitive)}")
    for j in skipped_junction: print(f"JUNCTION-SKIPPED: {j}", file=sys.stderr)
    for s in skipped_sensitive: print(f"SENSITIVE-SKIPPED: {s}", file=sys.stderr)
    return 0

if __name__ == "__main__":
    sys.exit(main())

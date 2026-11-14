#!/usr/bin/env python3
"""Copy local items to gdrive and verify (rclone check --one-way for trees; md5 compare for loose files).

Usage:
  python backup_items.py _work/backup_items.json [name ...] [--force]
- Resumable: a passing receipt is skipped unless the item is named with --force.
- Records receipts/<name>.json and receipts/md5/<name>.md5; appends drive_progress.log.
- Hash/metadata only; files streamed in 4 MiB chunks (RAM well under 6 GB).
"""
import os, sys, json, hashlib, subprocess, datetime, stat

RCLONE = r"C:\Users\jpeng\miniconda3\bin\rclone.exe"
ROOT = r"E:/rd-wt/reorg3/driveops"
RECEIPTS = os.path.join(ROOT, "receipts")
MD5DIR = os.path.join(RECEIPTS, "md5")
LOG = os.path.join(ROOT, "drive_progress.log")


def log(msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def scan_files(root):
    """(abs_path, rel_path) regular files; never follows reparse points."""
    root = os.path.abspath(root)
    stack, out = [root], []
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    if e.is_symlink():
                        continue
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    if stat.S_ISDIR(st.st_mode):
                        stack.append(e.path)
                    elif stat.S_ISREG(st.st_mode):
                        out.append((e.path, os.path.relpath(e.path, root).replace("\\", "/")))
        except OSError:
            pass
    return sorted(out)


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rclone(args, timeout=10800):
    p = subprocess.run([RCLONE] + args, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def items_for(item):
    """Return (upload_ops, verify_ops, entries):
    upload_ops: list of (local, remote) for rclone copy/copyto
    verify_ops: list of ('tree', local_dir, remote_dir) or ('file', abs_path, remote_file)
    entries: (abs_path, remote_rel) for the md5 manifest
    """
    kind = item["kind"]
    local = item["local"]
    if kind == "dir":
        d = os.path.abspath(local)
        entries = scan_files(d)
        return [("copy", d, item["dest"])], [("tree", d, item["dest"])], entries
    if kind == "dirs":
        ups, vers, ents = [], [], []
        for src in local:
            d = os.path.abspath(src)
            base = os.path.basename(d.rstrip("/\\"))
            rd = item["dest"].rstrip("/") + "/" + base
            ups.append(("copy", d, rd))
            vers.append(("tree", d, rd))
            for abspath, rel in scan_files(d):
                ents.append((abspath, base + "/" + rel))
        return ups, vers, sorted(ents)
    if kind in ("file", "files"):
        files = [local] if kind == "file" else list(local)
        ups, vers, ents = [], [], []
        for fp in files:
            a = os.path.abspath(fp)
            base = os.path.basename(a)
            rf = item["dest"].rstrip("/") + "/" + base
            ups.append(("copyto", a, rf))
            vers.append(("file", a, rf))
            ents.append((a, base))
        return ups, vers, ents
    raise ValueError("unknown kind " + kind)


def main():
    os.makedirs(RECEIPTS, exist_ok=True)
    os.makedirs(MD5DIR, exist_ok=True)
    manifest = json.load(open(sys.argv[1], encoding="utf-8"))
    args = [a for a in sys.argv[2:] if a != "--force"]
    force = "--force" in sys.argv[2:]
    wanted = set(args)
    for item in sorted(manifest, key=lambda x: x.get("priority", 99)):
        name = item["name"]
        rpath = os.path.join(RECEIPTS, name + ".json")
        if wanted and name not in wanted:
            continue
        if os.path.exists(rpath) and not force:
            if json.load(open(rpath, encoding="utf-8")).get("check") == "passed":
                log(f"SKIP {name} (receipt passed)")
                continue
        ups, vers, entries = items_for(item)
        # md5 manifest
        mpath = os.path.join(MD5DIR, name + ".md5")
        total = 0
        lines = []
        for abspath, rel in entries:
            sz = os.path.getsize(abspath)
            total += sz
            lines.append(f"{md5_of(abspath)}  {sz}  {rel}")
        with open(mpath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        nfiles = len(entries)
        check = "passed"
        detail = ""
        try:
            for op, a, b in ups:
                if op == "copy":
                    rc, out, err = rclone(["copy", a, b, "--transfers", "2", "--checkers", "4",
                                           "--drive-chunk-size", "64M", "--stats", "0"])
                else:
                    rc, out, err = rclone(["copyto", a, b, "--transfers", "2", "--checkers", "4",
                                           "--drive-chunk-size", "64M", "--stats", "0"])
                if rc != 0:
                    raise RuntimeError(f"{op} failed rc={rc}: {err.strip()[:400]}")
            for vop, a, b in vers:
                if vop == "tree":
                    rc, out, err = rclone(["check", "--one-way", "--checkers", "4", a, b])
                else:
                    rc, out, err = rclone(["md5sum", b])
                    got = out.strip().split()[0] if out.strip() else ""
                    want = md5_of(a)
                    if got != want:
                        raise RuntimeError(f"md5 mismatch {b}: remote={got} local={want}")
                    rc = 0
                if rc != 0:
                    check = "failed"
                    detail = (err or out).strip()[:1500]
                    break
        except Exception as ex:
            check = "failed"
            detail = str(ex)[:1500]
        rec = {"schema": "driveops_receipt/1", "name": name, "kind": item["kind"],
               "class": item.get("class"), "producer": item.get("producer"),
               "restore": item.get("restore"),
               "local": item["local"], "drive": item["dest"],
               "bytes": total, "files": nfiles,
               "md5_manifest": os.path.relpath(mpath, ROOT).replace("\\", "/"),
               "check": check,
               "check_method": "rclone check --one-way (size+MD5)" if item["kind"] in ("dir", "dirs") else "rclone md5sum vs local MD5",
               "check_detail": detail,
               "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
        json.dump(rec, open(rpath, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        log(f"{'OK' if check=='passed' else 'CHECK-FAILED'} {name}: {nfiles} files, {total} bytes, check={check}")


if __name__ == "__main__":
    main()

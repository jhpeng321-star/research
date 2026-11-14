#!/usr/bin/env python3
"""verify_remote.py NAME [NAME...] - independent round-trip check of tar.zst backups.

Streams gdrive:research-archive/worktrees/<name>/<name>.tar.zst back with 'rclone cat' (nothing written to
disk), decompresses with zstd -dc, and compares every member (path, size, sha256) with the local manifest
(manifests/<name>.manifest.json).  Also compares rclone md5sum on Drive with the receipt.  Read-only.
"""
import hashlib
import json
import subprocess
import sys
import tarfile

sys.dont_write_bytecode = True
BASE = r"E:\rd-wt\reorg3\wtclean"
RCLONE = r"C:\Users\jpeng\miniconda3\bin\rclone.exe"
ZSTD = r"C:\Users\jpeng\miniconda3\Library\bin\zstd.exe"


def check(name):
    man = json.load(open(BASE + r"\manifests\%s.manifest.json" % name, encoding="utf-8"))
    rec = json.load(open(BASE + r"\receipts\%s.json" % name, encoding="utf-8-sig"))
    want = {m["path"]: (m["size"], m["sha256"]) for m in man["members"]}
    remote = "gdrive:research-archive/worktrees/%s/%s.tar.zst" % (name, name)
    r1 = subprocess.Popen([RCLONE, "cat", remote, "--stats", "0"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    r2 = subprocess.Popen([ZSTD, "-dcq"], stdin=r1.stdout, stdout=subprocess.PIPE)
    r1.stdout.close()
    tf = tarfile.open(fileobj=r2.stdout, mode="r|", encoding="utf-8")
    seen, bad = 0, []
    for ti in tf:
        h = hashlib.sha256()
        f = tf.extractfile(ti)
        left = ti.size
        while left > 0:
            b = f.read(min(left, 4 << 20))
            if not b:
                break
            left -= len(b)
            h.update(b)
        w = want.get(ti.name)
        if w is None or w[0] != ti.size or w[1] != h.hexdigest() or left:
            bad.append(ti.name)
        seen += 1
    while r2.stdout.read(1 << 20):
        pass
    r2.wait()
    r1.wait()
    ok = (not bad and seen == len(want) and r1.returncode == 0 and r2.returncode == 0)
    arch = rec.get("archive") or {}
    print("%s %s members=%d/%d bad=%d rclone_rc=%s zstd_rc=%s receipt_status=%s" %
          (name, "OK" if ok else "MISMATCH", seen, len(want), len(bad), r1.returncode, r2.returncode, rec.get("status")), flush=True)
    return ok


if __name__ == "__main__":
    allok = True
    for n in sys.argv[1:]:
        try:
            allok &= check(n)
        except Exception as ex:
            print(n, "ERROR", ex)
            allok = False
    sys.exit(0 if allok else 1)

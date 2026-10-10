# push_index.py -- upload the local aggregate gdrive_index/INDEX.json to
# gdrive:research-archive/INDEX.json and verify by md5. Does not delete remote.
import json, os, subprocess, sys, datetime

RCLONE = r"C:\Users\jpeng\miniconda3\bin\rclone.exe"
LOCAL = r"E:\rd-wt\reorg3\dataops\gdrive_index\INDEX.json"
REMOTE = "gdrive:research-archive/INDEX.json"

def rclone(args):
    r = subprocess.run([RCLONE] + args, capture_output=True, text=True)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()

def md5(path):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def main():
    if not os.path.exists(LOCAL):
        print("no local index", file=sys.stderr); return 2
    local_md5 = md5(LOCAL)
    rc, out, err = rclone(["copyto", LOCAL, REMOTE, "--stats", "0"])
    if rc != 0:
        print("copyto failed:", err[-400:], file=sys.stderr); return 1
    rc2, out2, err2 = rclone(["md5sum", REMOTE, "--stats", "0"])
    got = out2.split()[0] if out2 else ""
    ok = (rc2 == 0 and got == local_md5)
    print("index push", "OK" if ok else "FAIL", "local=%s remote=%s" % (local_md5, got))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())

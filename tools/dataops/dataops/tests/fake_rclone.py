"""Fake rclone for offline unit/integration tests of upload_driver.py.

Emulates copyto / copy / md5sum / check against a local FAKE_DRIVE_ROOT by
stripping the "fake:" remote prefix. Never touches a real remote."""
import hashlib, os, shutil, sys

ROOT = os.environ.get("FAKE_DRIVE_ROOT", "")
LOG = os.environ.get("FAKE_LOG", "")
VALUE_FLAGS = {"--transfers", "--checkers", "--drive-chunk-size", "--stats",
               "--log-level", "--files-from"}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def walk(root):
    out = []
    for d, _, fs in os.walk(root):
        for f in fs:
            out.append(os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/"))
    return sorted(out)


def mapdest(dest):
    if ":" in dest:
        remote, _, path = dest.partition(":")
        return os.path.join(ROOT, path.replace("/", os.sep).lstrip("/\\"))
    return os.path.join(ROOT, dest.replace("/", os.sep).lstrip("/\\"))


def parse(argv):
    pos, i = [], 0
    while i < len(argv):
        t = argv[i]
        if t in VALUE_FLAGS:
            i += 2
            continue
        if t.startswith("--"):
            i += 1
            continue
        pos.append(t)
        i += 1
    return pos


def main():
    argv = sys.argv[1:]
    if not argv:
        return 2
    cmd, pos = argv[0], parse(argv[1:])
    if LOG and cmd in ("copyto", "copy"):
        with open(LOG, "a", encoding="utf-8") as f:
            f.write("%s\t%s\n" % (cmd, "\t".join(pos)))
    if cmd == "copyto":
        src, dest = pos[0], mapdest(pos[1])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(src, dest)
        return 0
    if cmd == "md5sum":
        dest = mapdest(pos[0])
        print("%s  %s" % (md5(dest), os.path.basename(dest)))
        return 0
    if cmd == "copy":
        src, dest = pos[0], mapdest(pos[1])
        for rel in walk(src):
            dp = os.path.join(dest, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            shutil.copyfile(os.path.join(src, rel.replace("/", os.sep)), dp)
        return 0
    if cmd == "check":
        src, dest = pos[0], mapdest(pos[1])
        ok = True
        for rel in walk(src):
            sp = os.path.join(src, rel.replace("/", os.sep))
            dp = os.path.join(dest, rel.replace("/", os.sep))
            if not os.path.exists(dp) or md5(sp) != md5(dp):
                ok = False
                print("DIFF", rel)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

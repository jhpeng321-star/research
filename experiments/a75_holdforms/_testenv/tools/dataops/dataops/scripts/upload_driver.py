# upload_driver.py -- resumable, junction-safe tar.zst uploader to Google Drive.
#
# For each queue unit: build member md5 manifest, tar.zst from a filtered file
# list (regular files only; junctions/symlinks and sensitive dirs skipped),
# upload with rclone, verify by md5, write a receipt, append INDEX entries and
# delete the temp tar only after verification. Never deletes/moves sources.
#
# Parallelism: --workers N (default 4) processes N queue items concurrently
# (compress + upload + verify per item). Per-item lock files keep any item from
# being processed twice, even by a second driver instance. Temp-tar space on D:
# is reserved across workers before compressing so concurrent tars cannot
# overrun the disk. Directories larger than --split-bytes are split into several
# tar.zst parts (per sub-folder, or file-list chunks for leaf dirs) so parts can
# upload in parallel. Directory items flagged as_files are uploaded as many
# files with --transfers 8 --checkers 8.
#
# Usage: python upload_driver.py --queue QUEUE.json [--workers 4] [--limit N]
#        [--max-seconds N] [--split-bytes N] [--zstd-threads N]
import argparse, datetime, hashlib, json, os, re, shutil, stat as st_mod
import subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

RCLONE = os.environ.get("DATAOPS_RCLONE", r"C:\Users\jpeng\miniconda3\bin\rclone.exe")
ZSTD = os.environ.get("DATAOPS_ZSTD", r"C:\Users\jpeng\miniconda3\Library\bin\zstd.exe")
WORK = os.environ.get("DATAOPS_WORK", r"E:\rd-wt\reorg3\dataops")
TMPDIR = os.environ.get("DATAOPS_TMPDIR", r"D:\rd-staging\_dataops_tmp")
RECEIPTS = os.path.join(WORK, "receipts")
LOCKS = os.path.join(RECEIPTS, "locks")
MANIFEST_MEMBERS = os.path.join(WORK, "manifests", "members")
PROGRESS = os.path.join(WORK, "data_progress.log")
INDEX_LOCAL = os.path.join(WORK, "gdrive_index", "INDEX.json")
SPLIT_CAP = int(os.environ.get("DATAOPS_SPLIT", 4 * 1024**3))  # split dirs larger than this
FREE_MARGIN = 2 * 1024**3
RESERVE_FRAC = float(os.environ.get("DATAOPS_RESERVE_FRAC", "1.0"))  # temp-tar reservation as a fraction of source bytes
DRIVE_CHUNK = os.environ.get("DATAOPS_DRIVE_CHUNK", "256M")
DIR_TRANSFERS = "8"
DIR_CHECKERS = "8"
SENSITIVE = {"sealed", "cal", "holdout", "inc1"}

_LOG_LOCK = threading.Lock()


def log(msg):
    line = "[%s] [pid=%d] %s" % (datetime.datetime.now().isoformat(timespec="seconds"),
                                 os.getpid(), msg)
    with _LOG_LOCK:
        print(line, flush=True)
        with open(PROGRESS, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def is_reparse(st):
    return (getattr(st, "st_file_attributes", 0) & st_mod.FILE_ATTRIBUTE_REPARSE_POINT) != 0


def sensitive(rel):
    for p in rel.replace("\\", "/").split("/"):
        lp = p.lower()
        if lp in SENSITIVE or "chunk5" in lp or "chunk-5" in lp:
            return True
    return False


def safe_id(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def walk_files(root):
    """Yield (rel, abspath, size) for regular files; skip junctions/sensitive."""
    root = os.path.abspath(root)
    out = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in sorted(it, key=lambda x: x.name):
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
                    out.append((rel, e.path, s.st_size))
    out.sort(key=lambda r: r[0].encode("utf-8"))
    return out


def measure(root):
    files = walk_files(root) if os.path.isdir(root) else []
    return len(files), sum(f[2] for f in files)


def hash_file(path, chunk=4 * 1024 * 1024):
    h = hashlib.md5()
    with open("\\\\?\\" + os.path.abspath(path), "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def file_parts(files, split_bytes):
    """Deterministically split a sorted file list into <= split_bytes chunks.

    Returns a list of (start, end) index pairs; the slice files[start:end] is a
    part. Kept pure so the same boundaries can be re-derived at run time."""
    parts = []
    start = 0
    b = 0
    for i, f in enumerate(files):
        if b and b + f[2] > split_bytes:
            parts.append((start, i))
            start = i
            b = 0
        b += f[2]
    if start < len(files):
        parts.append((start, len(files)))
    return parts


# ---------------------------------------------------------------------------
# per-item claiming (prevents double processing across workers/instances)
# ---------------------------------------------------------------------------
def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return True


def claim(uid, owner):
    os.makedirs(LOCKS, exist_ok=True)
    p = os.path.join(LOCKS, safe_id(uid) + ".lock")
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, (owner + "\n").encode("utf-8"))
            os.close(fd)
            return True
        except FileExistsError:
            pid = None
            try:
                with open(p, encoding="utf-8") as f:
                    pid = int(f.read().split(":")[0].strip())
            except Exception:
                pid = None
            if pid is not None and _pid_alive(pid):
                return False
            try:
                os.remove(p)
            except OSError:
                return False
    return False


def release_claim(uid):
    try:
        os.remove(os.path.join(LOCKS, safe_id(uid) + ".lock"))
    except OSError:
        pass


# ---------------------------------------------------------------------------
# cross-worker temp-tar space budget on TMPDIR (D:)
# ---------------------------------------------------------------------------
class SpaceBudget:
    def __init__(self, margin):
        self._lock = threading.Lock()
        self._reserved = 0
        self._margin = margin

    def try_reserve(self, size):
        with self._lock:
            free = shutil.disk_usage(TMPDIR).free
            if self._reserved + size + self._margin > free:
                return False
            self._reserved += size
            return True

    def release(self, size):
        with self._lock:
            self._reserved = max(0, self._reserved - size)

    @property
    def reserved(self):
        with self._lock:
            return self._reserved


# ---------------------------------------------------------------------------
# queue expansion / splitting
# ---------------------------------------------------------------------------
def _child_unit(parent, e, is_dir):
    return {"id": parent["id"] + "/" + e.name, "name": e.name, "src": e.path,
            "dest": (parent["dest"].rstrip("/") + "/" + e.name) if is_dir else parent["dest"].rstrip("/"),
            "class": parent.get("class", ""), "note": parent.get("note", ""),
            "prio": parent.get("prio", 0), "as_files": parent.get("as_files", False)}


def expand_queue(items, excludes=(), split_bytes=None):
    if split_bytes is None:
        split_bytes = SPLIT_CAP
    excludes = set(os.path.abspath(x) for x in excludes)
    units = []
    for it in items:
        src = it["src"]
        if not os.path.exists(src):
            log("SKIP-missing %s (%s)" % (it["id"], src))
            continue
        if os.path.abspath(src) in excludes:
            log("SKIP-excluded %s (%s)" % (it["id"], src))
            continue
        if os.path.isdir(src):
            files = walk_files(src)
            b = sum(f[2] for f in files)
            n = len(files)
            if b <= split_bytes or n == 0:
                units.append(dict(it, kind="dir", bytes=b, nfiles=n))
                continue
            subdirs, plain = [], []
            try:
                entries = sorted(os.scandir(src), key=lambda x: x.name)
            except OSError:
                entries = []
            for e in entries:
                try:
                    s = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if is_reparse(s) or sensitive(e.name):
                    continue
                if e.is_dir(follow_symlinks=False):
                    subdirs.append(e)
                elif e.is_file(follow_symlinks=False):
                    plain.append(e)
            if subdirs:
                for e in subdirs:
                    units.extend(expand_queue([_child_unit(it, e, True)], excludes, split_bytes))
                for e in plain:
                    try:
                        sz = e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
                    cu = _child_unit(it, e, False)
                    cu.update({"kind": "file", "bytes": sz, "nfiles": 1})
                    units.append(cu)
            else:
                parts = file_parts(files, split_bytes)
                if len(parts) <= 1:
                    units.append(dict(it, kind="dir", bytes=b, nfiles=n))
                    continue
                for i in range(len(parts)):
                    s, e = parts[i]
                    cu = dict(it)
                    cu.update({"kind": "dir", "id": it["id"] + "/part%03d" % i,
                               "name": (it.get("name") or "part") + "_part%03d" % i,
                               "_part": i, "_parts": len(parts), "_split_bytes": split_bytes,
                               "bytes": sum(f[2] for f in files[s:e]), "nfiles": e - s})
                    units.append(cu)
        else:
            units.append(dict(it, kind="file", bytes=os.path.getsize(src), nfiles=1))
    return units


def select_files(u):
    """Resolve the concrete file list for a unit (re-deriving part slices)."""
    if u["kind"] != "dir":
        return [(os.path.basename(u["src"]), u["src"], u["bytes"])]
    files = walk_files(u["src"])
    if "_part" in u:
        parts = file_parts(files, u.get("_split_bytes", SPLIT_CAP))
        i = u["_part"]
        if i < len(parts):
            s, e = parts[i]
            files = files[s:e]
    return files


# ---------------------------------------------------------------------------
# compression / rclone
# ---------------------------------------------------------------------------
def write_member_manifest(uid, files):
    os.makedirs(MANIFEST_MEMBERS, exist_ok=True)
    p = os.path.join(MANIFEST_MEMBERS, safe_id(uid) + ".md5")
    total = 0
    with open(p, "w", encoding="utf-8", newline="\n") as out:
        out.write("# member_manifest/1\n# id=%s\n# generated=%s\n" %
                  (uid, datetime.datetime.now(datetime.timezone.utc).isoformat()))
        for rel, abspath, size in files:
            out.write("%s\t%d\t%s\n" % (hash_file(abspath), size, rel))
            total += size
        out.write("# files=%d bytes=%d\n" % (len(files), total))
    return p, total


def make_tar_zst(files, archive, arcbase="", zstd_threads=2):
    """Stream a PAX tar of files into zstd. arcbase is a prefix for arcnames."""
    proc = subprocess.Popen([ZSTD, "-q", "-T%d" % max(1, zstd_threads), "-o", archive, "-"],
                            stdin=subprocess.PIPE)
    import tarfile
    with tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT) as tf:
        for rel, abspath, size in files:
            arc = (arcbase + "/" + rel) if arcbase else rel
            ti = tf.gettarinfo(abspath, arcname=arc)
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.mtime = int(ti.mtime)
            with open("\\\\?\\" + os.path.abspath(abspath), "rb") as fh:
                tf.addfile(ti, fh)
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("zstd exited %d" % rc)
    return os.path.getsize(archive)


def rclone(args, capture=True):
    r = subprocess.run([RCLONE] + args, capture_output=capture, text=True)
    return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()


def receipt_path(uid):
    return os.path.join(RECEIPTS, safe_id(uid) + ".json")


def load_receipts():
    ok = set()
    if os.path.isdir(RECEIPTS):
        for f in os.listdir(RECEIPTS):
            if f.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(RECEIPTS, f), encoding="utf-8"))
                    if d.get("check") == "passed":
                        ok.add(d.get("id"))
                        ok.add(safe_id(d.get("id") or ""))
                except Exception:
                    pass
    return ok


# ---------------------------------------------------------------------------
# per-item processing
# ---------------------------------------------------------------------------
def _write_receipt_and_index(u, uid, files, dest_uri, total, asize, amd5, mpath, usecs,
                             method, index_lock, index_entries, name, ts):
    rec = {
        "schema": "dataops_receipt/1", "id": u["id"], "name": name, "kind": u["kind"],
        "class": u.get("class", ""), "producer": u.get("note", ""),
        "local": u["src"], "drive": dest_uri,
        "bytes": total, "files": len(files), "archive_bytes": asize,
        "archive_md5": amd5, "member_manifest": (os.path.relpath(mpath, WORK).replace("\\", "/")
                                                 if mpath else None),
        "check": "passed", "check_method": method,
        "upload_seconds": round(usecs, 1), "timestamp": ts,
    }
    with open(receipt_path(uid), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    with index_lock:
        index_entries.append({
            "id": u["id"], "drive": rec["drive"], "bytes": total, "files": len(files),
            "archive_bytes": asize, "archive_md5": amd5, "check": "passed", "timestamp": ts,
        })
    return rec


def process_unit(u, args, budget, done, index_lock, index_entries):
    uid = safe_id(u["id"])
    if uid in done:
        return ("skip-done", uid, 0, 0.0)
    if args.deadline and time.time() > args.deadline:
        return ("budget", uid, 0, 0.0)
    if not claim(uid, args.owner):
        return ("skip-claimed", uid, 0, 0.0)
    try:
        if uid in done:
            return ("skip-done", uid, 0, 0.0)
        name = u.get("name") or os.path.basename(u["src"].rstrip("\\/")) or u["id"].split("/")[-1]
        files = select_files(u)
        if not files:
            log("SKIP-empty %s" % u["id"])
            return ("skip-empty", uid, 0, 0.0)
        total = sum(f[2] for f in files)
        ts = datetime.datetime.now().isoformat(timespec="seconds")

        if u["kind"] == "dir" and u.get("as_files"):
            dest_dir = "%s:%s" % (args.remote, u["dest"].rstrip("/"))
            t_up = time.time()
            rc, out, err = rclone(["copy", u["src"], dest_dir,
                                   "--transfers", DIR_TRANSFERS, "--checkers", DIR_CHECKERS,
                                   "--drive-chunk-size", DRIVE_CHUNK, "--stats", "0"])
            if rc != 0:
                log("FAIL-copy %s rc=%d %s" % (u["id"], rc, err[-400:]))
                return ("fail", uid, total, 0.0)
            rc2, out2, err2 = rclone(["check", u["src"], dest_dir, "--one-way", "--checksum",
                                      "--transfers", DIR_TRANSFERS, "--checkers", DIR_CHECKERS,
                                      "--stats", "0"])
            if rc2 != 0:
                log("FAIL-verify %s rc=%d %s" % (u["id"], rc2, err2[-300:]))
                return ("fail", uid, total, 0.0)
            usecs = time.time() - t_up
            _write_receipt_and_index(u, uid, files, dest_dir, total, total, "", None, usecs,
                                     "rclone check --one-way --checksum", index_lock,
                                     index_entries, name, ts)
            log("OK %s verified (files); %.1f MB/s" % (u["id"], (total / 1e6) / max(usecs, 0.001)))
            return ("ok", uid, total, usecs)

        if not budget.try_reserve(int(total * RESERVE_FRAC)):
            log("DEFER %s bytes=%d reserved=%d (temp tar would not fit on D:)" %
                (u["id"], total, budget.reserved))
            return ("defer", uid, total, 0.0)
        try:
            mpath, _ = write_member_manifest(uid, files)
            archive = os.path.join(TMPDIR, uid + ".tar.zst")
            if os.path.exists(archive):
                log("REBUILD %s: stale temp removed" % u["id"])
                try:
                    os.remove(archive)
                except OSError:
                    pass
            t_comp = time.time()
            try:
                asize = make_tar_zst(files, archive, arcbase=u.get("arcbase", ""),
                                     zstd_threads=args.zstd_threads)
            except Exception as e:
                log("FAIL-compress %s: %s" % (u["id"], e))
                return ("fail", uid, total, 0.0)
            amd5 = hash_file(archive)
            csecs = time.time() - t_comp
            arcname = safe_id(name) + ".tar.zst"
            dest_uri = "%s:%s/%s" % (args.remote, u["dest"].rstrip("/"), arcname)
            log("UPLOAD %s bytes=%d zst=%d compress=%.0fs -> %s" %
                (u["id"], total, asize, csecs, dest_uri))

            t_up = time.time()
            rc, out, err = rclone(["copyto", archive, dest_uri,
                                   "--transfers", DIR_TRANSFERS, "--checkers", DIR_CHECKERS,
                                   "--drive-chunk-size", DRIVE_CHUNK, "--stats", "0"])
            if rc != 0:
                log("FAIL-copy %s rc=%d %s" % (u["id"], rc, err[-400:]))
                return ("fail", uid, total, 0.0)
            rc2, out2, err2 = rclone(["md5sum", dest_uri, "--stats", "0"])
            got = out2.split()[0] if out2 else ""
            if rc2 != 0 or got != amd5:
                log("FAIL-verify %s rc=%d local=%s remote=%s %s" %
                    (u["id"], rc2, amd5, got, err2[-300:]))
                return ("fail", uid, total, 0.0)
            usecs = time.time() - t_up

            _write_receipt_and_index(u, uid, files, dest_uri, total, asize, amd5, mpath,
                                     usecs, "rclone md5sum vs local archive md5",
                                     index_lock, index_entries, name, ts)
            try:
                os.remove(archive)
                log("CLEANED temp %s.tar.zst" % uid)
            except OSError:
                pass
            log("OK %s verified; wire=%.1f MB/s" %
                (u["id"], (asize / 1e6) / max(usecs, 0.001)))
            return ("ok", uid, total, usecs)
        finally:
            budget.release(int(total * RESERVE_FRAC))
    finally:
        release_claim(uid)


def _load_index():
    if os.path.exists(INDEX_LOCAL):
        try:
            return json.load(open(INDEX_LOCAL, encoding="utf-8")).get("entries", [])
        except Exception:
            return []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--max-seconds", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split-bytes", type=int, default=SPLIT_CAP)
    ap.add_argument("--zstd-threads", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(TMPDIR, exist_ok=True)
    os.makedirs(RECEIPTS, exist_ok=True)
    os.makedirs(LOCKS, exist_ok=True)
    os.makedirs(MANIFEST_MEMBERS, exist_ok=True)
    os.makedirs(os.path.dirname(INDEX_LOCAL), exist_ok=True)

    q = json.load(open(a.queue, encoding="utf-8"))
    a.remote = q.get("remote", "gdrive")
    a.owner = "%d:%s" % (os.getpid(), datetime.datetime.now().isoformat(timespec="seconds"))
    t0 = time.time()
    a.deadline = (t0 + a.max_seconds) if a.max_seconds else 0
    cpu = os.cpu_count() or 4
    if not a.zstd_threads:
        a.zstd_threads = max(1, min(4, cpu // max(1, a.workers)))

    units = expand_queue(q["items"], q.get("exclude", []), a.split_bytes)
    # Small high-value items first (prio 0, smallest first), then larger runs.
    units.sort(key=lambda u: (u.get("prio", 0), u["bytes"]))
    done = load_receipts()
    if a.limit:
        remaining = [u for u in units if safe_id(u["id"]) not in done]
        units = remaining[:a.limit]
    log("QUEUE loaded: %d units; %d already verified; workers=%d split=%.1fGB zstd_t=%d" %
        (len(units), len(done), a.workers, a.split_bytes / 1e9, a.zstd_threads))

    budget = SpaceBudget(FREE_MARGIN)
    index_lock = threading.Lock()
    index_entries = _load_index()
    counts = {}
    counts_lock = threading.Lock()

    def run(u):
        try:
            st, uid, b, secs = process_unit(u, a, budget, done, index_lock, index_entries)
        except Exception as e:
            log("FAIL-worker %s: %s" % (u.get("id"), e))
            st, uid, b, secs = ("fail", safe_id(u.get("id", "")), 0, 0.0)
        with counts_lock:
            counts[st] = counts.get(st, 0) + 1
        return st, uid, b, secs

    processed = 0
    n = max(1, a.workers)
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(run, u) for u in units]
        for fut in as_completed(futs):
            st, uid, b, secs = fut.result()
            if st == "ok":
                processed += 1

    with open(INDEX_LOCAL, "w", encoding="utf-8") as f:
        json.dump({"schema": "research_archive_index/1",
                   "generated": datetime.datetime.now().isoformat(timespec="seconds"),
                   "entries": index_entries}, f, indent=1)
    log("DRIVER done processed=%d counts=%s" % (processed, json.dumps(counts, sort_keys=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

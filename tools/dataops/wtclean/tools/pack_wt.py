#!/usr/bin/env python3
"""pack_wt.py - back up ONE retired worktree's non-tracked files to Google Drive as a single tar.zst.

Usage (python -I):  pack_wt.py NAME [--dry-run] [--tmp-pref D|E]

Never deletes/moves any worktree or source file (it only deletes its OWN temp archive after a verified
upload).  Never opens rclone.conf.  Sensitive-token paths (sealed/cal/holdout/chunk5/inc1) are never
opened, hashed or archived; they are recorded by name only.  Junctions/symlinks are never followed.
Tracked uncommitted changes are committed on the worktree's own branch (detached HEAD: preserved via a
ref refs/heads/wip-preserve/NAME in the main repo), exactly like the earlier per-file uploader did.

Steps: commit -> enumerate (git ls-files --others [+ --ignored]) -> tar|zstd -T0 -10 (sha256 per member)
-> local verify (zstd -dc | tar -t == manifest; streamed sha256 == manifest) -> rclone copyto
(--drive-chunk-size 256M) archive + manifest -> rclone md5sum/lsjson verify -> receipt.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time

sys.dont_write_bytecode = True

BASE = r"E:\rd-wt\reorg3\wtclean"
ROOT = r"E:\rd-wt"
MAIN = r"E:\research-dense"
RCLONE = r"C:\Users\jpeng\miniconda3\bin\rclone.exe"
ZSTD = r"C:\Users\jpeng\miniconda3\Library\bin\zstd.exe"
GTAR = r"C:\Program Files\Git\usr\bin\tar.exe"
REMOTE_ROOT = "gdrive:research-archive/worktrees"
TMP_ROOTS = {"D": r"D:\rd-staging\_wtclean_tmp", "E": BASE + r"\_tmp"}
MARGIN = 1_500_000_000  # keep this much free beyond size*1.2
COMMIT_MSG = "wip: preserve before worktree retirement"
SENS_RE = re.compile(r"(?i)(^|[\\/_.\-\s])(sealed|holdout|chunk-?5|inc1|cal)([\\/_.\-\s]|$)")  # j4b allowed
GIT_RE = re.compile(r"(^|/)\.git(/|$)")
PROGRESS = BASE + r"\wt_progress.log"
INV = BASE + r"\inventory\inventory.json"
LOGDIR = BASE + r"\inventory\logs"
LOCKDIR = BASE + r"\locks"
MANDIR = BASE + r"\manifests"
LISTDIR = BASE + r"\lists"
RECDIR = BASE + r"\receipts"
for _d in (LOGDIR, LOCKDIR, MANDIR, LISTDIR, RECDIR):
    os.makedirs(_d, exist_ok=True)


def lp(p):
    p = os.path.normpath(p)
    return p if p.startswith("\\\\?\\") else "\\\\?\\" + p


def now_iso():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def log(msg):
    line = "%s   %s" % (now_iso(), msg)
    for _ in range(5):
        try:
            with open(PROGRESS, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            break
        except OSError:
            time.sleep(0.5)
    try:
        print(line, flush=True)
    except Exception:  # parent pipe gone (queue restarted): never let logging kill a pack
        pass


def run(cmd, input=None, timeout=3600, cwd=None, env=None):
    r = subprocess.run(cmd, input=input, capture_output=True, timeout=timeout, cwd=cwd, env=env)
    return r.returncode, r.stdout, r.stderr


def git(wt, *args, input=None, timeout=3600, env=None):
    return run(["git", "-C", wt] + list(args), input=input, timeout=timeout, env=env)


def dec(b):
    return b.decode("utf-8", errors="replace")


def is_reparse(path):
    try:
        st = os.lstat(lp(path))
    except OSError:
        return None
    return bool(getattr(st, "st_file_attributes", 0) & 0x400) or os.path.islink(path)


def sha256_md5_file(path, bs=8 << 20):
    h2, h5 = hashlib.sha256(), hashlib.md5()
    with open(lp(path), "rb") as f:
        while True:
            b = f.read(bs)
            if not b:
                break
            h2.update(b)
            h5.update(b)
    return h2.hexdigest(), h5.hexdigest()


# ------------------------------------------------------------------ locking / temp space
def pid_alive(pid):
    try:
        r = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH"], capture_output=True, text=True, timeout=30)
        return str(pid) in r.stdout
    except Exception:
        return True


def take_lock(name):
    p = os.path.join(LOCKDIR, name + ".lock")
    for _ in range(2):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"pid": os.getpid(), "ts": now_iso()}).encode())
            os.close(fd)
            return p
        except FileExistsError:
            try:
                d = json.load(open(p))
                if not pid_alive(int(d.get("pid", 0))):
                    os.remove(p)
                    continue
            except Exception:
                pass
            return None
    return None


class Mutex:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                os.mkdir(self.path)
                return self
            except FileExistsError:
                if time.time() - t0 > 120:  # stale
                    try:
                        os.rmdir(self.path)
                    except OSError:
                        pass
                time.sleep(0.2)

    def __exit__(self, *a):
        try:
            os.rmdir(self.path)
        except OSError:
            pass


def reserve_space(name, need, pref=None, wait_s=7200):
    """Pick D: or E: with free-reserved-MARGIN >= need; record reservation. Returns drive letter."""
    resf = os.path.join(LOCKDIR, "reservations.json")
    t0 = time.time()
    order = ["D", "E"] if pref != "E" else ["E", "D"]
    while True:
        with Mutex(os.path.join(LOCKDIR, "space.mutex")):
            try:
                res = json.load(open(resf))
            except Exception:
                res = {}
            res = {k: v for k, v in res.items() if pid_alive(v["pid"])}
            for d in order:
                free = shutil.disk_usage(d + ":\\").free
                resv = sum(v["bytes"] for v in res.values() if v["drive"] == d)
                if free - resv - MARGIN >= need:
                    res[name] = {"drive": d, "bytes": need, "pid": os.getpid(), "ts": now_iso()}
                    json.dump(res, open(resf, "w"))
                    return d
            json.dump(res, open(resf, "w"))
        if time.time() - t0 > wait_s:
            return None
        time.sleep(30)


def release_space(name):
    resf = os.path.join(LOCKDIR, "reservations.json")
    with Mutex(os.path.join(LOCKDIR, "space.mutex")):
        try:
            res = json.load(open(resf))
        except Exception:
            res = {}
        res.pop(name, None)
        json.dump(res, open(resf, "w"))


# ------------------------------------------------------------------ git / enumeration
def git_state(wt, name):
    st = {}
    rc, o, _ = git(wt, "rev-parse", "--abbrev-ref", "HEAD")
    st["branch"] = dec(o).strip() if rc == 0 else None
    rc, o, _ = git(wt, "rev-parse", "HEAD")
    st["head"] = dec(o).strip() if rc == 0 else None
    return st


def parse_status(out):
    ents = []
    parts = out.split(b"\0")
    i = 0
    while i < len(parts):
        e = parts[i]
        i += 1
        if len(e) < 4:
            continue
        xy, path = dec(e[:2]), dec(e[3:])
        paths = [path]
        if xy[0] in "RC" or xy[1] in "RC":
            if i < len(parts):
                paths.append(dec(parts[i]))
                i += 1
        ents.append((xy, paths))
    return ents


def commit_if_dirty(wt, name, rec):
    """Commit tracked changes (non-sensitive paths only). Fills rec fields."""
    rc, o, e = git(wt, "status", "--porcelain=v1", "-z", "-uno")
    if rc != 0:
        rec["status"], rec["error"] = "git_error", "git status failed: " + dec(e)[:200]
        return False
    ents = parse_status(o)
    rec["dirtyBefore"] = len(ents)
    sens, paths = [], []
    for xy, ps in ents:
        if any(SENS_RE.search(p) for p in ps):
            sens.append(ps[-1])
        else:
            paths.extend(ps)
    rec["trackedDirtySensitiveSkipped"] = sens
    if paths:
        spec = b"\0".join(p.encode("utf-8") for p in paths) + b"\0"
        lenv = dict(os.environ, GIT_LITERAL_PATHSPECS="1")
        rc, o, e = git(wt, "add", "-u", "--pathspec-from-file=-", "--pathspec-file-nul", input=spec, env=lenv)
        if rc != 0:
            rec["status"], rec["error"] = "commit_failed", "git add -u failed: " + dec(e)[:300]
            return False
    rc, _, _ = git(wt, "diff", "--cached", "--quiet")
    if rc == 1:
        r2 = git(wt, "-c", "user.name=worktree-archive", "-c", "user.email=worktree-archive@local",
                 "commit", "-m", COMMIT_MSG)
        if r2[0] != 0:
            r2 = git(wt, "-c", "user.name=worktree-archive", "-c", "user.email=worktree-archive@local",
                     "commit", "--no-verify", "-m", COMMIT_MSG)
        if r2[0] != 0:
            rec["status"] = "commit_failed"
            rec["error"] = "commit failed: " + dec(r2[1] + r2[2])[-300:].replace("\n", " ")
            return False
        rec["committed"] = True
        rc, o, _ = git(wt, "rev-parse", "HEAD")
        rec["commitSha"] = dec(o).strip()
        log("  %s committed %s -> branch=%s" % (name, rec["commitSha"], rec.get("branch")))
    return True


def ensure_preserve_ref(wt, name, rec):
    rc, o, _ = git(wt, "rev-parse", "--abbrev-ref", "HEAD")
    br = dec(o).strip()
    rec["branch"] = br
    rc, o, _ = git(wt, "rev-parse", "HEAD")
    head = dec(o).strip()
    rec["headSha"] = head
    if br == "HEAD":
        rec["headType"] = "detached"
        rc, o, _ = run(["git", "-C", MAIN, "for-each-ref", "--contains", head, "--count=1",
                        "--format=%(refname)", "refs/heads", "refs/remotes", "refs/tags"])[0:3]
        if not dec(o).strip():
            ref = "refs/heads/wip-preserve/" + name
            r = run(["git", "-C", MAIN, "update-ref", ref, head])
            if r[0] == 0:
                rec["preserveRef"] = ref
        else:
            rec["preserveRef"] = rec.get("preserveRef") or dec(o).strip().split("\n")[0]
    else:
        rec["headType"] = "branch"


def find_junctions(wt):
    """Walk the worktree without following reparse points; return relative junction/symlink roots."""
    out = []
    stack = [wt]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(lp(d))
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    stt = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                full = os.path.join(d, e.name)
                if getattr(stt, "st_file_attributes", 0) & 0x400 or e.is_symlink():
                    out.append(os.path.relpath(full, wt).replace("\\", "/"))
                elif e.is_dir(follow_symlinks=False):
                    if e.name == ".git":
                        continue
                    stack.append(full)
    return sorted(out)


def enumerate_files(wt, klass, junction_roots):
    """Return (files[(rel,size)], skipped_sens[], skipped_junction[], nested_dirs[], vanished[])."""
    cand = set()
    if klass == "gitdir":
        for dp, dns, fns in os.walk(wt):
            dns[:] = [d for d in dns if d != ".git" and not (getattr(os.lstat(lp(os.path.join(dp, d))), "st_file_attributes", 0) & 0x400)]
            for fn in fns:
                cand.add(os.path.relpath(os.path.join(dp, fn), wt).replace("\\", "/"))
    else:
        for extra in ([], ["--ignored"]):
            rc, o, e = git(wt, "ls-files", "--others", *extra, "--exclude-standard", "-z")
            if rc != 0:
                raise RuntimeError("git ls-files failed: " + dec(e)[:200])
            for p in o.split(b"\0"):
                if p:
                    cand.add(dec(p))
    jset = [j.rstrip("/") for j in junction_roots]
    files, sens, junc, nested, vanished = [], [], [], [], []
    dircache = {}

    def under_reparse(rel):
        parts = rel.split("/")
        cur = wt
        for i in range(len(parts) - 1):
            cur = os.path.join(cur, parts[i])
            v = dircache.get(cur)
            if v is None:
                v = is_reparse(cur)
                dircache[cur] = v
            if v:
                return True
        return False

    for rel in sorted(cand):
        if GIT_RE.search(rel):
            continue
        if rel.endswith("/"):
            nested.append(rel)
            continue
        if SENS_RE.search(rel):
            sens.append(rel)
            continue
        if any(rel == j or rel.startswith(j + "/") for j in jset) or under_reparse(rel):
            junc.append(rel)
            continue
        full = os.path.join(wt, rel.replace("/", "\\"))
        try:
            stt = os.lstat(lp(full))
        except OSError:
            vanished.append(rel)
            continue
        if getattr(stt, "st_file_attributes", 0) & 0x400:
            junc.append(rel)
            continue
        if stat_is_dir(stt):
            nested.append(rel)
            continue
        files.append((rel, stt.st_size))
    return files, sens, junc, nested, vanished


def stat_is_dir(stt):
    import stat as _s
    return _s.S_ISDIR(stt.st_mode)


# ------------------------------------------------------------------ tar | zstd
class HashReader:
    def __init__(self, f, size):
        self.f, self.left, self.h = f, size, hashlib.sha256()

    def read(self, n=-1):
        if n is None or n < 0 or n > self.left:
            n = self.left
        if n == 0:
            return b""
        b = self.f.read(n)
        self.left -= len(b)
        self.h.update(b)
        return b


def build_archive(wt, files, archive):
    members = []
    errors = []
    proc = subprocess.Popen([ZSTD, "-q", "-T0", "-10", "-f", "-o", archive, "-"], stdin=subprocess.PIPE)
    tf = tarfile.open(fileobj=proc.stdin, mode="w|", format=tarfile.PAX_FORMAT, bufsize=1 << 20,
                      encoding="utf-8", copybufsize=1 << 20)
    try:
        for rel, size in files:
            full = os.path.join(wt, rel.replace("/", "\\"))
            try:
                with open(lp(full), "rb") as f:
                    st = os.fstat(f.fileno())
                    sz = st.st_size
                    ti = tarfile.TarInfo(rel)
                    ti.size, ti.mtime, ti.mode, ti.type = sz, int(st.st_mtime), 0o644, tarfile.REGTYPE
                    hr = HashReader(f, sz)
                    tf.addfile(ti, hr)
                    if hr.left != 0:
                        raise IOError("short read, %d bytes missing" % hr.left)
                    members.append((rel, sz, hr.h.hexdigest()))
            except tarfile.TarError:
                raise
            except Exception as ex:  # unreadable/locked/shrunk file -> whole pack fails
                errors.append("%s: %s" % (rel, ex))
                raise
        tf.close()
        proc.stdin.close()
    except BaseException:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait()
        raise
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError("zstd exited %d" % rc)
    return members


def verify_local(archive, members):
    """(a) zstd -dc | tar -t name set == manifest; (b) streamed per-member sha256 == manifest."""
    want = {m[0]: (m[1], m[2]) for m in members}
    # (a) CLI listing
    p1 = subprocess.Popen([ZSTD, "-dcq", archive], stdout=subprocess.PIPE)
    p2 = subprocess.Popen([GTAR, "--quoting-style=literal", "-tf", "-"], stdin=p1.stdout, stdout=subprocess.PIPE)
    p1.stdout.close()
    names = p2.communicate()[0]
    p1.wait()
    listed = [x.rstrip("\r") for x in names.decode("utf-8", errors="replace").split("\n") if x.rstrip("\r")]
    ok_a = (p1.returncode == 0 and p2.returncode == 0 and len(listed) == len(want) and set(listed) == set(want))
    # (b) streamed content hash
    p = subprocess.Popen([ZSTD, "-dcq", archive], stdout=subprocess.PIPE)
    tf = tarfile.open(fileobj=p.stdout, mode="r|", encoding="utf-8")
    seen, bad = 0, []
    for ti in tf:
        if not ti.isfile():
            bad.append("nonfile:" + ti.name)
            continue
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
        if w is None or w[0] != ti.size or w[1] != h.hexdigest() or left != 0:
            bad.append(ti.name)
        seen += 1
    while p.stdout.read(1 << 20):  # drain the tar end-of-archive padding so zstd does not get EPIPE
        pass
    p.stdout.close()
    p.wait()
    ok_b = (p.returncode == 0 and seen == len(want) and not bad)
    return ok_a, ok_b, len(listed), seen, bad[:5]


# ------------------------------------------------------------------ rclone
def rclone(args, logf=None, timeout=36000):
    cmd = [RCLONE] + args
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if logf:
        with open(logf, "ab") as f:
            f.write(("\n$ rclone " + " ".join(args) + "\n").encode())
            f.write(r.stdout + r.stderr)
    return r.returncode, dec(r.stdout), dec(r.stderr)


def _inc(names):
    a = []
    for n in names:
        a += ["--include", n]
    return a


def remote_md5s(remote_dir, names, logf=None):
    """{name: md5} via one 'rclone md5sum DIR --include ..' call (Drive-side md5)."""
    out = {}
    for _ in range(3):
        rc, o, e = rclone(["md5sum", remote_dir, "--stats", "0"] + _inc(names), logf)
        if rc == 0:
            for line in o.splitlines():
                p = line.strip().split(None, 1)
                if len(p) == 2:
                    out[p[1].strip()] = p[0].lower()
            if all(n in out for n in names):
                break
        time.sleep(10)
    return out


def remote_sizes(remote_dir, names, logf=None):
    out = {}
    for _ in range(3):
        rc, o, e = rclone(["lsjson", remote_dir, "--files-only", "--no-modtime", "--no-mimetype"] + _inc(names), logf)
        if rc == 0:
            for it in json.loads(o):
                out[it["Name"]] = it["Size"]
            if all(n in out for n in names):
                break
        time.sleep(10)
    return out



_PROG = re.compile(r"([\d.]+)\s*(B|KiB|MiB|GiB|TiB)\s*/\s*([\d.]+)\s*(B|KiB|MiB|GiB|TiB)")
_UNIT = {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20, "GiB": 1 << 30, "TiB": 1 << 40}


def _done_bytes(logf):
    try:
        with open(logf, "rb") as f:
            f.seek(0, 2)
            n = f.tell()
            f.seek(max(0, n - 3000))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return 0
    m = None
    for m in _PROG.finditer(tail):
        pass
    return int(float(m.group(1)) * _UNIT[m.group(2)]) if m else 0


def upload_dir(src, dest, logf, size, max_attempts=6):
    """rclone copy src -> dest with a stall watchdog: the Clash upstream occasionally pins a connection to a
    very slow node (0.03 MB/s); kill OUR rclone and retry (new connection) when it is too slow."""
    chunk = "256M" if size <= (64 << 20) else "32M"   # >64 MB: smaller chunks give accurate progress for the rate watchdog
    rc = 1
    for attempt in range(1, max_attempts + 1):
        cmd = [RCLONE, "copy", src, dest, "--drive-chunk-size", chunk, "--checksum", "--stats", "10s",
               "--stats-one-line", "-v", "--retries", "3", "--low-level-retries", "10"]
        with open(logf, "ab") as lf:
            lf.write(("\n$ " + " ".join(cmd) + "\n").encode())
            lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            t0 = time.time()
            samples = [(t0, 0)]
            limit = 90 + size / 0.6e6
            while True:
                try:
                    rc = p.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    pass
                now = time.time()
                if chunk != "256M":
                    b = _done_bytes(logf)
                    samples.append((now, b))
                    samples = [x for x in samples if now - x[0] <= 300] or samples[-1:]
                    t_o, b_o = samples[0]
                    # slow node: < 0.6 MB/s over a >=240 s window (rclone accounts bytes as chunks are read)
                    stalled = (now - t0 > 240 and now - t_o >= 240 and (b - b_o) / max(now - t_o, 1) < 0.6e6)
                else:
                    stalled = (now - t0 > limit)
                if stalled:
                    p.kill()
                    p.wait()
                    rc = -9
                    with open(logf, "ab") as lf2:
                        lf2.write(("\n[watchdog] killed slow upload after %.0fs, attempt %d\n" % (now - t0, attempt)).encode())
                    log("  upload watchdog: killed slow rclone after %.0fs (attempt %d, %.0f MB)" % (now - t0, attempt, size / 1e6))
                    break
        if rc == 0:
            return 0
        time.sleep(5)
    return rc

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tmp-pref", default=None)
    ap.add_argument("--reuse", action="store_true", help="reuse an already built+hashed temp archive (same member set)")
    a = ap.parse_args()
    name = a.name
    inv = {t["name"]: t for t in json.load(open(INV, encoding="utf-8-sig"))}
    t = inv.get(name)
    if t is None:
        print("not in inventory:", name)
        return 3
    if t.get("active"):
        print("ACTIVE worktree, skipped:", name)
        return 3
    wt = t["path"]
    if not os.path.isdir(wt):
        print("path missing:", wt)
        return 3
    recpath = os.path.join(RECDIR, name + ".json")
    if os.path.exists(recpath):
        try:
            if json.load(open(recpath, encoding="utf-8-sig")).get("status") == "ok" and not a.dry_run:
                print("already ok:", name)
                return 0
        except Exception:
            pass
    lock = take_lock(name)
    if lock is None and not a.dry_run:
        print("locked by another worker:", name)
        return 4
    klass = t["class"]
    rec = {
        "name": name, "path": wt, "class": klass, "manual": False, "branch": t.get("branch"),
        "headBefore": t.get("head"), "headType": None, "dirtyBefore": t.get("trackedDirtyCount"),
        "committed": False, "commitSha": None, "preserveRef": None,
        "dest": "%s/%s" % (REMOTE_ROOT, name), "listFile": None, "uploadCount": 0, "uploadBytes": 0,
        "upload": {"ran": False, "exit": None, "log": None},
        "check": {"ran": False, "exit": None, "result": None, "log": None},
        "junctions": [], "skippedSensitiveCount": 0, "skippedSensitiveSample": [], "skippedJunctionCount": 0,
        "status": "ok", "error": None, "timestamp": None,
        "method": "tar.zst",
    }
    tmpdir = None
    t_start = time.time()
    try:
        log("  %s tar.zst start (inventory files=%s bytes=%s)" % (name, t.get("uploadCount"), t.get("uploadBytes")))
        # (1) commit
        if klass == "worktree" and not a.dry_run:
            if not commit_if_dirty(wt, name, rec) and rec["status"] != "ok":
                raise RuntimeError(rec["error"])
            ensure_preserve_ref(wt, name, rec)
        # (2) enumerate
        junc_roots = find_junctions(wt)
        rec["junctions"] = junc_roots
        rec["manual"] = len(junc_roots) > 0
        files, sens, junc, nested, vanished = enumerate_files(wt, klass, junc_roots)
        total = sum(s for _, s in files)
        rec["skippedSensitiveCount"] = len(sens)
        rec["skippedSensitiveSample"] = sens[:25]
        rec["skippedSensitive"] = sens
        rec["skippedJunctionCount"] = len(junc)
        rec["skippedJunctionPaths"] = junc[:200]
        rec["nestedRepoDirs"] = nested
        rec["vanishedDuringEnum"] = vanished
        log("  %s enum files=%d bytes=%d skipped_sens=%d skipped_junction_files=%d junction_roots=%d nested=%d"
            % (name, len(files), total, len(sens), len(junc), len(junc_roots), len(nested)))
        if a.dry_run:
            print(json.dumps({"files": len(files), "bytes": total, "sens": sens[:10], "junctions": junc_roots[:10],
                              "dirty": rec.get("dirtyBefore"), "nested": nested[:5]}, indent=1))
            return 0
        listf = os.path.join(LISTDIR, name + ".tar.txt")
        with open(listf, "w", encoding="utf-8", newline="\n") as f:
            f.write("".join(r + "\n" for r, _ in files))
        rec["listFile"] = listf
        rec["uploadCount"], rec["uploadBytes"] = len(files), total
        if not files:
            rec["method"] = "none"
            rec["status"] = "ok"
            rec["error"] = None
            log("  %s no untracked content (committed=%s)" % (name, rec["committed"]))
            return finish(rec, recpath, 0)
        # (3) temp space
        need = int(total * 1.2) + (64 << 20)
        members = None
        drv = None
        manf0 = os.path.join(MANDIR, name + ".manifest.json")
        if a.reuse and os.path.exists(manf0):
            for d_, root_ in TMP_ROOTS.items():
                cand = os.path.join(root_, name, name + ".tar.zst")
                if os.path.exists(cand):
                    man0 = json.load(open(manf0, encoding="utf-8"))
                    mem0 = [(m["path"], m["size"], m["sha256"]) for m in man0["members"]]
                    if sorted(m[0] for m in mem0) == sorted(r for r, _ in files) and                             os.path.getsize(cand) == man0["archive"]["size"]:
                        drv, tmpdir, members = d_, os.path.join(root_, name), mem0
                        log("  %s reusing existing archive %s" % (name, cand))
                    break
        if members is None:
            drv = reserve_space(name, need, a.tmp_pref)
            if drv is None:
                raise RuntimeError("no temp space (need %d)" % need)
            tmpdir = os.path.join(TMP_ROOTS[drv], name)
        os.makedirs(tmpdir, exist_ok=True)
        archive = os.path.join(tmpdir, name + ".tar.zst")
        t0 = time.time()
        for attempt in (1, 2):
            if members is not None:
                break
            try:
                members = build_archive(wt, files, archive)
                break
            except Exception as ex:
                log("  %s pack attempt %d failed: %s" % (name, attempt, ex))
                if attempt == 2:
                    raise RuntimeError("pack failed: %s" % ex)
                time.sleep(20)
                files, sens, junc, nested, vanished = enumerate_files(wt, klass, junc_roots)
        t_pack = time.time() - t0
        asize = os.path.getsize(archive)
        mbytes = sum(m[1] for m in members)
        asha, amd5 = sha256_md5_file(archive)
        log("  %s packed members=%d bytes=%d -> zst=%d (%.1fx) %.0fs to %s" %
            (name, len(members), mbytes, asize, (mbytes / asize) if asize else 0, t_pack, drv))
        # (4) local verify
        t0 = time.time()
        ok_a, ok_b, nlist, nseen, bad = verify_local(archive, members)
        t_ver = time.time() - t0
        if not (ok_a and ok_b):
            raise RuntimeError("local verify failed ok_tar_t=%s ok_sha=%s listed=%d seen=%d bad=%s" %
                               (ok_a, ok_b, nlist, nseen, bad))
        # manifest
        man = {
            "schema": "wtclean_tar_manifest/1", "name": name, "source": wt, "created": now_iso(),
            "method": "tar.zst (tar PAX | zstd -T0 -10)", "branch": rec.get("branch"), "head": rec.get("headSha"),
            "commitSha": rec.get("commitSha"), "preserveRef": rec.get("preserveRef"),
            "archive": {"name": name + ".tar.zst", "size": asize, "md5": amd5, "sha256": asha},
            "member_count": len(members), "member_bytes": mbytes,
            "skipped_sensitive": sens, "skipped_junction_files": junc, "junction_roots": junc_roots,
            "nested_repo_dirs": nested,
            "members": [{"path": p, "size": s, "sha256": h} for p, s, h in members],
        }
        manf = os.path.join(MANDIR, name + ".manifest.json")
        with open(manf, "w", encoding="utf-8", newline="\n") as f:
            json.dump(man, f, ensure_ascii=False)
        msize = os.path.getsize(manf)
        _, mmd5 = sha256_md5_file(manf)
        # (5) upload
        dest = "%s/%s" % (REMOTE_ROOT, name)
        upl = os.path.join(LOGDIR, name + ".tar.upload.log")
        rec["upload"].update(ran=True, log=upl)
        t0 = time.time()
        rc = 1
        shutil.copy2(manf, os.path.join(tmpdir, name + ".manifest.json"))  # one rclone call uploads both files
        rc = upload_dir(tmpdir, dest, upl, asize)
        t_up = time.time() - t0
        rec["upload"]["exit"] = rc
        if rc != 0:
            rec["status"], rec["error"] = "upload_failed", "rclone copyto exit %d" % rc
            raise RuntimeError(rec["error"])
        # (6) remote verify
        chk = os.path.join(LOGDIR, name + ".tar.check.log")
        rec["check"].update(ran=True, log=chk)
        md5s = remote_md5s(dest, [name + ".tar.zst", name + ".manifest.json"], chk)
        sizes = remote_sizes(dest, [name + ".tar.zst", name + ".manifest.json"], chk)
        rmd5, rmm5 = md5s.get(name + ".tar.zst"), md5s.get(name + ".manifest.json")
        rsz, rmsz = sizes.get(name + ".tar.zst"), sizes.get(name + ".manifest.json")
        okr = (rmd5 == amd5 and rsz == asize and rmm5 == mmd5 and rmsz == msize)
        res = ("tar.zst verified: remote md5 %s local %s; size %s/%s; manifest md5 %s size %s; members=%d (tar -t %d, sha256 %d); 0 differences found"
               % (rmd5, amd5, rsz, asize, "match" if rmm5 == mmd5 else "MISMATCH", "match" if rmsz == msize else "MISMATCH",
                  len(members), nlist, nseen)) if okr else \
              ("tar.zst VERIFY FAILED: remote md5 %s local %s; size %s/%s; manifest md5 %s/%s size %s/%s" %
               (rmd5, amd5, rsz, asize, rmm5, mmd5, rmsz, msize))
        rec["check"]["exit"] = 0 if okr else 1
        rec["check"]["result"] = res
        open(chk, "w", encoding="utf-8").write(res + "\n")
        if not okr:
            rec["status"], rec["error"] = "check_failed", res
            raise RuntimeError(res)
        rec["archive"] = {
            "remote": "%s/%s.tar.zst" % (dest, name), "name": name + ".tar.zst", "size": asize, "md5": amd5, "sha256": asha,
            "manifest_remote": "%s/%s.manifest.json" % (dest, name), "manifest_local": manf,
            "manifest_size": msize, "manifest_md5": mmd5, "member_count": len(members), "member_bytes": mbytes,
            "zstd_args": "-T0 -10", "pack_seconds": round(t_pack, 1), "verify_seconds": round(t_ver, 1),
            "upload_seconds": round(t_up, 1), "upload_MBps": round(asize / max(t_up, 0.001) / 1e6, 2),
            "verified": {"remote_md5": rmd5 == amd5, "remote_size": rsz == asize, "tar_t_equals_manifest": ok_a,
                         "stream_sha256_equals_manifest": ok_b, "manifest_remote_md5": rmm5 == mmd5},
        }
        rec["memberCount"], rec["memberBytes"] = len(members), mbytes
        rec["junctionsFound"] = junc_roots
        log("  %s ok tar.zst upload+check files=%d bytes=%d zst=%d up=%.0fs (%.2f MB/s) %s" %
            (name, len(members), mbytes, asize, t_up, asize / max(t_up, 0.001) / 1e6, res[:60]))
        # delete only our own temp archive (verified)
        try:
            os.remove(archive)
            os.rmdir(tmpdir)
        except OSError:
            pass
        tmpdir = None
        return finish(rec, recpath, 0)
    except Exception as ex:
        if rec["status"] == "ok":
            rec["status"] = "error"
        rec["error"] = rec["error"] or str(ex)
        log("  %s FAILED %s: %s" % (name, rec["status"], str(ex)[:300]))
        return finish(rec, recpath, 1)
    finally:
        if tmpdir and os.path.isdir(tmpdir):
            try:
                shutil.rmtree(tmpdir)  # our own temp dir only
            except OSError:
                pass
        release_space(name)
        if lock:
            try:
                os.remove(lock)
            except OSError:
                pass
        log("  %s done in %.0fs" % (name, time.time() - t_start))


def finish(rec, recpath, code):
    rec["timestamp"] = datetime.datetime.now().astimezone().isoformat()
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    if rec["status"] != "ok" and os.path.exists(recpath):
        # failed attempt: keep the earlier receipt untouched, park the failure elsewhere
        os.makedirs(BASE + "/receipts_failed", exist_ok=True)
        recpath = BASE + "/receipts_failed/%s.%s.json" % (rec["name"], stamp)
    elif os.path.exists(recpath):
        os.makedirs(BASE + "/receipts_prev", exist_ok=True)
        shutil.copy2(recpath, BASE + "/receipts_prev/%s.%s.json" % (rec["name"], stamp))
    tmp = recpath + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    os.replace(tmp, recpath)
    log("  %s RECEIPT status=%s%s" % (rec["name"], rec["status"], "" if code == 0 else " (parked: %s)" % os.path.basename(recpath)))
    return code


if __name__ == "__main__":
    sys.exit(main())

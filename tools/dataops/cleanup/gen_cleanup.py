#!/usr/bin/env python3
"""gen_cleanup.py - regenerate MORNING_CLEANUP.ps1 (the owner's clean-up script) from the Drive-backup receipts.

Lives in the repository (tools/dataops/cleanup/).  Re-runnable at any time:

    python -I tools\dataops\cleanup\gen_cleanup.py                       # regenerate script + manifest
    python -I tools\dataops\cleanup\gen_cleanup.py --reorg3-receipts     # compare: read E:\rd-wt\reorg3 instead of data_ops\receipts
    python -I tools\dataops\cleanup\gen_cleanup.py --summary             # also refresh CLEANUP_SUMMARY.md tables

READ-ONLY with respect to everything except the output folder (data_ops/cleanup_out, git-ignored).  It never
deletes, moves or edits research files; it never opens rclone.conf; it does not run MORNING_CLEANUP.ps1.  It
never reads the contents of sealed / CAL / HOLDOUT / chunk-5 / INC1 / j4b paths and never opens a receipt whose
FILE NAME is sensitive-named (those worktree entries are carried over from the previous manifest instead).

Inputs (all read-only):
  <receipts-root>/driveops/receipts/*.json (+ receipts/md5/*.md5)      mirrored-directory backups
  <receipts-root>/dataops/receipts/*.json                              tar.zst archive backups
  <receipts-root>/wtclean/receipts/*.json                              worktree backups (the ONLY source of worktree candidates)
  <heavy-root>/{dataops,wtclean}/{manifests,lists,_out,...}            member manifests / upload lists (E:\rd-wt\reorg3)
Defaults: receipts-root = <repo>/data_ops/receipts, heavy-root = E:\rd-wt\reorg3 (retired LAST, see README).
Outputs (<repo>/data_ops/cleanup_out): MORNING_CLEANUP.ps1, cleanup_manifest.json, data_registry_cache.toml,
  gen_state.json (+ CLEANUP_SUMMARY.md tables with --summary).
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True

BS = chr(92)
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))


def _repo_root():
    r = os.path.abspath(os.path.join(TOOL_DIR, "..", "..", ".."))
    return r.replace("/", BS)


MAIN = _repo_root()                       # E:\research-dense
OUT = MAIN + r"\data_ops\cleanup_out"       # generated, git-ignored
RECEIPTS = MAIN + r"\data_ops\receipts"     # tracked copy of the receipts
REORG = r"E:\rd-wt\reorg3"                  # heavy generated parts (manifests, lists, inventory): retired last
# receipt roots (tracked copy) and heavy roots (manifests / lists / _out); reassigned in main() from the flags
DRIVEOPS_R = RECEIPTS + r"\driveops"
DATAOPS_R = RECEIPTS + r"\dataops"
WTCLEAN_R = RECEIPTS + r"\wtclean"
DRIVEOPS = REORG + r"\driveops"
DATAOPS = REORG + r"\dataops"
WTCLEAN = REORG + r"\wtclean"
HERE = OUT                                  # kept for the summary helpers below
CLEAN = MAIN                                # single work tree since 2026-10-10 (formerly E:\rd-wt\ws-clean-int)
CACHE = MAIN + r"\data_cache"
SEALED_ROOT = MAIN + r"\sealed\relocated_20261010"
RCLONE = r"C:\Users\jpeng\miniconda3\bin\rclone.exe"
RDWT = r"E:\rd-wt"

GEN_VERSION = "2"

# Owner decision 2026-10-10: ALL worktrees retire.  Names listed here are never removed (mechanism kept; add
# names or pass --protect NAME).
PROTECTED_WORKTREES = []

# Worktree entries whose receipt FILE NAME is sensitive-named (never opened by this generator) are carried over
# from the previous manifest (they were generated before the sensitive-name rule was applied to receipts).
CARRY_WORKTREES = ["j4bfix"]

# Same rule as dataops upload_driver.sensitive(): these subtrees were never uploaded.
SENS_EXACT = {"sealed", "cal", "holdout", "inc1"}
SENS_CONTAINS = ("chunk5", "chunk-5")
# Broad sensitive-name rule = the one the read-only worktree packer (pack_wt_ro.py) used for the receipts: substring
# holdout/heldout/cal_split/chunk5/chunk-5/j4b/inc1/sealed or the whole token "cal"; applied to the relative path of an
# untracked file (case-insensitive).  MOVE_TO_SEALED's plan has the same rule except "heldout" (see the README).
BROAD_SENS_RE = r"(?i)(holdout|heldout|cal_split|chunk-?5|j4b|inc1|sealed|(^|[\\/_.\-\s])cal([\\/_.\-\s]|$))"   # == pack_wt_ro.py SENS_RE
BROAD_SENS = re.compile(BROAD_SENS_RE)

ALLOWED_ROOTS = [
    r"D:\rd-staging",
    r"E:\rd-wt",
    MAIN + r"\runs",
    MAIN + r"\output",
    r"E:\research-dense-backups",
    r"E:\rd-release",
    r"E:\rented_bundle",
    r"E:\rented_bundle_dspeed",
    r"E:\glm",
    r"E:\glm-worker",
    r"E:\migration_logs",
    r"E:\research-github",
    r"E:\research-dense-c-migrated",
    r"C:\Users\jpeng\AppData\Local\Temp",
]
# directories that must never be removed even when they become empty
CONTAINER_ROOTS = [
    "C:\\", "D:\\", "E:\\", "G:\\", MAIN, MAIN + r"\runs", MAIN + r"\output", MAIN + r"\tmp",
    MAIN + r"\runs\active", r"D:\rd-staging", r"D:\rd-staging\wmn1_feat", r"E:\rd-wt",
    r"C:\Users\jpeng\AppData\Local\Temp",
]
# literal paths in code that name a generic container are ignored when they are ANCESTORS of a candidate
GENERIC_LITERALS = [
    MAIN, MAIN + r"\runs", MAIN + r"\output", MAIN + r"\tmp", MAIN + r"\runs\active",
    r"D:\rd-staging", r"D:\rd-staging\wmn1_feat", r"E:\rd-wt", "E:\\", "D:\\", "C:\\",
]
# reference-scan: files whose literals are NOT treated as "active code" (archival copies of the data-ops tooling,
# which names E:\rd-wt\reorg3\... and E:\rd-wt\<worktree> paths in comments and default arguments)
IGNORE_REF_FILES = r"[\\/]tools[\\/](dataops|build_drive_registry|offload_map|prepush_check|prepush_allow)|[\\/]tests[\\/]unit[\\/]data[\\/]test_drive_registry|[\\/]tests[\\/]golden[\\/]data[\\/]"
USERDOC_PATHS = [  # personal admin documents / media: opt-in only (-IncludeUserDocs)
    r"C:\Users\jpeng\Downloads\Send Anywhere (2026-09-21 03-19-30)\ResearchProposalForm_Jinghan_filled.pdf",
    r"C:\Users\jpeng\Documents\Codex\2026-09-07\cuhk-2\work\research-registration\ResearchProposalForm.pdf",
    r"C:\Users\jpeng\OneDrive\Desktop\WM_short_20261005.pptx",
    r"C:\Users\jpeng\Downloads\sumo-video.mp4",
]


# ---------------------------------------------------------------- helpers
def nrm(p):
    """Windows path with backslashes and without trailing separator (keeps case)."""
    p = p.replace("/", "\\")
    if len(p) > 3:
        p = p.rstrip("\\")
    return p


def key(p):
    return nrm(p).lower()


def under(p, root):
    p, r = key(p), key(root)
    return p == r or p.startswith(r.rstrip("\\") + "\\")


def is_sensitive_name(name):
    n = name.lower()
    return n in SENS_EXACT or any(t in n for t in SENS_CONTAINS)


def path_is_sensitive(path):
    return any(is_sensitive_name(c) for c in nrm(path).split("\\") if c)


def broad_sensitive(text):
    """MOVE_TO_SEALED's broad rule (also matches j4b): used to keep sensitive-named FILE NAMES unopened."""
    return bool(BROAD_SENS.search(text))


def resolve(root_r, root_h, rel):
    """rel (relative, '/' or backslash) under the receipts root if it exists there, else under the heavy root."""
    rel = rel.replace("/", BS)
    for base in (root_r, root_h):
        p = base + BS + rel
        if os.path.exists(p):
            return p
    return root_r + BS + rel


def rj(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def sha256_of(path, limit_bytes=64 * 1024 * 1024):
    if not os.path.exists(path) or os.path.getsize(path) > limit_bytes:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lp(p):
    p = nrm(p)
    return p if p.startswith("\\\\?\\") else "\\\\?\\" + p


def walk_live(root):
    """(files, n_reparse, n_sensitive) under root; mirrors dataops walk rules; no content opened."""
    files = []
    n_rep = n_sens = 0
    if os.path.isfile(lp(root)):
        try:
            return [(nrm(root), os.path.getsize(lp(root)))], 0, 0
        except OSError:
            return [], 0, 0
    stack = [nrm(root)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(lp(d))
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if getattr(st, "st_file_attributes", 0) & 0x400:  # reparse point
                    n_rep += 1
                    continue
                if is_sensitive_name(e.name):
                    n_sens += 1
                    continue
                if e.is_dir(follow_symlinks=False):
                    stack.append(os.path.join(d, e.name))
                elif e.is_file(follow_symlinks=False):
                    files.append((os.path.join(d, e.name), st.st_size))
    return files, n_rep, n_sens


def run(cmd, cwd=None, timeout=300):
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return r.returncode, r.stdout, r.stderr
    except Exception as exc:  # pragma: no cover
        return 99, "", str(exc)


def parse_ts(s):
    try:
        return datetime.datetime.fromisoformat(s)
    except Exception:
        return None


def gib(n):
    return n / 1024 ** 3


# ---------------------------------------------------------------- collectors
def collect_driveops(items, notes):
    d = DRIVEOPS_R + r"\receipts"
    n = 0
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json"):
            continue
        if broad_sensitive(f):
            notes.append("driveops receipt %s has a sensitive-named file name: not opened" % f)
            continue
        r = rj(os.path.join(d, f))
        if r.get("schema") != "driveops_receipt/1":
            continue
        if r.get("check") != "passed":
            notes.append("driveops receipt %s not passed (%s) - skipped" % (r.get("name"), r.get("check")))
            continue
        locs = r["local"] if isinstance(r["local"], list) else [r["local"]]
        man = resolve(DRIVEOPS_R, DRIVEOPS, r["md5_manifest"])
        kind = r.get("kind")
        for loc in locs:
            loc = nrm(loc)
            if kind == "dirs":
                prefix = os.path.basename(loc) + "/"
            else:
                prefix = ""
            it = item_for(items, loc)
            it.setdefault("origin", []).append("driveops:" + r["name"])
            it["class"] = r.get("class") or it.get("class", "")
            it["backup_only"] = it.get("backup_only", False) or (r.get("class") == "KEEP-ACTIVE-backup-only")
            it["kind_hint"] = "file" if kind == "files" else "dir"
            it["proofs"].append({
                "t": "mirror", "receipt": r["name"], "remote": r["drive"].rstrip("/"),
                "manifest": man, "manifest_sha256": sha256_of(man), "prefix": prefix,
                "ts": r.get("timestamp"), "bytes_total": r.get("bytes"), "files_total": r.get("files"),
            })
            n += 1
    return n


def item_for(items, path):
    k = key(path)
    if k not in items:
        items[k] = {"id": k, "path": nrm(path), "proofs": [], "origin": [], "class": "", "backup_only": False}
    return items[k]


PART_RE = re.compile(r"/part\d{3}$")


def collect_dataops(items, notes):
    d = DATAOPS_R + r"\receipts"
    groups = {}
    n = 0
    for f in sorted(os.listdir(d)):
        if not f.endswith(".json") or f == "COVERAGE.json":
            continue
        if broad_sensitive(f):
            notes.append("dataops receipt %s has a sensitive-named file name: not opened" % f)
            continue
        try:
            r = rj(os.path.join(d, f))
        except Exception as exc:
            notes.append("dataops receipt %s unreadable: %s" % (f, exc))
            continue
        if r.get("schema") != "dataops_receipt/1" or r.get("check") != "passed" or not r.get("id"):
            continue
        loc = nrm(r["local"])
        man = resolve(DATAOPS_R, DATAOPS, r["member_manifest"])
        groups.setdefault(key(loc), {"path": loc, "kind": r.get("kind"), "class": r.get("class"), "parts": []})
        groups[key(loc)]["parts"].append({
            "id": r["id"], "remote": r["drive"], "md5": (r.get("archive_md5") or "").lower(),
            "bytes": r.get("archive_bytes"), "manifest": man, "manifest_sha256": sha256_of(man),
            "ts": r.get("timestamp"), "src_bytes": r.get("bytes"), "src_files": r.get("files"),
        })
        n += 1
    for k, g in groups.items():
        it = item_for(items, g["path"])
        it["origin"].append("dataops")
        it["class"] = it["class"] or g["class"] or ""
        it["kind_hint"] = "file" if g["kind"] == "file" else "dir"
        it["bytes"] = sum(p["src_bytes"] or 0 for p in g["parts"])
        it["files"] = sum(p["src_files"] or 0 for p in g["parts"])
        it["proofs"].append({"t": "archive", "parts": g["parts"]})
    return n


LEGACY = [  # pre-reorg archives; current Drive paths found by listing gdrive:research-archive/ (2026-10-10)
    # (receipt archive name, local path, remote, strip prefix of the receipt's member names)
    ("upload_v2.tar.zst", r"D:\rd-staging\wmn1_feat\upload_v2",
     "gdrive:research-archive/data/packs/archived_training_packs_v2/upload_v2.tar.zst", "upload_v2/"),
    ("upload_v2c.tar.zst", r"D:\rd-staging\wmn1_feat\upload_v2c",
     "gdrive:research-archive/data/packs/archived_training_packs_v2/upload_v2c.tar.zst", "upload_v2c/"),
    ("raw_chunk1_p40.tar.zst", r"D:\rd-staging\wmn1_feat\raw_chunk1_p40",
     "gdrive:research-archive/repo-evidence/runs/raw_and_superseded/raw_chunk1_p40.tar.zst", "raw_chunk1_p40/"),
    ("shards_v2__chunk1.tar.zst", r"D:\rd-staging\wmn1_feat\shards_v2\chunk1",
     "gdrive:research-archive/data/packs/derived_shards_v2/Chunks/shards_v2__chunk1.tar.zst", "shards_v2__chunk1/"),
    ("shards_v2__chunk1_fix27.tar.zst", r"D:\rd-staging\wmn1_feat\shards_v2\chunk1_fix27",
     "gdrive:research-archive/data/packs/derived_shards_v2/Chunks/shards_v2__chunk1_fix27.tar.zst", "shards_v2__chunk1_fix27/"),
    ("shards_v2__chunk2.tar.zst", r"D:\rd-staging\wmn1_feat\shards_v2\chunk2",
     "gdrive:research-archive/data/packs/derived_shards_v2/Chunks/shards_v2__chunk2.tar.zst", "shards_v2__chunk2/"),
]
LEGACY_INDEX = ("shards_v2__index.tar.zst", r"D:\rd-staging\wmn1_feat\shards_v2",
                "gdrive:research-archive/data/packs/derived_shards_v2/Index/shards_v2__index.tar.zst", "shards_v2__index/")


def collect_legacy(items, notes):
    n = 0
    specs = []
    for arch, loc, remote, strip in LEGACY:
        specs.append((arch, loc, remote, strip, "dir"))
    arch, base, remote, strip = LEGACY_INDEX
    rc0 = DATAOPS + r"\_out\drive_receipts\%s.upload.json" % arch
    if os.path.exists(rc0):
        for k in rj(rc0).get("files_sha256", {}):
            specs.append((arch, base + "\\" + k[len(strip):], remote, strip, "file"))
    for arch, loc, remote, strip, kind in specs:
        rc = DATAOPS + r"\_out\drive_receipts\%s.upload.json" % arch
        if not os.path.exists(rc):
            notes.append("legacy receipt missing: " + rc)
            continue
        r = rj(rc)
        it = item_for(items, loc)
        it["origin"].append("legacy:" + arch)
        it["class"] = it["class"] or "DRIVE-ONLY"
        it["kind_hint"] = kind
        it["bytes"] = r.get("original_bytes") if kind == "dir" else None
        it["files"] = r.get("n_files") if kind == "dir" else 1
        it["proofs"].append({
            "t": "legacy", "remote": remote, "md5": r.get("archive_md5"), "bytes": r.get("archive_bytes"),
            "receipt": rc, "receipt_sha256": sha256_of(rc), "strip": strip,
        })
        n += 1
    return n


def special_items(items, notes):
    # redundant subset of pack_v3 (owner: redundant)
    it = item_for(items, r"D:\rd-staging\pack_v3_r3_subset")
    it["origin"].append("owner:redundant")
    it["class"] = "REDUNDANT"
    it["kind_hint"] = "dir"
    it["special"] = {"kind": "redundant_of_keeplocal", "keep": "pack_v3",
                     "why": "row-subset of pack_v3 built for the R3 box run; pack_v3 is kept locally and on Drive"}
    # partial dropped scp copy
    p = MAIN + r"\runs\active\WM-S-R3-20261010\box\gpuA\switching_pack_gpuA.tar"
    it = item_for(items, p)
    it["origin"].append("owner:redundant")
    it["class"] = "REDUNDANT"
    it["kind_hint"] = "file"
    sz = os.path.getsize(p) if os.path.exists(p) else None
    it["special"] = {"kind": "fixed_size", "size": sz,
                     "why": "partial 176 MB dropped copy of the switching pack (scp now fetches file by file into wmn1_switching_train)"}
    # (the former ws-clean-int runs/tmp_int scratch item is gone: ws-clean-int retires as a whole worktree)
    # personal documents with a Drive copy: opt-in only
    for up in USERDOC_PATHS:
        if os.path.exists(up):
            u = item_for(items, up)
            u["origin"].append("owner:userdoc")
            u["class"] = "USERDOC"
            u["kind_hint"] = "file"
            u["userdoc"] = True
    return


def heavy_path(p):
    """Map a path recorded under the old heavy root to the configured heavy root (identity by default)."""
    if not p:
        return p
    pn = nrm(p)
    if key(pn).startswith(key(r"E:\rd-wt\reorg3") + BS) and key(REORG) != key(r"E:\rd-wt\reorg3"):
        return REORG + pn[len(r"E:\rd-wt\reorg3"):]
    return pn


def collect_worktrees(notes, carry_from):
    """Worktree candidates = every rd-wt/<name> directory that has an ok wtclean receipt (inventory.json is not used).

    Receipts whose FILE NAME is sensitive-named are not opened; names in CARRY_WORKTREES are carried over from the
    previous manifest(s) in `carry_from` (read before they are overwritten).
    """
    rdir = WTCLEAN_R + r"\receipts"
    out, gone, skipped = [], [], []
    seen = set()
    for f in sorted(os.listdir(rdir)):
        if not f.endswith(".json"):
            continue
        if broad_sensitive(f):
            skipped.append(f)
            continue
        try:
            r = rj(os.path.join(rdir, f))
        except Exception as exc:
            notes.append("wtclean receipt %s unreadable: %s" % (f, exc))
            continue
        name = r.get("name")
        path = nrm(r.get("path") or "")
        if not name or not under(path, RDWT) or key(path) == key(RDWT):
            continue
        if not os.path.isdir(lp(path)):
            gone.append(name)
            continue
        if r.get("status") not in ("ok", "commit_failed"):
            notes.append("wtclean receipt %s status %r: held by the script (RECEIPT_NOT_PASSED)" % (name, r.get("status")))
        tar = (r.get("method") == "tar.zst")
        sens_full = r.get("skippedSensitive")
        sens_names = sens_full if isinstance(sens_full, list) else (r.get("skippedSensitiveSample") or [])
        sens_count = int(r.get("skippedSensitiveCount") or len(sens_names))
        jn = r.get("junctions") or []
        lf = heavy_path(r.get("listFile") or "")
        ar = r.get("archive")
        if ar:
            ar = dict(ar)
            if ar.get("manifest_local"):
                ar["manifest_local"] = heavy_path(ar["manifest_local"])
        seen.add(name.lower())
        out.append({
            "name": name, "path": path, "cls": r.get("class") or "worktree", "branch": r.get("branch"),
            "head": r.get("headSha") or r.get("headBefore"), "status": r.get("status"),
            "upload_exit": (r.get("upload") or {}).get("exit"), "check_exit": (r.get("check") or {}).get("exit"),
            "dest": r.get("dest"), "list": lf, "list_sha256": sha256_of(lf) if lf else None,
            "files": r.get("uploadCount"), "bytes": r.get("uploadBytes"),
            "committed": r.get("committed"), "commit": r.get("commitSha"), "preserve_ref": r.get("preserveRef"),
            "junctions": len(jn), "junction_names": jn[:300],
            "method": r.get("method"), "archive": ar,
            "sens_count": sens_count, "sens": sens_names[:60], "sens_complete": bool(len(sens_names) >= sens_count),
            "dirty_paths": r.get("trackedDirtyPaths") or [],
            "dirty_sens_skipped": r.get("trackedDirtySensitiveSkipped") or [],
            "ts": r.get("timestamp"),
        })
    # carried entries: receipts that are sensitive-named (never opened here)
    carried, missing = [], []
    for name in CARRY_WORKTREES:
        if name.lower() in seen:
            continue
        path = RDWT + BS + name
        if not os.path.isdir(lp(path)):
            continue
        ent = None
        for mp in carry_from:
            if not os.path.exists(mp):
                continue
            try:
                with open(mp, encoding="utf-8") as fh:
                    m0 = json.load(fh)
            except Exception:
                continue
            for w in m0.get("worktrees", []):
                if w.get("name") == name:
                    ent = dict(w)
                    ent["carried_from"] = mp
                    break
            if ent:
                break
        if ent:
            ent["sealed_relocate"] = True
            carried.append(ent)
            seen.add(name.lower())
            out.append(ent)
        else:
            missing.append(name)
    # mark worktrees that must move sensitive-named untracked files into sealed/ themselves (root is sensitive-named)
    for w in out:
        if broad_sensitive(w["name"]):
            w["sealed_relocate"] = True
        w["last"] = (key(w["path"]) == key(REORG))
    # directories under E:\rd-wt without an ok receipt (and not carried): reported, never removed
    unreceipted = []
    for d in sorted(os.listdir(RDWT)):
        if d.lower() in seen:
            continue
        p = RDWT + BS + d
        if os.path.isdir(lp(p)):
            unreceipted.append({"name": d, "path": p})
    if skipped:
        notes.append("wtclean receipts not opened (sensitive-named file names): %s" % ", ".join(skipped))
    if missing:
        notes.append("carried worktree(s) with no previous manifest entry: %s (held as NO_RECEIPT)" % ", ".join(missing))
    return out, unreceipted, gone, carried, skipped


# ---------------------------------------------------------------- keep-local
def keep_local_items():
    mf = DATAOPS + r"\manifests"
    def m(n):
        p = mf + "\\" + n
        return {"manifest": p, "manifest_sha256": sha256_of(p)}
    K = []

    def add(name, src, cache, manifest=None, drive=None, ref_sha=None, old="remove", note="", complete=True,
            expect=None, registry=None):
        d = {"name": name, "source": nrm(src), "cache": cache, "old_policy": old, "note": note,
             "complete": complete, "drive": drive or [], "ref_sha256sums": ref_sha, "registry": registry or name}
        if manifest:
            d.update(m(manifest))
        if expect:
            d["expect"] = expect
        K.append(d)

    # packs: Drive proofs are tried in order; the first that fully verifies counts
    add("pack_v3", r"D:\rd-staging\wmn1_feat\pack_v3", "pack_v3", "pack_v3.md5",
        drive=[{"t": "tar_parts", "remote": "gdrive:research-archive/data/packs/training_pack_v3_k192",
                "parts_sums": "SHA256SUMS", "content_manifest": "upload_manifest_v3.json"},
               {"t": "dir_md5", "remote": "gdrive:research-archive/data/packs/pack_v3"}],
        ref_sha="SHA256SUMS", note="49 GB; Drive copy = tar.zst parts training_pack_v3_k192 (+ optional gpuA upload)")
    add("pack_dspeed", r"E:\rd-wt\dspeed-data\_out\pack_dspeed", "pack_dspeed", "pack_dspeed.md5",
        drive=[{"t": "dir_md5", "remote": "gdrive:research-archive/data/packs/pack_dspeed"},
               {"t": "dir_md5", "remote": "gdrive:research-archive/worktrees/dspeed-data/_out/pack_dspeed"}],
        ref_sha="SHA256SUMS", note="12 GB; Drive copy appears when wtclean finishes dspeed-data or gpuA uploads packs")
    add("pack_damping", r"E:\rd-wt\damping-snap\_out\pack_damping", "pack_damping", "pack_damping.md5",
        drive=[{"t": "dir_md5", "remote": "gdrive:research-archive/data/packs/pack_damping"},
               {"t": "dir_md5", "remote": "gdrive:research-archive/worktrees/damping-snap/_out/pack_damping"}],
        ref_sha="SHA256SUMS", note="4.9 GB")
    add("sidecars_packv3_v2", r"E:\rd-wt\ws-packv3fix\_out\sidecars_packv3_v2", "sidecars_packv3_v2",
        "sidecars_packv3_v2.md5",
        drive=[{"t": "dir_md5", "remote": "gdrive:research-archive/data/packs/sidecars/sidecars_packv3_v2"},
               {"t": "dir_md5", "remote": "gdrive:research-archive/worktrees/ws-packv3fix/_out/sidecars_packv3_v2"}])
    add("identities_dspeed", r"E:\rd-wt\a9\_out\identities_dspeed", "identities_dspeed", "identities_dspeed.md5",
        drive=[{"t": "dir_md5", "remote": "gdrive:research-archive/data/packs/sidecars/identities_dspeed"},
               {"t": "dir_md5", "remote": "gdrive:research-archive/worktrees/a9/_out/identities_dspeed"}])
    add("identities_damping", r"E:\rd-wt\damping-snap\_out\identities_damping", "identities_damping",
        "identities_damping.md5",
        drive=[{"t": "dir_md5", "remote": "gdrive:research-archive/data/packs/sidecars/identities_damping"},
               {"t": "dir_md5", "remote": "gdrive:research-archive/worktrees/damping-snap/_out/identities_damping"}])
    add("overlay_dspeed", r"E:\rd-wt\dspeed-data\_out\labels_overlay_dspeed", "overlay_dspeed",
        "labels_overlay_dspeed.md5",
        drive=[{"t": "dir_md5", "remote": "gdrive:research-archive/data/packs/sidecars/labels_overlay_dspeed"},
               {"t": "dir_md5", "remote": "gdrive:research-archive/worktrees/dspeed-data/_out/labels_overlay_dspeed"}])
    add("overlay_damping", r"E:\rd-wt\damping-snap\_out\labels_overlay_damping", "overlay_damping",
        "labels_overlay_damping.md5",
        drive=[{"t": "dir_md5", "remote": "gdrive:research-archive/data/packs/sidecars/labels_overlay_damping"},
               {"t": "dir_md5", "remote": "gdrive:research-archive/worktrees/damping-snap/_out/labels_overlay_damping"}])
    # lists (tracked in git in the single work tree, copied for a uniform registry)
    add("lpwo_v3", MAIN + r"\_out\lpwo_v3", "lpwo_v3", old="keep",
        note="tracked in git (_out/lpwo_v3 of the work tree); the cache copy is a convenience duplicate",
        drive=[{"t": "git_tracked", "repo": MAIN, "rel": "_out/lpwo_v3"}])
    # evaluation state sets: the sources sit inside worktrees that retire; the proof is the worktree's verified tar.zst
    # (members compared by size, and by SHA-256 against the cache copy)
    add("eval_states_v2", r"E:\rd-wt\ws-wallc1\_out\wallc1_states_v2", "wallc1_states_v2",
        drive=[{"t": "wt_tar", "worktree": "ws-wallc1", "rel": "_out/wallc1_states_v2"},
               {"t": "dir_listing", "remote": "gdrive:research-archive/worktrees/ws-wallc1/_out/wallc1_states_v2"}],
        note="source lives in ws-wallc1 (retires); proof = the worktree tar.zst on Drive", old="remove",
        registry="eval_states_dir_v2")
    add("eval_states_h60", r"E:\rd-wt\ws-r3-evalstates3\_out\wallc1_states_h60", "wallc1_states_h60",
        drive=[{"t": "wt_tar", "worktree": "ws-r3-evalstates3", "rel": "_out/wallc1_states_h60"},
               {"t": "dir_listing", "remote": "gdrive:research-archive/worktrees/ws-r3-evalstates3/_out/wallc1_states_h60"}],
        registry="eval_states_dir_h60")
    add("eval_states_h60_real", r"E:\rd-wt\ws-r3-evalstates3\_out\wallc1_states_h60_real", "wallc1_states_h60_real",
        drive=[{"t": "wt_tar", "worktree": "ws-r3-evalstates3", "rel": "_out/wallc1_states_h60_real"},
               {"t": "dir_listing", "remote": "gdrive:research-archive/worktrees/ws-r3-evalstates3/_out/wallc1_states_h60_real"}],
        registry="eval_states_dir_h60_real")
    # checkpoints
    add("ckpt_c1fast_dir", MAIN + r"\runs\active\WM-S-C1FAST-20261006\box\full_struct", "ckpt_c1fast", old="keep",
        note="C1 checkpoint dir (wm_s.pt + receipts); original stays under runs/ (frozen evidence path)",
        drive=[{"t": "archive_cover", "unit": "runs/active/WM-S-C1FAST-20261006"}], registry="ckpt_c1fast_dir")
    add("r3_results", MAIN + r"\runs\active\WM-S-R3-20261010\box\gpuA\wmn1_switching_train", "wmn1_switching_train",
        manifest=None, old="remove", complete=False,
        note="switching pack copy from gpuA (BOX_SHA256SUMS.txt is the authority; Stage 2 refuses it while *.part files or missing members exist). Round-3 results: add when box/gpuA results land.",
        ref_sha="BOX_SHA256SUMS.txt", drive=[{"t": "archive_cover", "unit": "runs/active/WM-S-R3-20261010"}],
        registry="pack_switching_train_dir")
    return K


# ---------------------------------------------------------------- pending / ETA
def queue_pending(items, notes):
    """pending = live bytes under each queue source - bytes covered by passing dataops receipts."""
    q = rj(DATAOPS + r"\_out\queue.json")
    excl = {key(x) for x in q.get("exclude", [])}
    covered_roots = {}
    for k, it in items.items():
        for p in it["proofs"]:
            if p["t"] in ("archive", "mirror", "legacy"):
                covered_roots[k] = it
    # parts: use member manifests to know which files each part covers
    covered_files = set()
    parent_part_dirs = set()
    for k, it in covered_roots.items():
        for p in it["proofs"]:
            if p["t"] != "archive":
                continue
            if len(p["parts"]) > 1 or any(PART_RE.search(x["id"]) for x in p["parts"]):
                parent_part_dirs.add(k)
                for part in p["parts"]:
                    try:
                        with open(part["manifest"], encoding="utf-8") as f:
                            for line in f:
                                if line.startswith("#") or not line.strip():
                                    continue
                                cols = line.rstrip("\n").split("\t")
                                if len(cols) >= 3:
                                    covered_files.add(key(it["path"] + "\\" + cols[2].replace("/", "\\")))
                    except OSError:
                        pass
    rows = []
    for qi in q["items"]:
        src = nrm(qi["src"])
        if key(src) in excl or not os.path.exists(lp(src)):
            continue
        files, nrep, nsens = walk_live(src)
        live = sum(s for _, s in files)
        pend_bytes = 0
        pend_files = 0
        sub = {}
        for fp, sz in files:
            fk = key(fp)
            cov = fk in covered_files
            if not cov:
                parts = fk.split("\\")
                for i in range(len(parts), 0, -1):
                    anc = "\\".join(parts[:i])
                    if anc in covered_roots and anc not in parent_part_dirs:
                        cov = True
                        break
            if not cov:
                pend_bytes += sz
                pend_files += 1
                if key(fp) == key(src):
                    rel = [os.path.basename(fp)]
                else:
                    rel = fp[len(src):].lstrip("\\").split("\\")
                bucket = rel[0] if len(rel) > 1 else "(files)"
                if qi["id"] in ("runs", "output", "shards_v3"):
                    bucket = "\\".join(rel[:2]) if (qi["id"] == "runs" and len(rel) > 2) else bucket
                s = sub.setdefault(bucket, [0, 0])
                s[0] += sz
                s[1] += 1
        rows.append({"id": qi["id"], "src": src, "live_bytes": live, "live_files": len(files),
                     "pending_bytes": pend_bytes, "pending_files": pend_files, "reparse": nrep, "sensitive": nsens,
                     "top": sorted(([k2, v[0], v[1]] for k2, v in sub.items()), key=lambda x: -x[1])[:12]})
    return rows


def upload_rates(notes):
    """Source-bytes/s of verified dataops units and files/s of verified worktree uploads."""
    rates = {}
    up = {}
    ev = []
    try:
        with open(DATAOPS + r"\data_progress.log", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.match(r"\[(\S+)\] \[pid=(\d+)\] (.*)", line)
                if not m:
                    continue
                t, _pid, msg = m.groups()
                m2 = re.match(r"UPLOAD (\S+) bytes=(\d+) zst=(\d+)", msg)
                if m2:
                    up[m2.group(1)] = (int(m2.group(2)), int(m2.group(3)), parse_ts(t))
                m3 = re.match(r"OK (\S+) verified; wire=([\d.]+) MB/s", msg)
                if m3 and m3.group(1) in up:
                    b, z, t0 = up[m3.group(1)]
                    ev.append((parse_ts(t), t0, b, z))
    except OSError:
        pass
    if ev:
        last_end = max(e[0] for e in ev)
        for label, mins in (("last60", 60), ("last180", 180)):
            sel = [e for e in ev if (last_end - e[0]).total_seconds() <= mins * 60]
            if len(sel) >= 2:
                t0 = min(e[1] for e in sel)
                el = max(1.0, (last_end - t0).total_seconds())
                rates["data_" + label] = {"src_Bps": sum(e[2] for e in sel) / el,
                                          "wire_Bps": sum(e[3] for e in sel) / el, "units": len(sel),
                                          "window_min": el / 60}
        big = [e for e in ev if e[2] >= 1e9]
        if big:
            el = sum(max(1.0, (e[0] - e[1]).total_seconds()) for e in big)
            rates["data_big_units"] = {"src_Bps": sum(e[2] for e in big) / el, "units": len(big)}
    # worktrees
    wev = []
    try:
        with open(WTCLEAN + r"\wt_progress.log", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = re.match(r"(\S+)\s+(\S+) ok upload\+check files=(\d+) bytes=(\d+)", line)
                if m:
                    wev.append((parse_ts(m.group(1)), int(m.group(3)), int(m.group(4))))
    except OSError:
        pass
    if len(wev) >= 3:
        recent = [e for e in wev if (wev[-1][0] - e[0]).total_seconds() <= 3 * 3600]
        if len(recent) >= 3:
            el = (recent[-1][0] - recent[0][0]).total_seconds()
            rates["wt_last3h"] = {"files_per_s": sum(e[1] for e in recent[1:]) / el,
                                  "Bps": sum(e[2] for e in recent[1:]) / el, "worktrees": len(recent) - 1}
    return rates


def free_space():
    out = {}
    for d in "CDEG":
        try:
            st = os.statvfs(d + ":\\") if hasattr(os, "statvfs") else None
        except Exception:
            st = None
        try:
            import shutil
            u = shutil.disk_usage(d + ":\\")
            out[d] = {"free": u.free, "total": u.total}
        except Exception:
            out[d] = None
    return out



def eta_text(nbytes, rate_bytes_s):
    if not rate_bytes_s or rate_bytes_s <= 0:
        return "n/a"
    return "%.1f h" % (nbytes / rate_bytes_s / 3600.0)


def pending_summary(rows, wt_pending, rates):
    """Plain-text lines + numbers describing what is not yet safe to delete (GiB = 2^30, MiB/s = 2^20 B/s)."""
    G, Mi = 2 ** 30, 2 ** 20
    lines = []
    num = {}
    d60 = rates.get("data_last60", {}).get("src_Bps")
    d180 = rates.get("data_last180", {}).get("src_Bps")
    big = rates.get("data_big_units", {}).get("src_Bps")
    pend_total = sum(r["pending_bytes"] for r in rows)
    pend_files = sum(r["pending_files"] for r in rows)
    num["data_pending_bytes"] = pend_total
    num["data_pending_files"] = pend_files
    lines.append("Dataops (tar.zst archives) pending: %.1f GiB in %d files; measured source rate %s MiB/s (last 60 min), %s MiB/s (last 3 h)." % (
        pend_total / G, pend_files, ("%.1f" % (d60 / Mi)) if d60 else "n/a", ("%.1f" % (d180 / Mi)) if d180 else "n/a"))
    lines.append("   ETA: %s at the last-hour rate, %s at the 3 h rate (%s at the big-units-only rate of %s MiB/s)." % (
        eta_text(pend_total, d60), eta_text(pend_total, d180), eta_text(pend_total, big), ("%.0f" % (big / Mi)) if big else "n/a"))
    for r in sorted(rows, key=lambda r: -r["pending_bytes"]):
        if r["pending_bytes"] < 1e7:
            continue
        lines.append("   %-34s %8.1f GiB pending of %8.1f GiB (%d files)" % (r["src"], r["pending_bytes"] / G, r["live_bytes"] / G, r["pending_files"]))
        for name, b, n in r["top"][:6]:
            lines.append("        %-60s %8.2f GiB (%d files)" % (name, b / G, n))
    wb = sum(x["bytes"] for x in wt_pending)
    wf = sum(x["files"] for x in wt_pending)
    num["wt_pending_bytes"] = wb
    num["wt_pending_files"] = wf
    num["wt_pending_n"] = len(wt_pending)
    w = rates.get("wt_last3h", {})
    fps = w.get("files_per_s")
    mbs = w.get("Bps")
    eta_f = (wf / fps / 3600.0) if fps else None
    lines.append("Worktree backups (wtclean) pending: %d worktrees, %d files, %.1f GiB; measured %.2f files/s and %.3f MiB/s (last 3 h)." % (
        len(wt_pending), wf, wb / G, fps or 0, (mbs or 0) / Mi))
    lines.append("   ETA: ~%s (file-count bound: Drive creates ~%.1f small files/s; the byte rate is not meaningful here). Largest: %s" % (
        ("%.0f h" % eta_f) if eta_f else "n/a",
        fps or 0,
        ", ".join("%s %.1f GiB/%d files" % (x["name"], x["bytes"] / G, x["files"]) for x in sorted(wt_pending, key=lambda x: -x["bytes"])[:6])))
    num["wt_eta_files_h"] = eta_f
    num["data_eta_h_60"] = (pend_total / d60 / 3600.0) if d60 else None
    num["data_eta_h_180"] = (pend_total / d180 / 3600.0) if d180 else None
    num["data_eta_h_big"] = (pend_total / big / 3600.0) if big else None
    return lines, num



# ---------------------------------------------------------------- CLEANUP_SUMMARY.md generated block
GEN_BEGIN = "<!-- GEN:BEGIN (written by gen_cleanup.py --summary; do not edit by hand) -->"
GEN_END = "<!-- GEN:END -->"


def _load_json(path):
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def _gib(n):
    return "%.1f" % (n / 2 ** 30)


def dir_size(path):
    files, _r, _s = walk_live(path)
    return sum(s for _, s in files), len(files)


def summary_block(manifest, state, keep):
    G = 2 ** 30
    r1 = _load_json(HERE + r"\report_stage1_latest.json")
    r2 = _load_json(HERE + r"\report_stage2_latest.json")
    r3 = _load_json(HERE + r"\report_stage3_latest.json")
    out = []
    w = out.append
    w("_Snapshot generated %s (manifest %s). Stage 1 report run %s (%s%s); Stage 2 report run %s._" % (
        datetime.datetime.now().isoformat(timespec="minutes"), manifest["generated"],
        r1["run"] if r1 else "n/a", r1["mode"] if r1 else "-", "" if not r1 else ", flags as in the test command below",
        r2["run"] if r2 else "n/a"))
    w("")
    free = state["free"]
    # ---------------- Stage 1 by drive
    w("### Stage 1 (remove Drive-verified items): what it would free now")
    w("")
    if r1:
        res = r1["results"]
        el = [x for x in res if x["status"] == "ELIGIBLE"]
        optional = [x for x in el if x["reason"].startswith("PARTIAL") or x["reason"].startswith("released by")]
        w("| drive | free now (GiB) | eligible items | would free (GiB), report flags | of which needs `-RemoveAroundProtected`/`-AllowSensitiveDup` (GiB) | default flags (GiB) |")
        w("|---|---:|---:|---:|---:|---:|")
        for d in "CDEG":
            sel = [x for x in el if x["drive"] == d]
            opt = [x for x in optional if x["drive"] == d]
            tot = sum(x["bytes"] for x in sel)
            ob = sum(x["bytes"] for x in opt)
            fr = free.get(d)
            w("| %s: | %s | %d | %s | %s | %s |" % (d, _gib(fr["free"]) if fr else "n/a", len(sel), _gib(tot), _gib(ob), _gib(tot - ob)))
        w("")
        w("Eligible items by kind: " + ", ".join("%s %d" % (k, v) for k, v in sorted(
            {t: sum(1 for x in el if x["type"] == t) for t in ("item", "worktree")}.items())) + ".")
        w("")
        w("Top eligible items (GiB):")
        w("")
        w("| GiB | drive | kind | path | note |")
        w("|---:|---|---|---|---|")
        for x in sorted(el, key=lambda x: -x["bytes"])[:16]:
            w("| %s | %s | %s | `%s` | %s |" % (_gib(x["bytes"]), x["drive"], x["type"], x["path"], (x["reason"] or "")[:90]))
        w("")
        w("### What stays, grouped by the first failing check")
        w("")
        w("| status | items | GiB | meaning | example |")
        w("|---|---:|---:|---|---|")
        meaning = {
            "SENSITIVE_SKIPPED": "worktree has sensitive-NAMED untracked files that the uploader skipped (e.g. holdout_guard.py, __pycache__/cal_split)",
            "SENSITIVE_INSIDE": "item contains protected-name entries that were never uploaded (sealed/CAL/HOLDOUT/chunk-5/INC1)",
            "REFERENCED": "active code/config still names this path (clean tree + main tree scan)",
            "LINKS": "junction/symlink inside: manual handling",
            "DIRTY_TRACKED": "worktree has uncommitted tracked changes (wtclean has not committed yet)",
            "NEW_UNBACKED": "untracked files exist that are not in the uploaded list",
            "RECEIPT_NOT_PASSED": "wtclean receipt not ok (check failed)",
            "NOT_REGISTERED": "not a registered worktree of the main repo",
            "BACKUP_ONLY": "class KEEP-ACTIVE-backup-only (label sidecars): backed up, not offered",
            "NESTED": "inside another eligible item (counted once, under the parent)",
            "KEEP_LOCAL": "is/contains a keep-local item (Stage 2 first)",
            "LOCAL_CHANGED": "local files changed after the backup was taken",
            "LOCAL_NOT_IN_BACKUP": "local files exist that are not in the backup manifest",
            "DRIVE_UNVERIFIED": "Drive copy missing or different at its current path",
            "RECEIPT_STALE": "manifest/list file changed since the generator ran (regenerate)",
            "IN_USE": "a running process mentions the path",
            "TRACKED": "git-tracked files inside the repo",
            "ERROR": "evaluation error",
        }
        by = {}
        for x in res:
            if x["status"] in ("ELIGIBLE", "GONE", "DELETED"):
                continue
            by.setdefault(x["status"], []).append(x)
        for st, xs in sorted(by.items(), key=lambda kv: -sum(x["bytes"] for x in kv[1])):
            ex = sorted(xs, key=lambda x: -x["bytes"])[0]
            w("| %s | %d | %s | %s | `%s` |" % (st, len(xs), _gib(sum(x["bytes"] for x in xs)), meaning.get(st, ""), ex["path"]))
        w("")
    else:
        w("(no Stage 1 report found: run `MORNING_CLEANUP.ps1 -Stage 1`)")
        w("")
    # ---------------- pending
    w("### Pending upload (never deleted; ETA from the measured dataops rates)")
    w("")
    w("```")
    for ln in manifest["pending_info"]["lines"]:
        ln = re.sub(r"\(best case ([0-9.]+ h) if big units keep streaming at ([0-9]+) MiB/s\)", r"(\1 at the big-units-only rate of \2 MiB/s)", ln)
        w(ln)
    w("```")
    w("")
    # ---------------- keep-local + space plan
    w("### Keep-local hot data (Stage 2) and the space plan")
    w("")
    w("| name | source | GiB | files | cache dir | Stage 2 state (report) | old copy after verify |")
    w("|---|---|---:|---:|---|---|---|")
    s2 = {x["name"]: x for x in (r2["stage2"] if r2 and r2.get("stage2") else [])}
    sizes = {}
    for k in keep:
        b, n = dir_size(k["source"]) if os.path.exists(lp(k["source"])) else (0, 0)
        sizes[k["name"]] = b
        st = s2.get(k["name"], {})
        w("| %s | `%s` | %s | %d | `data_cache/%s` | %s | %s |" % (
            k["name"], k["source"], _gib(b), n, k["cache"], st.get("state", "-"),
            "remove (Drive copy verified: %s)" % ("yes" if st.get("drive_ok") else "not yet") if k["old_policy"] == "remove" else "keep (%s)" % ("tracked/frozen path" if k["old_policy"] == "keep" else k["old_policy"])))
    w("")
    e_free = free["E"]["free"] if free.get("E") else 0
    d_free = free["D"]["free"] if free.get("D") else 0
    e_s1 = d_s1 = 0
    if r1:
        for x in r1["results"]:
            if x["status"] == "ELIGIBLE":
                if x["drive"] == "E":
                    e_s1 += x["bytes"]
                elif x["drive"] == "D":
                    d_s1 += x["bytes"]
    order = sorted(keep, key=lambda k: sizes.get(k["name"], 0))
    margin = 4 * G
    w("Space plan for **E:** (copy order = size ascending, margin 4 GiB, free space re-read before every copy):")
    w("")
    w("| step | action | GiB | E: free after |")
    w("|---|---|---:|---:|")
    cur = e_free
    w("| 0 | now | | %s |" % _gib(cur))
    cur += e_s1
    w("| 1 | Stage 1 `-Execute` (E: items incl. partials) | +%s | %s |" % (_gib(e_s1), _gib(cur)))
    for k in order:
        need = sizes.get(k["name"], 0)
        if need < 0.05 * G:
            continue
        ok = cur >= need + margin
        if ok:
            cur -= need
        w("| 2 | copy `%s` to data_cache | %s%s | %s |" % (k["name"], "-" if ok else "", _gib(need) if ok else _gib(need) + " (DEFER: needs +4 GiB margin)", _gib(cur)))
    small = sum(sizes.get(k["name"], 0) for k in order if sizes.get(k["name"], 0) < 0.05 * G)
    cur -= small
    w("| 2 | copy the small items (lists, sidecars, identities, overlays, checkpoint) | -%s | %s |" % (_gib(small), _gib(cur)))
    reclaim = sum(sizes.get(k["name"], 0) for k in keep if k["old_policy"] == "remove" and k["source"].upper().startswith("E:"))
    reclaim_note = "; the switching-pack duplicate under runs/active is included"
    w("| 3 | Stage 2b: remove old E: pack/state copies once their Drive copy verifies (`-RemoveOld`) | +%s | %s |" % (_gib(reclaim), _gib(cur + reclaim)))
    w("")
    w("**D:** free now %s GiB; Stage 1 frees %s GiB (shards, upload_v2/v2c, raw_chunk1_p40, pack_v3_r3_subset, cpu_payload, ...); after `pack_v3` is cached on E: and its Drive copy verifies (the `training_pack_v3_k192` tar parts already do), 2b frees another %s GiB on D:." % (
        _gib(d_free), _gib(d_s1), _gib(sizes.get("pack_v3", 0))))
    w("")
    # ---------------- D: leftovers
    w("### What would still be on D: after Stage 1 (D:\\rd-staging, first two levels)")
    w("")
    w("| path | GiB | why it stays |")
    w("|---|---:|---|")
    status = {}
    if r1:
        for x in r1["results"]:
            status[key(x["path"])] = x
    keepkeys = {key(k["source"]): k for k in keep}
    elig_keys = [key(p) for p, x in ((x["path"], x) for x in (r1["results"] if r1 else [])) if x["status"] in ("ELIGIBLE", "NESTED")]

    def has_elig_below(p):
        kp = key(p) + "\\"
        return any(e.startswith(kp) for e in elig_keys)

    def leftovers(p, depth):
        x = status.get(key(p))
        if x and x["status"] in ("ELIGIBLE", "NESTED", "GONE"):
            return
        if depth < 3 and os.path.isdir(lp(p)) and has_elig_below(p) and key(p) not in keepkeys:
            try:
                kids = sorted(os.listdir(lp(p)))
            except OSError:
                kids = []
            for kid in kids:
                leftovers(p + "\\" + kid, depth + 1)
            return
        b, n = dir_size(p) if os.path.isdir(lp(p)) else ((os.path.getsize(lp(p)), 1) if os.path.exists(lp(p)) else (0, 0))
        if key(p) in keepkeys:
            why = "keep-local (Stage 2 -> data_cache, then old copy removed after Drive proof)"
        elif x:
            why = "%s: %s" % (x["status"], (x["reason"] or "")[:110])
        elif "_dataops_tmp" in p:
            why = "dataops temp area (uploader scratch; empties itself)"
        else:
            why = "no passing receipt (upload pending or not queued)"
        w("| `%s` | %s | %s |" % (p, _gib(b), why))

    base = r"D:\rd-staging"
    for kid in sorted(os.listdir(lp(base))):
        leftovers(base + "\\" + kid, 1)
    w("")
    return "\n".join(out) + "\n"


def refresh_summary(manifest, state):
    keep = manifest["keep_local"]
    block = summary_block(manifest, state, keep)
    path = HERE + r"\CLEANUP_SUMMARY.md"
    try:
        with open(path, encoding="utf-8") as f:
            md = f.read()
    except OSError:
        md = ""
    if GEN_BEGIN in md and GEN_END in md:
        a = md.index(GEN_BEGIN) + len(GEN_BEGIN)
        b = md.index(GEN_END)
        md = md[:a] + "\n" + block + md[b:]
    else:
        md = md + "\n" + GEN_BEGIN + "\n" + block + GEN_END + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(md)
    print("CLEANUP_SUMMARY.md generated block refreshed")


# ---------------------------------------------------------------- main
def main():
    global OUT, HERE, RECEIPTS, REORG, DRIVEOPS_R, DATAOPS_R, WTCLEAN_R, DRIVEOPS, DATAOPS, WTCLEAN, PROTECTED_WORKTREES
    ap = argparse.ArgumentParser(description="regenerate MORNING_CLEANUP.ps1 + cleanup_manifest.json")
    ap.add_argument("--summary", action="store_true", help="refresh the generated block of CLEANUP_SUMMARY.md")
    ap.add_argument("--pending", action="store_true",
                    help="also walk the dataops queue sources for the 'pending upload' tables (slow; off by default)")
    ap.add_argument("--summary-only", action="store_true",
                    help="only refresh the generated block of CLEANUP_SUMMARY.md from the last manifest + stage reports")
    ap.add_argument("--out", default=None, help="output folder (default <repo>/data_ops/cleanup_out)")
    ap.add_argument("--receipts-root", default=None,
                    help="folder holding dataops/ driveops/ wtclean/ receipts (default <repo>/data_ops/receipts)")
    ap.add_argument("--reorg3-receipts", action="store_true",
                    help=r"comparison mode: read the receipts from E:\rd-wt\reorg3 (the original location)")
    ap.add_argument("--heavy-root", default=None,
                    help=r"folder with the heavy generated parts (manifests, lists, _out); default E:\rd-wt\reorg3")
    ap.add_argument("--protect", action="append", default=[], metavar="NAME",
                    help="worktree name that must never be removed (repeatable; the default list is empty)")
    ap.add_argument("--carry-from", action="append", default=[], metavar="MANIFEST",
                    help="previous manifest(s) to carry sensitive-named-receipt worktree entries from")
    args = ap.parse_args()

    if args.out:
        OUT = nrm(os.path.abspath(args.out))
    HERE = OUT
    if args.reorg3_receipts:
        RECEIPTS = r"E:\rd-wt\reorg3"
    elif args.receipts_root:
        RECEIPTS = nrm(os.path.abspath(args.receipts_root))
    if args.heavy_root:
        REORG = nrm(os.path.abspath(args.heavy_root))
    DRIVEOPS_R, DATAOPS_R, WTCLEAN_R = RECEIPTS + r"\driveops", RECEIPTS + r"\dataops", RECEIPTS + r"\wtclean"
    DRIVEOPS, DATAOPS, WTCLEAN = REORG + r"\driveops", REORG + r"\dataops", REORG + r"\wtclean"
    PROTECTED_WORKTREES = list(PROTECTED_WORKTREES) + list(args.protect)
    os.makedirs(OUT, exist_ok=True)

    if args.summary_only:
        with open(OUT + r"\cleanup_manifest.json", encoding="utf-8") as f:
            man0 = json.load(f)
        with open(OUT + r"\gen_state.json", encoding="utf-8") as f:
            st0 = json.load(f)
        st0["free"] = free_space()
        refresh_summary(man0, st0)
        return

    carry_from = [nrm(p) for p in args.carry_from] + [OUT + r"\cleanup_manifest.json",
                                                        MAIN + r"\tmp\morning\cleanup_manifest.json"]
    notes = []
    items = {}
    n_d = collect_driveops(items, notes)
    n_a = collect_dataops(items, notes)
    n_l = collect_legacy(items, notes)
    special_items(items, notes)
    wts, wt_unreceipted, wt_gone, wt_carried, wt_skipped_names = collect_worktrees(notes, carry_from)
    keep = keep_local_items()

    # drop candidates the generator already knows to be out of bounds (recorded for the report)
    known_held = []
    cand = []
    for k, it in sorted(items.items()):
        if path_is_sensitive(it["path"]):
            known_held.append({"path": it["path"], "reason": "protected name (sealed/CAL/HOLDOUT/chunk-5/INC1)"})
            continue
        if not any(under(it["path"], r) for r in ALLOWED_ROOTS):
            known_held.append({"path": it["path"], "reason": "outside the deletion allow-list"})
            continue
        if not it["proofs"] and not it.get("special"):
            continue
        cand.append(it)

    pending_rows = queue_pending(items, notes) if args.pending else []
    rates = upload_rates(notes) if args.pending else {}
    if args.pending:
        plines, pnum = pending_summary(pending_rows, [], rates)
    else:
        plines, pnum = [], {}
    manifest = {
        "schema": "morning_cleanup_manifest/2",
        "pending_info": {"lines": plines, "numbers": pnum},
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "generator_version": GEN_VERSION,
        "paths": {"rclone": RCLONE, "main": MAIN, "clean": MAIN, "cache": CACHE,
                  "out": OUT, "reorg3": REORG, "receipts": RECEIPTS, "dataops": DATAOPS, "driveops": DRIVEOPS,
                  "wtclean": WTCLEAN, "rdwt": RDWT, "sealed_root": SEALED_ROOT,
                  "sealed_manifest": SEALED_ROOT + r"\MOVE_MANIFEST.csv",
                  "journal": OUT + r"\journal.log", "removed_map": OUT + r"\removed_paths.jsonl"},
        "allowed_roots": ALLOWED_ROOTS,
        "container_roots": CONTAINER_ROOTS,
        "generic_literals": GENERIC_LITERALS,
        "protected": {"exact": sorted(SENS_EXACT), "contains": list(SENS_CONTAINS), "broad_regex": BROAD_SENS_RE,
                      "roots": [OUT, MAIN + r"\.git", CACHE]},
        "hold_until_flag": [{"path": REORG, "flag": "RetireReorg",
                             "why": "holds the upload lists / member manifests that every worktree check reads; "
                                    "retire it LAST with -RetireReorg (removed only if no other worktree remains)"}],
        "ref_scan": {
            "dirs": {MAIN: ["traffic_shepherd", "baselines", "tools", "tests", "scripts", "configs", "src", "protocols"]},
            "root_files": ["*.py", "*.toml", "*.json", "*.sh", "*.ps1", "*.cfg", "*.ini", "*.yaml", "*.yml"],
            "ignore_files": IGNORE_REF_FILES,
        },
        "keep_local": keep,
        "items": cand,
        "worktrees": wts,
        "unreceipted_dirs": wt_unreceipted,
        "worktrees_gone_at_generation": wt_gone,
        "worktree_receipts_not_opened": wt_skipped_names,
        "active_worktrees": [],
        "protected_worktrees": PROTECTED_WORKTREES,
        "gen_known_held": known_held,
        "counts": {"driveops_proofs": n_d, "dataops_receipts": n_a, "legacy": n_l, "items": len(cand),
                   "worktrees_receipted": len(wts), "worktrees_carried": len(wt_carried),
                   "worktrees_unreceipted": len(wt_unreceipted), "worktrees_already_gone": len(wt_gone)},
    }

    with open(OUT + r"\cleanup_manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=True)

    # PowerShell script = template + embedded manifest
    tpl_path = TOOL_DIR + r"\morning_cleanup.template.ps1"
    with open(tpl_path, encoding="utf-8") as f:
        tpl = f.read()
    blob = json.dumps(manifest, ensure_ascii=True, separators=(",", ":"))
    if "\n'@" in blob:
        raise SystemExit("manifest contains a here-string terminator")
    if len(blob) > 1_900_000:
        print("WARNING: manifest %d bytes may exceed ConvertFrom-Json limits on Windows PowerShell 5.1" % len(blob))
    ps = tpl.replace("@@MANIFEST_JSON@@", blob).replace("@@GENERATED@@", manifest["generated"])
    ps = ps.replace("@@OUT_DIR@@", OUT).replace("@@TOOL_DIR@@", TOOL_DIR)
    ps = ps.replace("\r\n", "\n")
    with open(OUT + r"\MORNING_CLEANUP.ps1", "w", encoding="ascii", newline="\n") as f:
        f.write(ps)

    state = {"pending_rows": pending_rows, "rates": rates, "notes": notes,
             "free": free_space(), "manifest_counts": manifest["counts"], "known_held": known_held,
             "generated": manifest["generated"]}
    with open(OUT + r"\gen_state.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(state, f, indent=1)

    print("generated:", manifest["generated"])
    print("out:", OUT, "  receipts:", RECEIPTS, "  heavy:", REORG)
    print("Stage 1 items: %d   worktrees (receipts, dir present): %d (carried %d)   dirs without receipt: %d   receipts of already-removed dirs: %d" %
          (len(cand), len(wts), len(wt_carried), len(wt_unreceipted), len(wt_gone)))
    print("manifest bytes: %d; MORNING_CLEANUP.ps1 bytes: %d" % (len(blob), len(ps)))
    for n in notes:
        print("NOTE:", n)
    if args.summary:
        refresh_summary(manifest, state)


if __name__ == "__main__":
    main()

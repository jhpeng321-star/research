#!/usr/bin/env python3
"""verify_reorg3_retirable.py - is E:\\rd-wt\\reorg3 safe to remove?  (READ-ONLY; optional --bundle-dir writes git bundles)

reorg3 holds the data-ops tooling and the Drive receipts/lists/manifests that MORNING_CLEANUP.ps1 reads, so it is
retired LAST and by hand.  This script classifies every file of reorg3 (junctions are never followed, nested .git
folders are not walked):

  drive      member of reorg3.tar.txt (the verified tar.zst on Drive: heavy manifests/lists/inventory/_out/logs)
  repo       copied into the repository (archive/rdwt_results_20261010/PATH_MAP.json row whose copy exists with the same
             size, or the same relative path under data_ops/receipts or tools/dataops with equal size + MD5)
  junk       __pycache__, .pytest_cache, *.pyc, _tmp, locks
  sensitive  sensitive-named (names only, never opened): must already be in sealed\\ (MOVE_TO_SEALED.ps1)
  RESIDUE    none of the above: copy it somewhere or decide it is disposable before removing reorg3

and lists the nested git repositories (their history is NOT in the tar.zst): with --bundle-dir DIR it writes one
`git bundle` per nested repository so the history can be uploaded before the folder is removed.

Exit code 0 = no residue.  Usage:
    python -I tools\\dataops\\cleanup\\verify_reorg3_retirable.py [--root E:\\rd-wt\\reorg3] [--bundle-dir DIR] [--show 40]
"""
import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
import sys

sys.dont_write_bytecode = True
BS = chr(92)
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
SENS = re.compile(r"(?i)(holdout|heldout|cal_split|chunk-?5|j4b|inc1|sealed|(^|[\\/_.\-\s])cal([\\/_.\-\s]|$))")
JUNK = re.compile(r"(?i)(^|/)(__pycache__|\.pytest_cache|_tmp|locks)(/|$)|\.pyc$")


def lp(p):
    p = os.path.normpath(p)
    return p if p.startswith("\\\\?\\") else "\\\\?\\" + p


def md5(path):
    h = hashlib.md5()
    with open(lp(path), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"E:\rd-wt\reorg3")
    ap.add_argument("--bundle-dir", default=None)
    ap.add_argument("--show", type=int, default=40)
    a = ap.parse_args()
    root = a.root.replace("/", BS).rstrip(BS)
    lst_path = root + r"\wtclean\lists\reorg3.tar.txt"
    lst = set()
    if os.path.exists(lst_path):
        with open(lst_path, encoding="utf-8") as f:
            lst = {l.rstrip("\r\n") for l in f if l.strip()}
    else:
        print("WARNING: %s missing: nothing counts as 'on Drive'" % lst_path)
    pm = {}
    pmp = os.path.join(REPO, "archive", "rdwt_results_20261010", "PATH_MAP.json")
    if os.path.exists(pmp):
        with open(pmp, encoding="utf-8-sig") as f:
            for r in json.load(f).get("files", []):
                pm[r["old"].replace(BS, "/").lower()] = r
    cls = collections.defaultdict(list)
    nested = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = list(os.scandir(lp(d)))
        except OSError as exc:
            cls["RESIDUE"].append("%s (unreadable: %s)" % (d, exc))
            continue
        for e in it:
            full = os.path.join(d, e.name)
            st = e.stat(follow_symlinks=False)
            if getattr(st, "st_file_attributes", 0) & 0x400:
                cls["RESIDUE"].append(full + " (junction/symlink: unlink it first)")
                continue
            if e.is_dir(follow_symlinks=False):
                if e.name == ".git":
                    nested.append(d)
                    continue
                stack.append(full)
                continue
            rel = os.path.relpath(full, root).replace(BS, "/")
            if rel in lst:
                cls["drive"].append(rel)
            elif SENS.search(rel):
                cls["sensitive"].append(rel)
            elif JUNK.search(rel):
                cls["junk"].append(rel)
            else:
                r = pm.get(("e:/rd-wt/reorg3/" + rel).lower())
                ok = False
                if r and os.path.exists(r["new"]) and os.path.getsize(r["new"]) == st.st_size:
                    ok = True
                if not ok:
                    for base in ("data_ops/receipts", "tools/dataops"):
                        cand = os.path.join(REPO, base.replace("/", BS), rel.replace("/", BS))
                        if os.path.exists(cand) and os.path.getsize(cand) == st.st_size and md5(cand) == md5(full):
                            ok = True
                            break
                (cls["repo"] if ok else cls["RESIDUE"]).append(rel)
    for k in ("drive", "repo", "junk", "sensitive", "RESIDUE"):
        print("%-10s %6d" % (k, len(cls[k])))
    for x in sorted(cls["RESIDUE"])[: a.show]:
        print("   RESIDUE", x)
    print("nested git repositories (history NOT on Drive): %d" % len(nested))
    for d in sorted(nested):
        r = subprocess.run(["git", "-C", d, "rev-list", "--all", "--count"], capture_output=True, text=True)
        print("   %-40s commits=%s" % (os.path.relpath(d, root), r.stdout.strip() or "?"))
        if a.bundle_dir:
            os.makedirs(a.bundle_dir, exist_ok=True)
            name = os.path.relpath(d, root).replace(BS, "_").replace("/", "_") or "reorg3"
            out = os.path.join(a.bundle_dir, "reorg3_nested_%s.bundle" % name)
            b = subprocess.run(["git", "-C", d, "bundle", "create", out, "--all"], capture_output=True, text=True)
            print("      bundle ->", out, "rc=%d" % b.returncode, (b.stderr or "").strip()[:120])
    sys.exit(1 if cls["RESIDUE"] else 0)


if __name__ == "__main__":
    main()

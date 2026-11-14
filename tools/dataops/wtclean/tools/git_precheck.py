#!/usr/bin/env python3
"""git_precheck.py - read-only: for every non-active worktree with an ok-ish receipt, check the template's git
pre-conditions (clean tracked tree, HEAD reachable from a branch/tag in the main repo)."""
import json, glob, os, subprocess, sys
sys.dont_write_bytecode = True
BASE = r"E:\rd-wt\reorg3\wtclean"; MAIN = r"E:\research-dense"
def g(*a, cwd=None):
    r = subprocess.run(["git"] + list(a), capture_output=True, text=True, cwd=cwd, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout.strip(), r.stderr.strip()
inv = {t["name"]: t for t in json.load(open(BASE + r"\inventory\inventory.json", encoding="utf-8-sig"))}
bad = []
n = 0
for f in sorted(glob.glob(BASE + r"\receipts\*.json")):
    r = json.load(open(f, encoding="utf-8-sig")); nm = r["name"]; t = inv.get(nm)
    if not t or t.get("active") or t["class"] != "worktree" or not os.path.isdir(t["path"]):
        continue
    n += 1
    wt = t["path"]
    rc, st, err = g("-C", wt, "status", "--porcelain=v1", "--untracked-files=no")
    rc2, head, _ = g("-C", wt, "rev-parse", "HEAD")
    rc3, br, _ = g("-C", wt, "rev-parse", "--abbrev-ref", "HEAD")
    prob = []
    if rc != 0: prob.append("git status failed: " + err[:80])
    elif st: prob.append("DIRTY_TRACKED %d: %s" % (len(st.splitlines()), st.splitlines()[0][:70]))
    if br == "HEAD":
        rc4, o, _ = g("-C", MAIN, "for-each-ref", "--contains", head, "--count=1", "--format=%(refname)", "refs/heads", "refs/remotes", "refs/tags")
        if not o: prob.append("COMMIT_UNREACHABLE " + head[:8])
    else:
        rc4, o, _ = g("-C", MAIN, "rev-parse", "--verify", "--quiet", "refs/heads/" + br)
        if rc4 != 0: prob.append("BRANCH_MISSING " + br)
        elif o != head:
            rc5, _, _ = g("-C", MAIN, "merge-base", "--is-ancestor", head, "refs/heads/" + br)
            if rc5 != 0: prob.append("BRANCH_MISMATCH %s head %s" % (br, head[:8]))
    if prob: bad.append((nm, r.get("status"), r.get("method"), prob))
print("checked", n, "worktree receipts; problems:", len(bad))
for b in bad: print(" ", b)

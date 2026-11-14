#!/usr/bin/env python3
"""mirror_keeplocal.py - per-file Drive mirrors of the SMALL keep-local sub-directories that
E:/research-dense/tmp/morning/gen_cleanup.py proves with dir_md5 under worktrees/<wt>/_out/<dir>.
(The two large packs, pack_dspeed 12 GB and pack_damping 5 GB, are NOT mirrored here.)
Additive upload + rclone check only; sources are only read.  Lists come from the tar manifests
(already filtered for sensitive names / junctions)."""
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
BASE = r"E:\rd-wt\reorg3\wtclean"
RCLONE = r"C:\Users\jpeng\miniconda3\bin\rclone.exe"
ITEMS = [("ws-packv3fix", "_out/sidecars_packv3_v2"), ("a9", "_out/identities_dspeed"),
         ("damping-snap", "_out/identities_damping"), ("damping-snap", "_out/labels_overlay_damping"),
         ("dspeed-data", "_out/labels_overlay_dspeed")]


def main():
    out = []
    for wt, sub in ITEMS:
        man = json.load(open(BASE + r"\manifests\%s.manifest.json" % wt, encoding="utf-8"))
        rels = [m["path"][len(sub) + 1:] for m in man["members"] if m["path"].startswith(sub + "/")]
        src = r"E:\rd-wt\%s\%s" % (wt, sub.replace("/", "\\"))
        disk = sum(len(fs) for _, _, fs in os.walk(src))
        lst = BASE + r"\lists\mirror_%s__%s.txt" % (wt, sub.replace("/", "_"))
        open(lst, "w", encoding="utf-8", newline="\n").write("".join(r + "\n" for r in rels))
        dest = "gdrive:research-archive/worktrees/%s/%s" % (wt, sub)
        r1 = subprocess.run([RCLONE, "copy", src, dest, "--files-from", lst, "--transfers", "4", "--checksum",
                             "--stats", "0", "--timeout", "60s", "--retries", "3"], capture_output=True, text=True)
        r2 = subprocess.run([RCLONE, "check", src, dest, "--files-from", lst, "--one-way", "--checksum"],
                            capture_output=True, text=True)
        res = {"worktree": wt, "dir": sub, "files_listed": len(rels), "files_on_disk": disk, "dest": dest,
               "copy_rc": r1.returncode, "check_rc": r2.returncode,
               "check_tail": [l for l in (r2.stderr + r2.stdout).splitlines() if "differences" in l or "matching" in l][-2:]}
        print(json.dumps(res), flush=True)
        out.append(res)
    json.dump(out, open(BASE + r"\receipts_mirror_keeplocal.json", "w"), indent=1)


if __name__ == "__main__":
    main()

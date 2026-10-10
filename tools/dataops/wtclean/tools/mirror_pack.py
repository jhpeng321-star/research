#!/usr/bin/env python3
"""mirror_pack.py WT SUBDIR [TRANSFERS] - per-file Drive mirror of one large keep-local pack dir (additive upload +
rclone check --checksum). List comes from the tar manifest (sensitive-name filtered). Sources are only read."""
import json, os, subprocess, sys
sys.dont_write_bytecode = True
BASE = r"E:\rd-wt\reorg3\wtclean"; RCLONE = r"C:\Users\jpeng\miniconda3\bin\rclone.exe"
wt, sub = sys.argv[1], sys.argv[2]; tr = sys.argv[3] if len(sys.argv) > 3 else "6"
man = json.load(open(BASE + r"\manifests\%s.manifest.json" % wt, encoding="utf-8"))
rels = [m["path"][len(sub) + 1:] for m in man["members"] if m["path"].startswith(sub + "/")]
src = r"E:\rd-wt\%s\%s" % (wt, sub.replace("/", "\\"))
disk = sum(len(fs) for _, _, fs in os.walk(src))
lst = BASE + r"\lists\mirror_%s__%s.txt" % (wt, sub.replace("/", "_"))
open(lst, "w", encoding="utf-8", newline="\n").write("".join(r + "\n" for r in rels))
dest = "gdrive:research-archive/worktrees/%s/%s" % (wt, sub)
print("files listed", len(rels), "on disk", disk, flush=True)
for attempt in (1, 2, 3):
    r1 = subprocess.run([RCLONE, "copy", src, dest, "--files-from", lst, "--transfers", tr, "--drive-chunk-size", "64M",
                         "--checksum", "--stats", "60s", "--stats-one-line", "-v", "--retries", "3",
                         "--low-level-retries", "10"])
    if r1.returncode == 0:
        break
r2 = subprocess.run([RCLONE, "check", src, dest, "--files-from", lst, "--one-way", "--checksum"], capture_output=True, text=True)
print(json.dumps({"worktree": wt, "dir": sub, "files_listed": len(rels), "files_on_disk": disk, "dest": dest,
                  "copy_rc": r1.returncode, "check_rc": r2.returncode,
                  "check_tail": [l for l in (r2.stderr + r2.stdout).splitlines() if "differences" in l or "matching" in l][-2:]}), flush=True)

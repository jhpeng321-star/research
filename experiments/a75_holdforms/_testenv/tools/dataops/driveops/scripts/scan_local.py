#!/usr/bin/env python3
"""Fast metadata-only sizes for a list of local paths. Skips reparse points (junctions/symlinks)."""
import os, sys, json, stat

def scan(root):
    files = 0
    total = 0
    errors = 0
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for e in it:
                    try:
                        if e.is_symlink():
                            continue
                        st = e.stat(follow_symlinks=False)
                        if stat.S_ISDIR(st.st_mode):
                            stack.append(e.path)
                        elif stat.S_ISREG(st.st_mode):
                            files += 1
                            total += st.st_size
                    except OSError:
                        errors += 1
        except OSError:
            errors += 1
    return total, files, errors

paths = json.load(open(sys.argv[1], encoding="utf-8")) if len(sys.argv) > 1 else []
out = {}
for p in paths:
    if os.path.isdir(p):
        t, f, e = scan(p)
        out[p] = {"kind": "dir", "bytes": t, "files": f, "errors": e, "exists": True}
    elif os.path.isfile(p):
        try:
            st = os.stat(p, follow_symlinks=False)
            out[p] = {"kind": "file", "bytes": st.st_size, "files": 1, "errors": 0, "exists": True}
        except OSError as ex:
            out[p] = {"exists": False, "error": str(ex)}
    else:
        out[p] = {"exists": False}
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
for k, v in out.items():
    print(f"{v.get('bytes',0):>14} {v.get('files',0):>7} {v.get('exists')}  {k}")

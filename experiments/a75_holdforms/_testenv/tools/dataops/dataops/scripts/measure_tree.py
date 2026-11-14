# measure_tree.py -- fast metadata-only tree sizer. Never follows junctions/symlinks.
# Usage: python measure_tree.py PATH [PATH ...]
# Prints one line per path: TAB files<TAB bytes<TAB dirs<TAB junctions
import os, sys, stat

def is_reparse(st):
    return (getattr(st, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT) != 0

def measure(root):
    files = 0; bytes_ = 0; dirs = 0; junc = 0
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if is_reparse(st):
                    junc += 1
                    continue
                if e.is_dir(follow_symlinks=False):
                    dirs += 1
                    stack.append(e.path)
                elif e.is_file(follow_symlinks=False):
                    files += 1; bytes_ += st.st_size
    return files, bytes_, dirs, junc

for p in sys.argv[1:]:
    p = os.path.abspath(p)
    if not os.path.exists(p):
        print(f"MISSING\t{p}", flush=True); continue
    f, b, dd, j = measure(p)
    print(f"OK\t{p}\tfiles={f}\tbytes={b}\tdirs={dd}\tjunctions={j}", flush=True)

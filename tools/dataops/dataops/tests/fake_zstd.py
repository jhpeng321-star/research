"""Fake zstd: copies stdin to the -o output unchanged (no compression)."""
import sys

out = None
a = sys.argv[1:]
for i, t in enumerate(a):
    if t == "-o" and i + 1 < len(a):
        out = a[i + 1]
# real zstd treats a bare "2" after "-T" as an input file; reject stray positionals
pos = [t for i, t in enumerate(a) if not t.startswith("-") and not (i > 0 and a[i - 1] == "-o")]
if pos:
    sys.stderr.write("fake zstd: unexpected input files %r" % pos)
    sys.exit(1)
data = sys.stdin.buffer.read()
if out:
    with open(out, "wb") as f:
        f.write(data)
sys.exit(0)

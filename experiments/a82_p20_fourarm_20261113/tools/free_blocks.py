import re, sys, json
from pathlib import Path
srcs = [Path(p) for p in sys.argv[1:]]
pat = re.compile(r'(?<![\d.])(9\d{5})(?:\s*(?:-|–|\.\.|to)\s*(\d{2,6}))?(?![\d])')
iv = []
for p in srcs:
    s = p.read_text(encoding='utf-8', errors='replace')
    for m in pat.finditer(s):
        a = int(m.group(1)); b = m.group(2)
        if b:
            b = int(b) if len(b) == 6 else int(str(a)[:6-len(b)] + b)
            if b < a or b - a > 20000: b = a
        else:
            b = a
        iv.append((a, b))
used = sorted(set(iv))
def free(lo, hi):
    return not any(a <= hi and b >= lo for a, b in used)
out = []
for start in range(900000, 1000000, 100):
    if free(start, start+399):
        out.append(start)
print(json.dumps(dict(n_intervals=len(used), free_400_starts=out)))

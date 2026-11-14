"""AMENDMENT 82 part B (A82 four-arm confirmation, 20 %) table-only edit (idempotent, zero SUMO): the fresh 400-seed
block A82_RANGES = ((988200, 988599),) appended to the runner seed-range table of the A76 runner (79fc9efd)
(physics-neutral runtime registration, labels only; same pattern as ops/a76/host/patch_tables_a76.py).
usage: patch_tables_a82.py tables <nm_eval_runner.py>
Refuses: CRLF file, a forbidden block, an overlap with any (lo, hi) pair already in the file, or a file already carrying
a different A82_RANGES.
"""
from pathlib import Path
import re
import sys

BLOCKS = ((988200, 988599),)
ANCHOR = "A76_RANGES = ((997100, 997199), (997200, 997299), (997300, 997399))\n"
TAIL = "                       (lo, hi, lo) for lo, hi in A76_RANGES)"
TAIL_NEW = ("                       (lo, hi, lo) for lo, hi in A76_RANGES) + tuple(\n"
            "                       (lo, hi, lo) for lo, hi in A82_RANGES)")
ADD = ("#: amendment 82 (A82 four-arm confirmation at D2800_p20; DRAFT): FRESH audited block 988200-988599 (the first\n"
       "#: 96 / 192 / 384 used, as registered; the rest reserved unused); repeat = seed - block start. Physics-neutral\n"
       "#: runtime registration (labels only; TRAIN roles still refuse them).\n"
       "A82_RANGES = ((988200, 988599),)\n")
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699), (987200, 987599))


def once(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f'anchor not found exactly once: {old[:60]!r}')
    return s.replace(old, new)


def runner(s):
    m = re.search(r'^A82_RANGES = (.*)$', s, re.M)
    if m:
        if m.group(1) != '((988200, 988599),)':
            raise SystemExit(f'runner already carries a different A82_RANGES {m.group(1)}')
        return s
    for lo, hi in BLOCKS:
        if any(lo <= b1 and hi >= b0 for b0, b1 in FORBIDDEN):
            raise SystemExit(f'{lo}-{hi}: forbidden block')
        for a, b in re.findall(r'\((\d{6}), (\d{6})\)', s):
            if lo <= int(b) and hi >= int(a):
                raise SystemExit(f'{lo}-{hi} overlaps the runner range ({a}, {b})')
    s = once(s, ANCHOR, ANCHOR+ADD)
    return once(s, TAIL, TAIL_NEW)


def main():
    if len(sys.argv) != 3 or sys.argv[1] != 'tables':
        raise SystemExit(__doc__)
    p = Path(sys.argv[2])
    s = p.read_bytes().decode('utf-8')
    if '\r\n' in s:
        raise SystemExit(f'{p}: CRLF file; convert first')
    t = runner(s)
    if t != s:
        p.write_bytes(t.encode('utf-8'))
    print('patched' if t != s else 'unchanged', p)


if __name__ == '__main__':
    main()

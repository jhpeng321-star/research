"""AMENDMENT 96 part (c) table-only edit (idempotent, zero SUMO): the audited fresh block A96C_RANGES = ((984400, 984599),)
(984400-984591 used, 984592-984599 reserved unused) appended to the runner seed-range table of the A82 part-B runner
(2de0f1c0). Copy of patch_tables_a82.py with the A96C names; physics-neutral runtime registration, labels only.
usage: patch_tables_a96c.py tables <nm_eval_runner.py>
Refuses: CRLF file, a forbidden block, an overlap with any (lo, hi) pair already in the file, or a file already carrying
a different A96C_RANGES.
"""
from pathlib import Path
import re
import sys

BLOCKS = ((984400, 984599),)
ANCHOR = "A82_RANGES = ((988200, 988599),)\n"
TAIL = "                       (lo, hi, lo) for lo, hi in A82_RANGES)"
TAIL_NEW = ("                       (lo, hi, lo) for lo, hi in A82_RANGES) + tuple(\n"
            "                       (lo, hi, lo) for lo, hi in A96C_RANGES)")
ADD = ("#: amendment 96 part (c) (matched 20 % panel at D2800_p20): FRESH audited block 984400-984599 (984400-984591\n"
       "#: used, the rest reserved unused); repeat = seed - block start. Physics-neutral runtime registration (labels\n"
       "#: only; TRAIN roles still refuse them).\n"
       "A96C_RANGES = ((984400, 984599),)\n")
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699), (987200, 987599),
             (988200, 988999), (989000, 989699), (992400, 992523))


def once(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f'anchor not found exactly once: {old[:60]!r}')
    return s.replace(old, new)


def runner(s):
    m = re.search(r'^A96C_RANGES = (.*)$', s, re.M)
    if m:
        if m.group(1) != '((984400, 984599),)':
            raise SystemExit(f'runner already carries a different A96C_RANGES {m.group(1)}')
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

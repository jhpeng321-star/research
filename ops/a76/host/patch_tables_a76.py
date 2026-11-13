"""AMENDMENT 76 (A76-D2P20) table-only edit (idempotent, zero SUMO): the three fresh 100-seed blocks
A76_RANGES = ((997100, 997199), (997200, 997299), (997300, 997399)) appended to the runner seed-range table
(physics-neutral runtime registration, labels only; same pattern as ops/a73/host/patch_tables.py and
ops/a75d2/host/patch_tables_d2.py).
usage: patch_tables_a76.py tables <nm_eval_runner.py>
Refuses: CRLF file, a forbidden block, an overlap with any (lo, hi) pair already in the file, or a file already carrying
a different A76_RANGES.
"""
from pathlib import Path
import re
import sys

BLOCKS = ((997100, 997199), (997200, 997299), (997300, 997399))
ANCHOR = "A44_MECH_RANGE = (992400, 992499)\n"
TAIL = "(A44_MECH_RANGE[0], A44_MECH_RANGE[1], A44_MECH_RANGE[0]),)"
TAIL_NEW = ("(A44_MECH_RANGE[0], A44_MECH_RANGE[1], A44_MECH_RANGE[0]),) + tuple(\n"
            "                       (lo, hi, lo) for lo, hi in A76_RANGES)")
ADD = ("#: amendment 76 (A76-D2P20): FRESH audited blocks 997100-997199 (mechanics, 4 used), 997200-997299 (admission,\n"
       "#: 24 used), 997300-997399 (confirmation, 96 used); repeat = seed - block start. Physics-neutral runtime\n"
       "#: registration (labels only; TRAIN roles still refuse them).\n"
       "A76_RANGES = ((997100, 997199), (997200, 997299), (997300, 997399))\n")
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699))


def once(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f'anchor not found exactly once: {old[:60]!r}')
    return s.replace(old, new)


def runner(s):
    m = re.search(r'^A76_RANGES = (.*)$', s, re.M)
    if m:
        if m.group(1) != '((997100, 997199), (997200, 997299), (997300, 997399))':
            raise SystemExit(f'runner already carries a different A76_RANGES {m.group(1)}')
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

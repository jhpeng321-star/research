"""AMENDMENT 81 table-only edit (idempotent, zero SUMO): the audited fresh 400-seed panel block A81_RANGES =
((990100, 990499),) appended to the runner seed-range table (physics-neutral runtime registration, labels only; same
pattern as ops/a75d2/host/patch_tables_d2.py; repeat = seed - 990100, as the A33 192-world block 999500-999699).
usage: patch_tables_a81.py tables <nm_eval_runner.py>
Refuses: CRLF file, a forbidden block, an overlap with any (lo, hi) pair already in the file, a file already carrying a
different A81_RANGES, or a runner that is not the A75-D2 runner layout (anchor lines).
"""
from pathlib import Path
import re
import sys

LO, HI = 990100, 990499
ANCHOR = "A75D2_MECH_RANGE = (997400, 997499)\n"
TAIL = "(A75D2_MECH_RANGE[0], A75D2_MECH_RANGE[1], A75D2_MECH_RANGE[0]),)"
TAIL_NEW = ("(A75D2_MECH_RANGE[0], A75D2_MECH_RANGE[1], A75D2_MECH_RANGE[0]),) + tuple(\n"
            "                       (lo, hi, lo) for lo, hi in A81_RANGES)")
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699))
ADD = ("#: amendment 81 (S-BIGCONFIRM-A81, post-hoc): FRESH audited panel block 990100-990499 (384 used; repeat = seed -\n"
       "#: block start). Same physics-neutral runtime registration (labels only; TRAIN roles still refuse them).\n"
       "A81_RANGES = ((990100, 990499),)\n")


def once(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f'anchor not found exactly once: {old[:60]!r}')
    return s.replace(old, new)


def runner(s):
    if any(LO <= b1 and HI >= b0 for b0, b1 in FORBIDDEN):
        raise SystemExit(f'{LO}-{HI}: forbidden block')
    m = re.search(r'^A81_RANGES = (.*)$', s, re.M)
    if m:
        if m.group(1) != f'(({LO}, {HI}),)':
            raise SystemExit(f'runner already carries A81_RANGES {m.group(1)}')
        return s
    for a, b in re.findall(r'\((\d{6}), (\d{6})\)', s):
        if LO <= int(b) and HI >= int(a):
            raise SystemExit(f'{LO}-{HI} overlaps the runner range ({a}, {b})')
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

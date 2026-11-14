"""AMENDMENT 75 stage D2 table-only edit (idempotent, zero SUMO): the coordinator-assigned 100-seed mechanics block
A75D2_MECH_RANGE = (S0, S0+99) appended to the runner seed-range table (physics-neutral runtime registration, labels
only; same pattern as ops/a73/host/patch_tables.py).
usage: patch_tables_d2.py tables <nm_eval_runner.py> <S0>
Refuses: CRLF file, S0 not a multiple of 100, a forbidden block, an overlap with any (lo, hi) pair already in the file,
or a file already carrying a different A75D2_MECH_RANGE.
"""
from pathlib import Path
import re
import sys

ANCHOR = "A44_MECH_RANGE = (992400, 992499)\n"
TAIL = "(A44_MECH_RANGE[0], A44_MECH_RANGE[1], A44_MECH_RANGE[0]),)"
TAIL_NEW = ("(A44_MECH_RANGE[0], A44_MECH_RANGE[1], A44_MECH_RANGE[0]),\n"
            "                       (A75D2_MECH_RANGE[0], A75D2_MECH_RANGE[1], A75D2_MECH_RANGE[0]),)")
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699))


def add(lo):
    return ("#: amendment 75 stage D2 (S-HOLDFORMS-D2): coordinator-assigned audited mechanics block %d-%d (4 used;\n"
            "#: repeat = seed - block start). Physics-neutral runtime registration (labels only).\n"
            "A75D2_MECH_RANGE = (%d, %d)\n" % (lo, lo+99, lo, lo+99))


def once(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f'anchor not found exactly once: {old[:60]!r}')
    return s.replace(old, new)


def runner(s, lo):
    if lo % 100 or not 900000 <= lo <= 999900:
        raise SystemExit(f'{lo}: not a 100-seed block start')
    hi = lo+99
    if any(lo <= b1 and hi >= b0 for b0, b1 in FORBIDDEN):
        raise SystemExit(f'{lo}-{hi}: forbidden block')
    m = re.search(r'^A75D2_MECH_RANGE = \((\d+), (\d+)\)$', s, re.M)
    if m:
        if (int(m.group(1)), int(m.group(2))) != (lo, hi):
            raise SystemExit(f'runner already carries A75D2_MECH_RANGE {m.group(1)}-{m.group(2)} != {lo}-{hi}')
        return s
    for a, b in re.findall(r'\((\d{6}), (\d{6})\)', s):
        if lo <= int(b) and hi >= int(a):
            raise SystemExit(f'{lo}-{hi} overlaps the runner range ({a}, {b})')
    s = once(s, ANCHOR, ANCHOR+add(lo))
    return once(s, TAIL, TAIL_NEW)


def main():
    if len(sys.argv) != 4 or sys.argv[1] != 'tables':
        raise SystemExit(__doc__)
    p = Path(sys.argv[2])
    s = p.read_bytes().decode('utf-8')
    if '\r\n' in s:
        raise SystemExit(f'{p}: CRLF file; convert first')
    t = runner(s, int(sys.argv[3]))
    if t != s:
        p.write_bytes(t.encode('utf-8'))
    print('patched' if t != s else 'unchanged', p)


if __name__ == '__main__':
    main()

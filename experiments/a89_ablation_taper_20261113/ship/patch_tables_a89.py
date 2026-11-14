"""AMENDMENT 89 table-only edit (idempotent, zero SUMO). Copy of patch_tables_a81.py (ffb3dfbb); CHANGED lines are
marked "# A89". The audited fresh block A89_RANGES = ((988400, 988499),) (988400-988495 used, 988496-988499 reserved
unused) is appended to the runner seed-range table of the A81 runner (538bd0bf), physics-neutral, labels only.
usage: patch_tables_a89.py tables <nm_eval_runner.py>
Refuses: CRLF file, a forbidden block, an overlap with any (lo, hi) pair already in the file, a file already carrying a
different A89_RANGES, or a runner that is not the A81 runner layout (anchor lines).
"""
from pathlib import Path
import re
import sys

LO, HI = 988400, 988499                                                    # A89
ANCHOR = "A81_RANGES = ((990100, 990499),)\n"                               # A89
TAIL = "(lo, hi, lo) for lo, hi in A81_RANGES)"                              # A89
TAIL_NEW = ("(lo, hi, lo) for lo, hi in A81_RANGES) + tuple(\n"               # A89
            "                       (lo, hi, lo) for lo, hi in A89_RANGES)")
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699))
ADD = ("#: amendment 89 (A89, post-hoc): FRESH audited panel block 988400-988499 (96 used, 988400-988495; repeat =\n"  # A89
       "#: seed - block start). Same physics-neutral runtime registration (labels only; TRAIN roles still refuse them).\n"
       "A89_RANGES = ((988400, 988499),)\n")


def once(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f'anchor not found exactly once: {old[:60]!r}')
    return s.replace(old, new)


def runner(s):
    if any(LO <= b1 and HI >= b0 for b0, b1 in FORBIDDEN):
        raise SystemExit(f'{LO}-{HI}: forbidden block')
    m = re.search(r'^A89_RANGES = (.*)$', s, re.M)                          # A89
    if m:
        if m.group(1) != f'(({LO}, {HI}),)':
            raise SystemExit(f'runner already carries A89_RANGES {m.group(1)}')
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

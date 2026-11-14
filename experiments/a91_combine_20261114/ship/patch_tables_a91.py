"""AMENDMENT 91 table-only edit (idempotent, zero SUMO). Copy of patch_tables_a89.py (eb94d315); CHANGED lines are
marked "# A91". The audited fresh block A91_RANGES = ((988600, 988699),) (988600-988695 used, 988696-988699 reserved
unused) is appended to the runner seed-range table of the A89 runner (982ec0cf), physics-neutral, labels only.
usage: patch_tables_a91.py tables <nm_eval_runner.py>
Refuses: CRLF file, a forbidden block, an overlap with any (lo, hi) pair already in the file, a file already carrying a
different A91_RANGES, or a runner that is not the A89 runner layout (anchor lines).
"""
from pathlib import Path
import re
import sys

LO, HI = 988600, 988699                                                    # A91
ANCHOR = "A89_RANGES = ((988400, 988499),)\n"                               # A91
TAIL = "(lo, hi, lo) for lo, hi in A89_RANGES)"                              # A91
TAIL_NEW = ("(lo, hi, lo) for lo, hi in A89_RANGES) + tuple(\n"               # A91
            "                       (lo, hi, lo) for lo, hi in A91_RANGES)")
FORBIDDEN = ((964000, 964199), (962000, 962999), (977000, 977999), (989600, 989699), (992000, 992999),  # A91
             (996000, 996999), (988200, 988599))                                                      # A91
ADD = ("#: amendment 91 (A91, post-hoc): FRESH audited panel block 988600-988699 (96 used, 988600-988695; repeat =\n"  # A91
       "#: seed - block start). Same physics-neutral runtime registration (labels only; TRAIN roles still refuse them).\n"
       "A91_RANGES = ((988600, 988699),)\n")


def once(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f'anchor not found exactly once: {old[:60]!r}')
    return s.replace(old, new)


def runner(s):
    if any(LO <= b1 and HI >= b0 for b0, b1 in FORBIDDEN):
        raise SystemExit(f'{LO}-{HI}: forbidden block')
    m = re.search(r'^A91_RANGES = (.*)$', s, re.M)                          # A91
    if m:
        if m.group(1) != f'(({LO}, {HI}),)':
            raise SystemExit(f'runner already carries A91_RANGES {m.group(1)}')
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

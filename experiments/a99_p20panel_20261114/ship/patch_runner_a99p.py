"""A99 PANEL runner table-only edit (idempotent, zero SUMO): the A96(c) audited block A96C_RANGES = ((984400, 984599),)
(984400-984591 used, 984592-984599 reserved unused) appended to the seed-range table of the A99 admission runner
(0c5c2a4b, which carries A96_RANGES 988900-988999). Same lines as patch_tables_a96c.py, anchored on A96_RANGES;
physics-neutral runtime registration, labels only (repeat = seed - 984400, as in the A96(c) staged panel).
usage: patch_runner_a99p.py tables <nm_eval_runner.py>
"""
from pathlib import Path
import re
import sys

BLOCKS = ((984400, 984599),)
ANCHOR = "A96_RANGES = ((988900, 988999),)\n"
TAIL = "                       (lo, hi, lo) for lo, hi in A96_RANGES)"
TAIL_NEW = ("                       (lo, hi, lo) for lo, hi in A96_RANGES) + tuple(\n"
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

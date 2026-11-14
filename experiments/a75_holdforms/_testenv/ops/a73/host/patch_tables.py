"""AMENDMENT 73 table-only edit (idempotent, zero SUMO): one more seed block in the runner seed-range table.
usage: patch_tables.py tables <nm_eval_runner.py>
The A73 settings are registered process-locally by ops/a73_lcside20.py (nm_finaleval is not edited).
"""
from pathlib import Path
import sys

#: the newest block already in the tree (A70 in the laptop / small-bank tree; A68 in the A69 host tree)
ANCHORS = (("A70_RANGES = ((996300, 996399),)\n", "A70_RANGES"),
           ("A68_RANGES = ((992500, 992599), (996200, 996299))\n", "A68_RANGES"))
RUNNER_ADD = ("#: amendment 73 (S-LCSIDE20): FRESH audited confirmation block 996400-996499 (96 used; repeat = seed - block\n"
              "#: start). Same physics-neutral runtime registration (labels only; TRAIN roles still refuse them). The A73\n"
              "#: mechanics / verification worlds 992408-992415 / 992524-992547 lie inside the A44 / A68 blocks below.\n"
              "A73_RANGES = ((996400, 996499),)\n")
TUPLE = "                       (lo, hi, lo) for lo, hi in %s)"
TUPLE_ADD = " + tuple(\n                       (lo, hi, lo) for lo, hi in A73_RANGES)"


def rw(path, fn):
    p = Path(path)
    s = p.read_bytes().decode('utf-8')
    if '\r\n' in s:
        raise SystemExit(f'{p}: CRLF file; convert first')
    t = fn(s)
    if t != s:
        p.write_bytes(t.encode('utf-8'))
    print('patched' if t != s else 'unchanged', p)


def once(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f'anchor not found exactly once: {old[:60]!r}')
    return s.replace(old, new)


def runner(s):
    if 'A73_RANGES' in s:
        return s
    line, name = next(((a, n) for a, n in ANCHORS if a in s), (None, None))
    if line is None:
        raise SystemExit('no known seed-range anchor in the runner')
    s = once(s, line, line+RUNNER_ADD)
    return once(s, TUPLE % name, TUPLE % name+TUPLE_ADD)


def main():
    if len(sys.argv) != 3 or sys.argv[1] != 'tables':
        raise SystemExit(__doc__)
    rw(sys.argv[2], runner)


if __name__ == '__main__':
    main()

"""AMENDMENT 77 table-only edit (idempotent, zero SUMO): one more seed block in the runner seed-range table.
usage: patch_tables.py tables <nm_eval_runner.py>
The A77 setting is registered process-locally by ops/a77_mappo20c.py (nm_finaleval is not edited).
Same code as ops/a73/host/patch_tables.py; anchors extended to the A73 (laptop) and A50 (A66 host) trees.
"""
from pathlib import Path
import sys

#: the newest block already in the tree (A73 laptop tree, A70 small-bank tree, A68 A69 tree, A50 A66 host tree)
ANCHORS = (("A73_RANGES = ((996400, 996499),)\n", "A73_RANGES"),
           ("A70_RANGES = ((996300, 996399),)\n", "A70_RANGES"),
           ("A68_RANGES = ((992500, 992599), (996200, 996299))\n", "A68_RANGES"),
           ("A50_RANGES = ((992800, 992899), (996800, 996899))\n", "A50_RANGES"))
RUNNER_ADD = ("#: amendment 77 (S-MAPPO20C): FRESH audited MAPPO comparator panel block 997600-997699 (96 used; repeat =\n"
              "#: seed - block start). Same physics-neutral runtime registration (labels only; TRAIN roles still refuse them).\n"
              "A77_RANGES = ((997600, 997699),)\n")
TUPLE = "                       (lo, hi, lo) for lo, hi in %s)"
TUPLE_ADD = " + tuple(\n                       (lo, hi, lo) for lo, hi in A77_RANGES)"


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
    if 'A77_RANGES' in s:
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

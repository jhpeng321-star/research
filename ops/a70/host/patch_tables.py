"""AMENDMENT 70 table-only edits (idempotent, zero SUMO): the runner seed-range table and the nm_finaleval settings.
usage: patch_tables.py tables <runner.py> <nm_finaleval.py>
       patch_tables.py pin <nm_finaleval.py> <a70_smallbank.py> <panel_sha256>
"""
from pathlib import Path
import sys

RUNNER_ANCHOR = "A68_RANGES = ((992500, 992599), (996200, 996299))\n"
RUNNER_ADD = ("#: amendment 70 (S-SMALLBANK60C, post-hoc): FRESH audited panel block 996300-996399 (96 used; repeat = seed - block\n"
              "#: start). Same physics-neutral runtime registration (labels only; TRAIN roles still refuse them); same A37 executor arms.\n"
              "A70_RANGES = ((996300, 996399),)\n")
TUPLE_OLD = "                       (lo, hi, lo) for lo, hi in A68_RANGES) + (\n"
TUPLE_NEW = ("                       (lo, hi, lo) for lo, hi in A68_RANGES) + tuple(\n"
             "                       (lo, hi, lo) for lo, hi in A70_RANGES) + (\n")
TUPLE_END = "                       (lo, hi, lo) for lo, hi in A68_RANGES)\n"
TUPLE_END_NEW = ("                       (lo, hi, lo) for lo, hi in A68_RANGES) + tuple(\n"
                 "                       (lo, hi, lo) for lo, hi in A70_RANGES)\n")
SET_ANCHOR = "MAP_SPEC_REL = 'assets/research/newmap_candidates/wz3to2/MAP_SPEC.json'\n"
SET_ADD = '''#: amendment 70 (S-SMALLBANK60C, POST-HOC small-bank test): FRESH audited panel 996300-996395 (996396-996399 reserved
#: unused). Same plant / bindings as a35_D2800_p60 (= the A42 / A50 / A68 cells); executed through the eval runner
#: (scope label only); never pooled.
A70_CELLS = (('a70c_D2800_p60', 996300, 96, 'small-bank panel', 'nm_eval_panel_wz3to2_a70c_D2800_p60.json'),)
for _name, _lo, _n, _what, _file in A70_CELLS:
    SETTINGS[_name] = dict(SETTINGS['a35_D2800_p60'],
        N_WORLDS=_n, EVAL_RANGE=(_lo, _lo+99), EVAL_EXTRA_RANGE=(_lo, _lo+99), SEEDS_RESERVED=(_lo+_n, _lo+99),
        PANEL_SIZE_REASON=(f'MAIN amendment 70: {_what} = {_n} fresh worlds {_lo}-{_lo+_n-1} (D2800_p60); '
                           f'{_lo+_n}-{_lo+99} reserved unused'),
        CELL_AUTHORITY=(f'MAIN amendment 70 (S-SMALLBANK60C, post-hoc): D2800_p60 {_what} of the best A68-admitted '
                        'plans as a small bank on top of MAPPO60C'),
        PANEL_AUTHORITY=(f'A70 amendment 70: fresh audited integers {_lo}-{_lo+_n-1} (D2800_p60) for the {_what} '
                         'ONLY; executed through the eval runner (scope label only); never pooled'),
        EXPOSURE_EXTRA=dict(
            checked=('amendment-70 audit: 996300-996399 as standalone integers absent from REGISTRATION.md seed use, '
                     'protocols/experiment_history.json, every c4/* branch tip, the run and agent dirs '
                     '(ops/a70/SEED_AUDIT_RAW.json)',),
            disclosed=f'; amendment 70: fresh integers ({_what}), registered at runtime by the runner'),
        PANEL_FILE=HERE/_file, PANEL_3C_FILE=None)
BANK_SETTINGS = BANK_SETTINGS + tuple(c[0] for c in A70_CELLS)
'''
PIN_LINE = ("#: amendment 70 block (pinned after the panel build, before any amendment-70 episode)\n"
            "SETTINGS_PINS['a70c_D2800_p60'] = dict(PANEL_SHA256='%s', PANEL_3C_SHA256=None)\n")
DRV_OLD = "PANEL_SHA256 = dict(c=None)"


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


def main():
    cmd = sys.argv[1]
    if cmd == 'tables':
        def runner(s):
            if 'A70_RANGES' in s:
                return s
            s = once(s, RUNNER_ANCHOR, RUNNER_ANCHOR+RUNNER_ADD)
            if TUPLE_OLD in s:       # tree with the A44 mechanics range after the A68 ranges (laptop branch)
                return once(s, TUPLE_OLD, TUPLE_NEW)
            return once(s, TUPLE_END, TUPLE_END_NEW)     # the A68 host tree: the A68 ranges end the table
        rw(sys.argv[2], runner)
        rw(sys.argv[3], lambda s: s if 'A70_CELLS' in s else once(s, SET_ANCHOR, SET_ADD+SET_ANCHOR))
    elif cmd == 'pin':
        sha = sys.argv[4]
        if len(sha) != 64:
            raise SystemExit('panel sha')
        rw(sys.argv[2], lambda s: s if PIN_LINE % sha in s else once(s, SET_ANCHOR, PIN_LINE % sha+SET_ANCHOR))
        rw(sys.argv[3], lambda s: s if f"PANEL_SHA256 = dict(c='{sha}')" in s
           else once(s, DRV_OLD, f"PANEL_SHA256 = dict(c='{sha}')"))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()

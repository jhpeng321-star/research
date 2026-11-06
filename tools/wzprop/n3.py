"""The vendored N3/PLANGEN verification pipeline, bound to a run root.

n3_pipeline/ holds VERBATIM copies of the cpuV PLANGEN code (runs/active/C4-V7FT-20261017/plangen/
code, identical decision rules to NEWMAP2 nm2_plan/nm2_analyze incl. amendments 3-4):
    pg_plan.py     funnel planner + statistics (boot_lb, rep_eval, best_per_family, rep1_picks, ...)
    pg_analyze.py  verdict (admission), v2v/RSU tags, example rows (coopmwm-mined-move-v1)
    nm2_core.py    one SUMO job per process (libsumo); JOB.json -> RESULT.json
    population.py  scenario HDV population (verbatim traffic_shepherd/c4/population.py)
Their bytes are pinned (raw and LF-normalised sha256); verify() refuses any drift.

bind(root, ...) points the module globals at a WM-C run root WITHOUT editing the code:
BASE/RUNS/LEDGER/WORLDS_FILE/MAPSPEC/CAP, and optionally native_dir -> a source root's natives (the
same saved t0 states as the reference arms) and worlds -> a WORLDS.json checked by the wz3to2 guard.
Every admission statistic and rule therefore comes from the vendored code itself.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from . import geometry as G

VENDOR = Path(__file__).resolve().parent / 'n3_pipeline'
SOURCE = 'E:/research-dense/runs/active/C4-V7FT-20261017/plangen/code (cpuV /home/ubuntu/plangen_20261106/code)'
PINS = {
    'pg_plan.py': dict(sha256='e8de81f5dc5588d0f8f9a826f2632c75abb251a229a23ae84ae497b4386cd44a',
                       lf_sha256='e8de81f5dc5588d0f8f9a826f2632c75abb251a229a23ae84ae497b4386cd44a'),
    'pg_analyze.py': dict(sha256='cc70b0fef3a54211ba20419ee837311bedb2501b2e4225638621d4d28aa2a32e',
                          lf_sha256='1332644612a6ed7fd0068fd8519c24ca161d439edb4375b41303763ec6848056'),
    'nm2_core.py': dict(sha256='ad6b4bd63426ca41f6ebf8b974db48eea508532f076981f652139de15fa20abb',
                        lf_sha256='ad6b4bd63426ca41f6ebf8b974db48eea508532f076981f652139de15fa20abb'),
    'population.py': dict(sha256='6af5ee232dd99f5ae0c142fdaf1b7de50f23f04d58f70def61482b76f4b2e6c6',
                          lf_sha256='5e8c61a6c1d5195e2830a5f6825e7a0ed4fc0c3260e243880ab956ddbff5849b'),
}
CORE = VENDOR / 'nm2_core.py'
_ORIG = {}


class VendorDriftError(RuntimeError):
    """A vendored N3 file differs from its pinned bytes."""


def lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def verify(vendor=VENDOR):
    out = {}
    for name, pin in PINS.items():
        got = lf_sha256(Path(vendor) / name)
        if got != pin['lf_sha256']:
            raise VendorDriftError(f'{name}: lf_sha256 {got} != pinned {pin["lf_sha256"]}')
        out[name] = got
    return out


def modules():
    """(pg_plan, pg_analyze) imported from the vendored directory (one module object per process)."""
    if 'pg_plan' not in sys.modules:
        verify()
        if str(VENDOR) not in sys.path:
            sys.path.insert(0, str(VENDOR))
    import pg_analyze
    import pg_plan
    if Path(pg_plan.__file__).resolve().parent != VENDOR or Path(pg_analyze.__file__).resolve().parent != VENDOR:
        raise VendorDriftError('pg_plan/pg_analyze were imported from outside the vendored directory')
    if not _ORIG:
        _ORIG.update(native_dir=pg_plan.native_dir, worlds=pg_plan.worlds, CAP=pg_plan.CAP)
    return pg_plan, pg_analyze


def read_worlds(path):
    """WORLDS.json [{demand, penetration, seed[, wave]}] -> [(d, p, s)]; wz3to2 TRAIN seeds only."""
    out = []
    for w in json.loads(G.check_path(path).read_text()):
        out.append((int(w['demand']), float(w['penetration']), G.check_seed(w['seed'], where=f'({path})')))
    if len(set(out)) != len(out):
        raise ValueError(f'duplicate worlds in {path}')
    return out


def bind(root, *, natives_root=None, map_spec=None, cap=None):
    """Point the vendored pg_plan at `root` (runs/, ledger/, MOMENTS.json, WORLDS.json)."""
    P, A = modules()
    root = G.check_path(Path(root).resolve())
    P.BASE = root
    P.RUNS = root / 'runs'
    P.LEDGER = root / 'ledger' / 'CHARGES.jsonl'
    P.WORLDS_FILE = root / 'WORLDS.json'
    nat_base = G.check_path(Path(natives_root).resolve()) if natives_root else root
    P.MAPSPEC = Path(map_spec) if map_spec else nat_base / 'maps' / 'wz3to2' / 'MAP_SPEC.json'
    P.CAP = float(cap) if cap is not None else _ORIG['CAP']
    if natives_root:
        nat_runs = nat_base / 'runs'
        P.native_dir = lambda d, p, s: nat_runs / P.cell_key(d, p) / f's{s}' / 'native'
    else:
        P.native_dir = _ORIG['native_dir']
    P.worlds = lambda: read_worlds(P.WORLDS_FILE)
    return P, A


def bound_paths():
    P, _ = modules()
    return dict(BASE=str(P.BASE), RUNS=str(P.RUNS), LEDGER=str(P.LEDGER), MAPSPEC=str(P.MAPSPEC), CAP=P.CAP,
                native_dir_example=str(P.native_dir(3200, 0.6, 989200)))

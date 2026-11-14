"""Zero-SUMO unit tests of the A81 driver (pure parts + the shipped runner patch)."""
import importlib.util
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a81_bigconfirm as M  # noqa: E402


def test_constants():
    assert M.CONFIRM_N == 384 and M.IDENT_N == 2 and M.HORIZON == 1200
    assert (M.MECH_CAP, M.CONFIRM_CAP, M.CAP_SIM_S) == (7_200, 1_382_400, 1_389_600)
    assert M.SEEDS_C == tuple(range(990100, 990484)) and M.RESERVED_C == (990484, 990499)
    assert M.ARMS == ('A75D2_BANK_60C', 'DUO_HOLD_60C', 'MAPPO60C')
    assert M.BANK_SHA256.startswith('c325ec1d') and M.SECONDARY == (('A75D2_BANK_60C', 'MAPPO60C'),
                                                                    ('DUO_HOLD_60C', 'MAPPO60C'))
    for lo, hi in M.FORBIDDEN:
        assert not (M.BLOCK[0] <= hi and M.BLOCK[1] >= lo)


def test_block_free():
    ok = ((993000, 993299, 993000), (996300, 996399, 996300), (990100, 990499, 990100))
    assert M.block_free(M.BLOCK, ok)
    with pytest.raises(ValueError):
        M.block_free(M.BLOCK, ok[:2])                                   # not registered
    with pytest.raises(ValueError):
        M.block_free(M.BLOCK, ok + ((990400, 990599, 990400),))         # overlap
    with pytest.raises(ValueError):
        M.block_free(M.BLOCK, ((990100, 990499, 990000),))              # wrong base
    with pytest.raises(ValueError):
        M.block_free((964100, 964499), ((964100, 964499, 964100),))     # forbidden


def _row(ci, all_pass):
    return dict(total_minus_ref=dict(diff=-1.0, ci95=list(ci)), groups=dict(all_pass=all_pass), tail_cvar90={})


def test_primary_rule():
    assert M.primary_verdict(_row((-5, -1), True))['verdict'] == 'SUPPORTED'
    assert M.primary_verdict(_row((-5, 0.0), True))['verdict'] == 'NOT SUPPORTED'   # upper must be < 0
    assert M.primary_verdict(_row((-5, -1), False))['verdict'] == 'NOT SUPPORTED'
    assert M.primary_verdict(_row((-5, -1), None))['verdict'] == 'NOT SUPPORTED'
    assert 'DUO_HOLD_60C' in M.primary_verdict(_row((-5, -1), True))['contrast']


def test_ident_row():
    sha = 'x'*64
    rec = dict(world_id='w', J_veh_s=1.5, safety=dict(collisions=0), world=dict(seed=1),
               executor_sha256_loaded=sha, a37=dict(activations=2, acts=[1], issued=3, activations_by_family={}),
               vehicle_log=dict(sha256='a'))
    rep = dict(ok=True, per_vehicle={'1': 2})
    r = M.ident_row('DUO_HOLD_60C', rec, dict(rec, vehicle_log=dict(sha256='b')), rep, dict(rep), sha)
    assert r['PASS'] and not r['vehicle_log_gz_sha_equal']
    assert not M.ident_row('DUO_HOLD_60C', rec, dict(rec, J_veh_s=1.6), rep, rep, sha)['PASS']
    assert not M.ident_row('DUO_HOLD_60C', rec, rec, rep, dict(rep, per_vehicle={'1': 3}), sha)['PASS']
    assert not M.ident_row('DUO_HOLD_60C', rec, dict(rec, a37=dict(rec['a37'], activations=3)), rep, rep, sha)['PASS']
    assert not M.ident_row('DUO_HOLD_60C', rec, rec, rep, rep, 'y'*64)['PASS']


def test_runner_patch():
    spec = importlib.util.spec_from_file_location('p81', HERE/'a81'/'host'/'patch_tables_a81.py')
    P = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P)
    src = ("A44_MECH_RANGE = (992400, 992499)\nA75D2_MECH_RANGE = (997400, 997499)\nX = (\n"
           "                       (A75D2_MECH_RANGE[0], A75D2_MECH_RANGE[1], A75D2_MECH_RANGE[0]),)\n")
    out = P.runner(src)
    assert 'A81_RANGES = ((990100, 990499),)' in out and 'for lo, hi in A81_RANGES)' in out
    assert P.runner(out) == out
    ns = {}
    exec(out, ns)
    assert ns['X'] == ((997400, 997499, 997400), (990100, 990499, 990100))
    with pytest.raises(SystemExit):
        P.runner(src + 'Y = (990000, 990199)\n')

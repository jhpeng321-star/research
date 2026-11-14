"""A98 CPU tests (zero SUMO)."""
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a81_bigconfirm as B  # noqa: E402
import a98_addon as M  # noqa: E402


def test_constants():
    assert M.ARMS == ('BANK_S4', 'BANK_S4S5') and M.SEEDS_C == tuple(range(988700, 988892))
    assert (M.MECH_EPISODES, M.MECH_CAP, M.CONFIRM_CAP, M.CAP_SIM_S) == (4, 4_800, 460_800, 465_600)
    assert M.PINS['BANK_S4'].startswith('a428cd1d') and M.PINS['BANK_S4S5'].startswith('40b16ed9')
    assert M.SETTINGS['c'] == 'a94c_D2800_p60' and M.RANGES['c'] == 'A94_RANGES'
    assert all(k == 'secondary' for _, _, k in M.CONTRASTS) and len(M.CONTRASTS) == 8
    got = {(a, r) for a, r, _ in M.CONTRASTS}
    for a in M.ARMS:
        assert {(a, 'BANK_HAND'), (a, 'BANK_SEARCHED'), (a, 'DUO_HOLD_60C'), (a, 'MAPPO60C')} <= got


def test_point():
    M.point()
    assert B.AMENDMENT == '98' and B.ARMS == M.ARMS and B.STAGE_CAP == dict(mech=4_800, confirm=460_800)
    assert B.PREFIX == dict(i='a98i_', c='a98c_') and B.MECH_FILE == 'A98_MECH.json'
    assert B.PANEL_C == 'A94_PANEL_C.json' and B.RESULTS_FILE == 'A98_RESULTS_D2800_p60.json'
    assert B.waves is M.waves and B.mech_compare is M.mech_compare and B.prepare is M.prepare


def test_bank_sources_pinned():
    for arm in M.ARMS:
        p = M.src_of(arm)
        if p.is_file():
            import hashlib
            assert hashlib.sha256(p.read_bytes()).hexdigest() == M.PINS[arm]


def test_env():
    assert os.environ.get('A94_HAND_SHA256') is not None

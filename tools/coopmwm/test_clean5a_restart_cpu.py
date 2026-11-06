"""clean5a zero-physics config pieces: 6-cell worlds, asserted guidance sources, registered keys."""
from collections import Counter
from pathlib import Path

import pytest

from tools.coopmwm import clean5a_restart as c5, bank_breadth as bb, map_descriptor as md

SPEC = Path(__file__).resolve().parents[2]/'assets/research/newmap_candidates/wz3to2'


def _doc(rows=None, sha='a'*64):
    rows = rows if rows is not None else [dict(arm='B', regime=c, family='wall_h_solo2', baseline='mappo', n=24,
                                               mean_veh_s=1., lb95_veh_s=.5, ub95_veh_s=2., harm_rate=0.)
                                          for c in c5.CELL_KEYS]
    return dict(schema='x', regime_rule='cell', provenance='t', weight=1., clip=1.,
                source=dict(sources=[dict(name='REGIME_RESULTS.jsonl', sha256=sha, baseline='mappo')], sha256='b'*64),
                rows=rows)


@pytest.fixture
def wz(monkeypatch):
    md._reset_for_tests()
    monkeypatch.setenv(md.ENV, str(SPEC/'MAP_SPEC.json'))
    yield md.current()
    md._reset_for_tests()


def test_worlds_cover_six_cells_and_registered_train_seeds(wz):
    worlds = c5.worlds_by_update()
    assert sorted(worlds, key=int) == [str(u) for u in range(1, 25)]
    cells = Counter()
    for u, rows in worlds.items():
        assert len(rows) == 16 and len({w['seed'] for w in rows}) == 16
        assert all(989220 <= w['seed'] <= 989267 and w['map'] == 'wz3to2' for w in rows)
        cells.update(w['cell_key'] for w in rows)
    assert set(cells) == set(c5.CELL_KEYS) and max(cells.values())-min(cells.values()) <= 16


def test_keys_and_asserted_guidance(wz):
    cfg = c5.keys(desc=wz, guidance_doc=_doc(), provenance='MAIN clean5a GO')
    assert cfg['imitation_epochs'] == 20 and bb.ev_only(cfg)
    assert cfg['g4_extension']['families'] == sorted(c5.G4_FAMILIES)
    from tools.coopmwm import track2_backend as tb, fullpol_ev, practice_offer
    assert tb.frozen_control_arm(cfg) and fullpol_ev.settings(cfg) is not None
    assert practice_offer.settings(cfg)['fraction'] == .5
    out = c5.check_guidance(cfg)
    assert out['rows'] == 6 and out['native_fallback_cells'] == [] and out['table_sha256'] == 'b'*64
    for bad in (_doc(rows=[]), _doc(sha='short'),
                _doc(rows=[dict(arm='R', regime=c, baseline='mappo') for c in c5.CELL_KEYS]),
                _doc(rows=[dict(arm='B', regime='D3200_p60', baseline='mappo')])):
        with pytest.raises(ValueError):
            c5.check_guidance(c5.keys(desc=wz, guidance_doc=bad, provenance='t'))
    with pytest.raises(ValueError):
        c5.check_guidance(dict(cfg, mined_moves='/x.jsonl'))


def test_ev_only_key_validation():
    assert not bb.ev_only({})
    with pytest.raises(ValueError):
        bb.ev_only(dict(guidance_mode='ev'))

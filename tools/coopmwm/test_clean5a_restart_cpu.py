"""clean5a zero-physics config pieces: 6-cell worlds, asserted guidance sources, registered keys."""
from collections import Counter
from pathlib import Path

import pytest

from tools.coopmwm import clean5a_restart as c5, bank_breadth as bb, map_descriptor as md
from tools.coopmwm.test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)

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


def test_exact_clean5a_keys_g2_pairs_offer_only_n2_options_for_b_and_r(wz, cfg, monkeypatch):
    """N11 F10 (MAIN 2026-11-06): with the EXACT clean5a keys (clean4_restart.levers + clean5a_restart.keys,
    G2 on), a G2 pair ego in B and in R gets only N2 options; the legacy G2 pair slate is never reached."""
    import numpy as np
    from . import clean4_restart as c4, coordination_opportunities as co, feasible_proposals as fpp
    from .clean4_receipts import CLEAN5A_ALLOWED
    from .test_lane_drop_cpu import make, row, tick
    exact = c4.levers(dict(cfg, trigger_candidates=8, trigger_partner_slots=4),
                      lane_drop_keys={}, v2v_keys={}, w2_keys={})
    exact.update(c5.keys(desc=wz, guidance_doc=_doc(), provenance='MAIN clean5a GO'))
    assert co.settings(exact) is not None and bb.ev_only(exact)
    calls = []
    monkeypatch.setattr(co, 'candidates', lambda state, rng, n: calls.append(state['self']) or [])
    drop = 75.44                                       # wz3to2 lane-drop s (MapView.lane_drop_s_m)
    n2_scene = [row('a', drop-1200, 0), row('b', drop-1190, 1), row('c', drop-1185, 2), row('d', drop-1250, 0),
                row('h1', drop-1150, 1, cav=False), row('h2', drop-1210, 2, cav=False)]
    g2_scene = [row('m', -60., 1), row('r', -55., 2), row('e', -90., 0), row('h', -40., 0, cav=False)]
    for arm in ('B', 'R'):
        p = make(exact, arm)
        p.observe(n2_scene, 300, native_commands={r['vid']: .3 for r in n2_scene if r['is_cav']})
        state = dict(p.local_states['a'], acquirable_cav_ids=['a', 'b'], coordination_partner='b')
        out = [o for o in fpp.candidates(state, np.random.default_rng(0), 8) if o is not None]
        fams = {o.family.split(':', 1)[0] for o in out}
        assert out and fams <= set(CLEAN5A_ALLOWED), (arm, fams)
        pairs, offered = 0, set()
        for t in range(301, 311):                  # real G2 graph on the policy path (periodic decisions)
            p.last_decision = None
            p.executor.active = []
            rows = [dict(r_, s_m=r_['s_m']+.5*(t-301)) for r_ in g2_scene+n2_scene]
            trig = tick(p, rows, t, deterministic=False, seed=t)['track2_trigger']
            pairs += trig['proposals']['coordination_opportunities']['disjoint_pairs']
            offered |= {f.split(':', 1)[0] for f, n in trig['proposals']['proposed_by_family'].items() if n}
        assert pairs > 0 and offered and offered <= set(CLEAN5A_ALLOWED), (arm, pairs, offered)
    assert calls == []

"""WM-ACT sensitivity arm (post-registration, pre-outcome): ranking rule tests. Every score below is a synthetic
fixture; the one test that opens real files reads only the WM score files (no SUMO outcome) and writes to tmp."""
import json
from pathlib import Path
import shutil
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import cvlib as L  # noqa: E402
import e1_order as E  # noqa: E402
import e1_order_act as A  # noqa: E402
import wm_act_score as G  # noqa: E402


def _rows(spec):
    """spec = [(hash, screen_score, screen_activated, registered_finalist, final_activated, final_score)]"""
    return [dict(desc_hash=h, key='k'+h, screen_summary=dict(score=ss), screen_activated=sa, finalist=fi,
                 activated=fa if fi else None, summary=dict(score=fs) if fi else None)
            for h, ss, sa, fi, fa, fs in spec]


def test_finalists_never_activated_last_and_fill():
    spec = [('a', 0.0, False, True, False, 0.0), ('b', -5.0, True, False, None, None),
            ('c', 0.0, False, True, True, 3.0), ('d', -1.0, True, False, None, None),
            ('e', -1.0, True, False, None, None), ('f', 2.0, True, True, True, 1.0),
            ('g', 0.0, False, False, None, None)]
    rows = _rows(spec)
    # activated by screen score (ties d/e in pool order), then the never-activated ones in pool order (all 0.0)
    assert A.act_finalists(rows, 7) == ['f', 'd', 'e', 'b', 'a', 'c', 'g']
    assert A.act_finalists(rows, 3) == ['f', 'd', 'e']                     # 0.0 of a never-activated plan never wins
    assert A.act_finalists(rows, 5) == ['f', 'd', 'e', 'b', 'a']           # fill only when fewer activated than n
    for n in (1, 3, 5, 7):
        assert A.act_finalists(rows, n) == G.act_finalists(rows, n)        # GPU script and ingest agree


def test_plan_reuse_and_probe():
    spec = [('a', 0.0, False, True, True, 4.0), ('b', -5.0, True, False, None, None),
            ('c', -1.0, True, True, False, 0.0), ('d', -2.0, True, True, True, -3.0),
            ('e', 0.0, False, True, False, 0.0)]
    base = dict(rows=_rows(spec), finalist_hashes=['a', 'e', 'c', 'd'])
    fin, new, reused, probe = G.plan(base, 3)
    assert fin == ['c', 'd', 'b'] and new == ['b'] and reused == ['c', 'd']
    assert probe == 'd'                                                    # first reused finalist the WM activated
    base2 = dict(rows=_rows([('a', 0.0, False, True, True, 4.0), ('b', -5.0, True, False, None, None)]),
                 finalist_hashes=['a'])
    assert G.plan(base2, 1) == (['b'], ['b'], [], 'a')                     # else a registered activated finalist
    assert G.plan(base, 3, 'all') == (['c', 'd', 'b'], ['c', 'd', 'b'], [], None)
    none_new = dict(rows=_rows([('a', 1.0, True, True, True, 4.0), ('b', -5.0, False, False, None, None)]),
                    finalist_hashes=['a'])
    assert G.plan(none_new, 1) == (['a'], [], ['a'], None)                 # nothing to score -> no probe, no GPU


def test_act_rank_order(monkeypatch):
    monkeypatch.setattr(L, 'TOP_K', 2)
    spec = [('a', 0.0, False, False, None, None), ('b', -5.0, True, False, None, None),
            ('c', -1.0, True, False, None, None), ('d', -2.0, True, False, None, None),
            ('e', 0.0, False, False, None, None), ('f', -9.0, True, False, None, None),
            ('g', -3.0, True, False, None, None)]
    rows = _rows(spec)
    pool = [r['desc_hash'] for r in rows]
    assert A.act_finalists(rows, 4) == ['c', 'd', 'g', 'b']
    final = [dict(desc_hash='c', activated=False, summary=dict(score=0.0)),      # never activated on a final state
             dict(desc_hash='d', activated=True, summary=dict(score=-7.0)),
             dict(desc_hash='g', activated=True, summary=dict(score=-7.0)),      # tie with d -> screen rank
             dict(desc_hash='b', activated=True, summary=dict(score=-1.0))]
    got = A.act_rank(pool, rows, final, 4)
    # activated finalists by final score (a negative score still beats a never-activated 0.0), then c, then the
    # non-finalists: screen-activated f, then never-activated a, e in pool order
    assert got == ['b', 'd', 'g', 'c', 'f', 'a', 'e']
    assert sorted(got) == sorted(pool)
    with pytest.raises(L.Refused):
        A.act_rank(pool, rows, final[:3], 4)
    with pytest.raises(L.Refused):
        A.act_rank(pool[::-1], rows, final, 4)


def test_ingest_on_registered_scores(tmp_path, capsys):
    """Real registered score files + a synthetic WM_ACT_SCORES (new finalists get fixture scores): the order file
    is written under tmp, the registered orders are untouched, top six = activated finalists only when >= 6."""
    scores = HERE/'wm_scores'
    if not (scores/'WM_SCORES_seed67.json').is_file() or not E.pool_file(HERE/'out', 67).is_file():
        pytest.skip('registered WM scores not present')
    out = tmp_path/'out'
    (out/'e1'/'pool').mkdir(parents=True)
    shutil.copy(E.pool_file(HERE/'out', 67), E.pool_file(out, 67))
    base = L.read_json(scores/'WM_SCORES_seed67.json')
    by = {r['desc_hash']: r for r in base['rows']}
    fin, new, reused, probe = G.plan(base, base['finalists'])
    assert len(fin) == 12 and all(by[h]['screen_activated'] for h in fin) and set(reused) <= set(base['finalist_hashes'])
    final_rows = [dict(desc_hash=h, key=by[h]['key'], source='new' if h in new else 'reused',
                       activated=True if h in new else bool(by[h]['activated']),
                       summary=dict(score=-1000.0-i) if h in new else by[h]['summary']) for i, h in enumerate(fin)]
    act = tmp_path/'act'
    act.mkdir()
    (act/'WM_ACT_SCORES_seed67.json').write_text(json.dumps(dict(
        mode='reuse', search_seed=67, pool_doc_sha256=base['pool_doc_sha256'],
        base_scores_sha256=L.sha_file(scores/'WM_SCORES_seed67.json'), cfg_sha256=base['cfg_sha256'],
        wm_checkpoint_sha256=base['wm_checkpoint_sha256'], n_final_states=base['n_final_states'], K=base['K'],
        H=base['H'], act_finalist_hashes=fin, new=new, reused=reused, probe=None, final_rows=final_rows,
        wm_rollouts=0)), encoding='utf-8')
    args = ['--out', str(out), '--seeds', '67', '--scores-dir', str(scores), '--act-scores-dir', str(act)]
    assert A.main(['ingest']+args) == 0
    doc = L.verify_seal(L.read_json(E.order_path(out, 'WMACT', 67)))
    assert doc['arm'] == 'WMACT' and doc['label'] == A.LABEL and doc['n'] == 120 and len(set(doc['order'])) == 120
    assert set(doc['top']) <= set(fin) and doc['unscored'] == 0
    assert all(doc['final_activated'][h] for h in doc['top'])
    assert not E.order_path(out, 'WM', 67).exists()
    # a tampered reused row is refused
    final_rows[fin.index(reused[0])]['summary'] = dict(score=1e9)
    bad = json.loads((act/'WM_ACT_SCORES_seed67.json').read_text())
    bad['final_rows'] = final_rows
    (act/'WM_ACT_SCORES_seed67.json').write_text(json.dumps(bad), encoding='utf-8')
    with pytest.raises(L.Refused):
        A.act_doc(out, 67, scores, act)

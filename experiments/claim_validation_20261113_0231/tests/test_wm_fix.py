"""E1-FIX (Amendment 74, POST HOC REPAIR): pure logic of the repaired WM scoring - Fix-1 ordering, draw seeding and
its independence from chunking, gate G0, ingest. No WM, no SUMO; every score below is a synthetic fixture and
everything is written under tmp."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import cvlib as L  # noqa: E402
import e1_order as E  # noqa: E402
import e1_order_fix as OF  # noqa: E402
import e1_pool  # noqa: E402
import wm_fix_crn as C  # noqa: E402
import wm_fix_score as F  # noqa: E402

CFG = dict(search_grid=dict(v_w=[18, 19, 20, 21, 22], dmax=[1.0, 1.5], hold_s=[90, 120, 180], onset_s=[0, 20, 40]),
           onset_max=30)


def _row(h, score, act=True):
    return dict(desc_hash=h, activated=act, summary=dict(score=score, gain_vs_base=-score))


# ------------------------------------------------------------------------------------------------- Fix 1 (screen)
def test_fix_finalists_never_activated_last_and_stable():
    rows = [_row('a', 0.0, False), _row('b', 5.0), _row('c', 9.0, False), _row('d', 5.0), _row('e', -3.0),
            _row('f', 0.0, False)]
    # activated by score descending, ties in pool order; never-activated after ALL activated ones (even at a higher
    # score), by the same stable sort
    assert F.fix_finalists(rows, 6) == ['b', 'd', 'e', 'c', 'a', 'f']
    assert F.fix_finalists(rows, 2) == ['b', 'd']
    assert F.fix_finalists(rows, 4) == ['b', 'd', 'e', 'c']
    assert F.fix_finalists(rows, 3, 'gain_vs_base') == ['e', 'b', 'd']
    # the registered sort (score only) would have put the never-activated zero-score plans above a negative one
    assert sorted(rows, key=lambda r: r['summary']['score'], reverse=True)[0]['desc_hash'] == 'c'


# -------------------------------------------------------------------------------------------- Fix 1 (final order)
def _screen(hashes, scores, act):
    return [dict(desc_hash=h, screen_activated=a, screen_summary=dict(score=s)) for h, s, a in zip(hashes, scores, act)]


def test_fix_rank_orders_final_then_rest(monkeypatch):
    monkeypatch.setattr(L, 'TOP_K', 2)
    hs = list('abcdefgh')
    scr = _screen(hs, [1, 8, 7, 6, 0, 5, 0, 9], [True, True, True, True, False, True, False, False])
    fin = F.fix_finalists([dict(desc_hash=r['desc_hash'], activated=r['screen_activated'],
                                summary=r['screen_summary']) for r in scr], 4)
    assert fin == ['b', 'c', 'd', 'f']
    final = [_row('b', 0.0, False), _row('c', -4.0), _row('d', 2.0), _row('f', 2.0)]
    order = OF.fix_rank(hs, scr, fin, final)
    # finalists activated at the final stage by final score (tie d/f by screen rank), then the finalist the WM never
    # activated on a final state, then non-finalists: screen-activated first, then never-activated by score/pool order
    assert order == ['d', 'f', 'c', 'b', 'a', 'h', 'e', 'g']
    assert sorted(order) == hs
    with pytest.raises(L.Refused):                                    # finalists are not the Fix-1 finalists
        OF.fix_rank(hs, scr, ['h', 'b', 'c', 'd'], final)
    with pytest.raises(L.Refused):                                    # rows not in pool order
        OF.fix_rank(hs[::-1], scr, fin, final)
    with pytest.raises(L.Refused):                                    # a finalist without a final row
        OF.fix_rank(hs, scr, fin, final[:3])


# ------------------------------------------------------------------------------------------------- Fix 2 (seeds)
def test_draw_seeds_depend_on_seed_state_draw_only():
    a = C.arrival_seed(11, 'd1', 0)
    assert a == C.arrival_seed(11, 'd1', 0) and isinstance(a, int) and 0 <= a < 16**15
    others = {C.arrival_seed(12, 'd1', 0), C.arrival_seed(11, 'd2', 0), C.arrival_seed(11, 'd1', 1),
              C.pivot_seed(11, 'd1', 0), C.torch_seed(11, 'd1', 0)}
    assert a not in others and len(others) == 5
    assert C.arrival_seed(1, 'x', 11) != C.arrival_seed(11, 'x', 1)   # no field mixing
    assert C.pivot_seed(11, 'd1', 3) == C.pivot_seed(11, 'd1', 3) != C.pivot_seed(11, 'd1', 4)
    assert 0 <= C.torch_seed(11, 'd1', 0) < 2**63-1


@pytest.mark.parametrize('n_actions,K', [(14, 8), (32, 4), (1, 8), (5, 2), (33, 3)])
def test_row_draws_independent_of_chunking(n_actions, K):
    """planapi rows are draw-major; whatever max_rows cuts them into, row i gets draw i // n_actions, so the seed of
    a row is a function of (seed, state, draw) alone and every action (base included) of a draw shares it."""
    total = n_actions*K
    want = [k for k in range(K) for _ in range(n_actions)]
    for max_rows in (1, 3, 5, 32, 64, 1000):
        got, off = [], 0
        while off < total:
            b = min(max_rows, total-off)
            got += C.row_draws(off, b, n_actions)
            off += b
        assert got == want
    seeds = [C.arrival_seed(7, 'dg', k) for k in want]
    for a in range(n_actions):                                        # same draw -> same stream for every action
        assert [seeds[k*n_actions+a] for k in range(K)] == [C.arrival_seed(7, 'dg', k) for k in range(K)]


def test_lane_split():
    assert C.lane_split([]) == 0.5
    assert C.lane_split([dict(lane=0), dict(lane=1), dict(lane=1), dict(lane=2)]) == pytest.approx(2/3)
    assert C.lane_split([dict(lane=0), dict()]) == 0.0


def test_pivot_keys_per_draw_whatever_the_chunk():
    torch = pytest.importorskip('torch')

    def keys(ks):
        ctx = dict(ks=ks)
        p = C._TorchProxy(torch, lambda k: C.pivot_seed(5, 'dg', k), ctx)
        out = p.randint(0, 1 << 52, (len(ks), 7), device='cpu', dtype=torch.long)
        assert p.calls == 1 and out.shape == (len(ks), 7) and out.dtype == torch.long
        return out
    full = keys([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert torch.equal(full[0], full[1]) and torch.equal(full[3], full[5]) and not torch.equal(full[0], full[3])
    # other chunkings / another action subset: the key row of draw k is the same row
    assert torch.equal(keys([0, 0])[0], full[0]) and torch.equal(keys([1, 2, 2])[0], full[3])
    assert torch.equal(keys([2])[0], full[8])
    assert p_other(torch) is True
    with pytest.raises(RuntimeError):                                 # row count does not match the chunk
        C._TorchProxy(torch, lambda k: 1, dict(ks=[0, 1])).randint(0, 9, (3, 7))
    with pytest.raises(RuntimeError):
        C._TorchProxy(torch, lambda k: 1, dict(ks=None)).randint(0, 9, (3, 7))
    assert C._TorchProxy(torch, lambda k: 1, dict(ks=None)).zeros is torch.zeros      # everything else passes through


def p_other(torch):
    a = C._TorchProxy(torch, lambda k: C.pivot_seed(5, 'dg', k), dict(ks=[0])).randint(0, 1 << 52, (1, 7))
    b = C._TorchProxy(torch, lambda k: C.pivot_seed(6, 'dg', k), dict(ks=[0])).randint(0, 1 << 52, (1, 7))
    return not torch.equal(a, b)


# ---------------------------------------------------------------------------------------------------- G0 + ingest
def _docs(pool, seed, bump=None, repeat_rows=32):
    """Synthetic main + repeat score docs of one seed: all 12 first pool plans screen-activated except index 3;
    final score = index (so the top is 11, 10, ...), finalist index 5 never activated at the final stage."""
    es = pool['pool']
    srows = [dict(desc_hash=e['desc_hash'], activated=e['index'] != 3, summary=dict(score=-float(e['index'])))
             for e in es]
    fin = F.fix_finalists(srows, 12)
    idx = {e['desc_hash']: e['index'] for e in es}

    def final(h, d=0.0):
        act = idx[h] != 5
        return dict(desc_hash=h, key=str(idx[h]), activated=act, summary=dict(score=(float(idx[h]) if act else 0.0)+d))
    f = {h: final(h) for h in fin}
    common = dict(pool_doc_sha256=pool['doc_sha256'], search_seed=seed, cfg_sha256='x', states_sha256='s',
                  wm_checkpoint_sha256=L.WM_CKPT_SHA256, n_final_states=18, H=300, K_screen=4, finalists=12,
                  K_final_registered=4, K=8, crn_module_sha256='c', scorer_sha256='f', score_field='score',
                  finalist_hashes=fin, wm_rollouts=10, crn_log=dict(residual_global_rng_draws={}))
    main = dict(common, mode='main', max_rows=64, n_screen_states=4, screen_worlds=[1, 2, 3, 4],
                rows=[dict(desc_hash=r['desc_hash'], key=str(idx[r['desc_hash']]), n_roles=1,
                           screen_summary=r['summary'], screen_activated=r['activated'],
                           finalist=r['desc_hash'] in f,
                           activated=f[r['desc_hash']]['activated'] if r['desc_hash'] in f else None,
                           summary=f[r['desc_hash']]['summary'] if r['desc_hash'] in f else None) for r in srows])
    rep = dict(common, mode='repeat', max_rows=repeat_rows, main_max_rows=64,
               final_rows=[final(h, (bump or {}).get(idx[h], 0.0)) for h in fin])
    return main, rep


def _six(tmp_path, bump_seed=None, bump=None):
    cfg = tmp_path/'cfg.json'
    cfg.write_text(json.dumps(dict(config=CFG)), encoding='utf-8')
    out = tmp_path/'out'
    seeds = list(L.SEARCH_SEEDS)
    assert e1_pool.main(['--cfg', str(cfg), '--out', str(out), '--seeds']+[str(s) for s in seeds]) == 0
    fd, rd, sd = tmp_path/'fix', tmp_path/'rep', tmp_path/'scores'
    for d in (fd, rd, sd):
        d.mkdir()
    for s in seeds:
        pool = e1_pool.load_pool(out, s)
        m, r = _docs(pool, s, bump if s == bump_seed else None)
        mp = fd/f'WM_FIX_SCORES_seed{s}.json'
        mp.write_text(json.dumps(m), encoding='utf-8')
        r['main_scores_sha256'] = L.sha_file(mp)
        (rd/f'WM_FIX_SCORES_seed{s}.json').write_text(json.dumps(r), encoding='utf-8')
        (sd/f'WM_SCORES_seed{s}.json').write_text(json.dumps(dict(
            pool_doc_sha256=pool['doc_sha256'], cfg_sha256='x', states_sha256='s', n_final_states=18, H=300,
            K_screen=4, K=4, finalists=12, screen_worlds=[1, 2, 3, 4],
            finalist_hashes=[e['desc_hash'] for e in pool['pool'][:12]])), encoding='utf-8')
    args = ['--out', str(out), '--fix-dir', str(fd), '--repeat-dir', str(rd), '--scores-dir', str(sd)]
    return out, args, fd, rd, sd


def test_g0_seed_and_gate():
    es = [dict(desc_hash=f'h{i:02d}', index=i) for i in range(20)]
    pool = dict(pool=es, doc_sha256='p')
    hs = [e['desc_hash'] for e in es]
    m, r = _docs(pool, 11)
    g = OF.g0_seed(hs, m, r)
    assert g['same_first_six'] and g['same_full_order'] and g['max_abs_score_diff'] == 0.0
    assert g['activation_flags_equal'] and g['max_rows'] == [64, 32]
    # Fix 1 at both stages: index 3 (never screen-activated) is a finalist only as fill-up... and index 5 (never
    # activated at the final stage) ranks after every activated finalist
    order = OF.fix_rank(hs, m['rows'], m['finalist_hashes'], r['final_rows'])
    assert order[:6] == ['h12', 'h11', 'h10', 'h09', 'h08', 'h07'] and order[11] == 'h05' and order[-1] == 'h03'
    # a float-level difference that changes nothing in the order passes and is reported
    m2, r2 = _docs(pool, 11, bump={10: 1e-6, 2: -1e-6})
    g2 = OF.g0_seed(hs, m2, r2)
    assert g2['same_first_six'] and g2['max_abs_score_diff'] == pytest.approx(1e-6)
    # a swap inside the first six fails; a swap below the first six does not fail the gate
    g3 = OF.g0_seed(hs, *_docs(pool, 11, bump={10: 1.5}))
    assert not g3['same_first_six'] and g3['first_six_main'] != g3['first_six_repeat']
    g4 = OF.g0_seed(hs, *_docs(pool, 11, bump={1: 1.5}))
    assert g4['same_first_six'] and not g4['same_full_order']
    # same set, another order = FAIL (the gate is on the order)
    assert sorted(g3['first_six_main']) == sorted(g3['first_six_repeat'])
    with pytest.raises(L.Refused):                                    # the repeat used the same chunk size
        OF.g0_seed(hs, *_docs(pool, 11, repeat_rows=64))
    rows = [dict(g, search_seed=s) for s in L.SEARCH_SEEDS]
    assert OF.g0(rows)['verdict'] == 'PASS' and OF.g0(rows)['stop'] is None
    assert OF.g0(rows[:5])['verdict'] == 'FAIL'                       # a seed is missing
    bad = OF.g0(rows[:5]+[dict(g3, search_seed=103)])
    assert bad['verdict'] == 'FAIL' and bad['stop'] == OF.G0_FAIL and bad['max_abs_score_diff'] == 1.5
    assert OF.g0([])['verdict'] == 'FAIL'


def test_ingest_writes_orders_only_after_g0(tmp_path, capsys):
    out, args, fd, rd, sd = _six(tmp_path)
    assert OF.main(['g0']+args) == 0
    assert 'G0 PASS max |score diff| 0.0' in capsys.readouterr().out
    assert OF.main(['ingest', '--dry-run']+args) == 0
    assert not E.order_path(out, OF.ARM, 11).exists()
    assert OF.main(['ingest']+args) == 0
    for s in L.SEARCH_SEEDS:
        od = L.verify_seal(L.read_json(E.order_path(out, 'WMFIX', s)))
        pool = e1_pool.load_pool(out, s)
        h = [e['desc_hash'] for e in pool['pool']]
        assert od['arm'] == 'WMFIX' and od['label'] == F.LABEL and od['g0']['verdict'] == 'PASS'
        assert od['top'] == [h[i] for i in (12, 11, 10, 9, 8, 7)] and sorted(od['order']) == sorted(h)
        assert od['K'] == 8 and od['K_final_registered'] == 4 and od['max_rows'] == [64, 32] and od['unscored'] == 0
        assert od['pool_doc_sha256'] == pool['doc_sha256']
    assert OF.main(['ingest']+args) == 0                              # write-once: the same content is kept
    assert OF.main(['show', '--out', str(out)]) == 0
    # the main file changed after the repeat was made -> not a main run and its repeat
    p = fd/'WM_FIX_SCORES_seed23.json'
    p.write_text(p.read_text(encoding='utf-8')+'\n', encoding='utf-8')
    with pytest.raises(L.Refused):
        OF.main(['g0']+args)
    capsys.readouterr()


def test_g0_fail_stops_ingest(tmp_path, capsys):
    out, args, fd, rd, sd = _six(tmp_path, bump_seed=67, bump={10: 1.5})
    assert OF.main(['g0']+args) == 4
    assert 'G0 FAIL' in capsys.readouterr().out
    with pytest.raises(L.Refused):                                    # even for a seed that agrees
        OF.main(['ingest', '--seeds', '11']+args)
    assert not (out/'e1'/'order').exists() or not list((out/'e1'/'order').glob('ORDER_WMFIX_*'))
    with pytest.raises(SystemExit):                                   # the gate is defined on all six seeds
        OF.main(['g0', '--seeds', '11']+args)
    capsys.readouterr()


def test_protocol_identity_refusals(tmp_path, capsys):
    out, args, fd, rd, sd = _six(tmp_path)
    for name, field, value in (('WM_FIX_SCORES_seed42.json', 'K', 4), ('WM_FIX_SCORES_seed42.json', 'H', 120),
                               ('WM_FIX_SCORES_seed42.json', 'cfg_sha256', 'y')):
        p = rd/name
        keep = p.read_text(encoding='utf-8')
        p.write_text(json.dumps(dict(json.loads(keep), **{field: value})), encoding='utf-8')
        with pytest.raises(L.Refused):
            OF.main(['g0']+args)
        p.write_text(keep, encoding='utf-8')
    assert OF.main(['g0']+args) == 0
    capsys.readouterr()

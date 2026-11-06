"""CPU tests of WMEV (tools.wzprop.regime_u0 / regime_u0_compare): rule port, CRN, entries, SUMO windows, analysis.

The WM-backed tests use the UNTRAINED-on-wz3to2 WM2 b8797242 (merge-map WM, road-bound to wz3to2) on tiny fixtures;
they are skipped when its checkpoint (env WMEV_WM2, default _wmev/wm2/wm_s.pt) or the u0 policy (env WMEV_U0, default
_wmev/u0/policy_00169.pt) is absent.  Run:
    PYTHONPATH="src;." COOPMWM_MAP_SPEC=assets/research/newmap_candidates/wz3to2/MAP_SPEC.json \
        python -m pytest -q tools/wzprop/test_regime_u0_cpu.py
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pytest

from tools.wzprop import geometry as G
from tools.wzprop import regime_u0 as R
from tools.wzprop import regime_u0_compare as C

ROOT = Path(__file__).resolve().parents[2]
MAP_SPEC = ROOT / 'assets/research/newmap_candidates/wz3to2/MAP_SPEC.json'
WM2 = Path(os.environ.get('WMEV_WM2', ROOT / '_wmev/wm2/wm_s.pt'))
U0 = Path(os.environ.get('WMEV_U0', ROOT / '_wmev/u0/policy_00169.pt'))
SUMO_WIN = ROOT / 'wmev_data/SUMO_U0_WINDOWS.json'
TAPES = Path('E:/research-dense/runs/active/C4-V7FT-20261017/wmqual/tapes')
needs_wm = pytest.mark.skipif(not (WM2.exists() and U0.exists()), reason='WM2 / u0 checkpoint not present')


# ------------------------------------------------------------------------------------------ rule port
def test_pick_wall_vectorised_matches_verbatim_rule():
    rng = random.Random(7)
    for trial in range(300):
        n = rng.randint(0, 14)
        cavs = [dict(s_m=rng.uniform(-1900, 100), lane=rng.choice((0, 1, 2)), v=10.0) for _ in range(n)]
        ref = R.pick_wall(cavs)
        got = R.pick_wall_slots([c['s_m'] for c in cavs], [c['lane'] for c in cavs], list(range(n)))
        if ref is None:
            assert got is None, trial
        else:
            assert got == [cavs.index(c) for c in ref], trial          # SUMO lane 0, 1, 2 order


def test_arms_and_rule_parameters_are_the_registered_ones():
    assert R.WALL_H == dict(v_w=15.0, hold_s=90, dmax=1.5, amax=1.0)
    assert R.WALL_G == dict(v_w=22.0, hold_s=120, dmax=1.0, amax=1.0)
    assert R.ZONE_S == pytest.approx((-1724.56, -324.56)) and R.RELEASE_S == pytest.approx(75.44)
    assert R.arm_rule('wall_h_solo2') == (R.WALL_H, 2) and R.arm_rule('wall_g') == (R.WALL_G, None)
    assert R.arm_rule('u0') is None and R.arm_rule('native') is None
    with pytest.raises(ValueError):
        R.arm_rule('wall_x')
    assert R.arms_for('D3200_p60', ['auto']) == ['u0', 'wall_h_solo2', 'wall_h', 'wall_h_solo0']
    assert R.arms_for('D3200_p20', ['auto', 'native']) == ['u0', 'wall_h_solo2', 'wall_h', 'wall_g', 'native']
    assert R.arms_for('D2400_p40', ['auto']) == ['u0', 'wall_h_solo2', 'wall_h']


def _rule_state(s, lane, present=None, cav=None):
    torch = pytest.importorskip('torch')
    n = len(s)
    st = dict(s_m=torch.tensor([s], dtype=torch.float32).repeat(3, 1),
              lane=torch.tensor([lane]).repeat(3, 1),
              present=torch.tensor([present or [True] * n]).repeat(3, 1),
              is_cav=torch.tensor([cav or [True] * n]).repeat(3, 1))
    return st


def test_rule_update_trigger_cooldown_release_semantics():
    """Rows u0 / wall_h / wall_h_solo2: wall picked at t=0, solo keeps only the SUMO-lane-2 (WM lane 0) member,
    cooldown = hold (90 s), release at hold end / at the drop / when gone; the base row never acts."""
    torch = pytest.importorskip('torch')
    ctl = R.U0RuleController.__new__(R.U0RuleController)
    B, n = 3, 6
    ctl.rules = [R.arm_rule(a) for a in ('u0', 'wall_h', 'wall_h_solo2')]
    ctl.member = torch.zeros(B, n, dtype=torch.bool)
    ctl.start = torch.zeros(B, n)
    ctl.v_w, ctl.dmax, ctl.amax, ctl.hold = (torch.zeros(B, n) for _ in range(4))
    ctl.next_ok = [-math.inf] * B
    ctl.acts = [[] for _ in range(B)]
    ctl.updated_step = -1
    s = [-1000.0, -1050.0, -980.0, -500.0, 10.0, -2000.0]
    lane = [0, 1, 2, 1, 0, 2]                         # slots 0/1/2 form a wall (spread 70 m)
    st = _rule_state(s, lane)
    ctl._update(st, 0)
    assert ctl.member[0].sum() == 0 and not ctl.acts[0]
    assert ctl.member[1].tolist() == [True, True, True, False, False, False]
    assert ctl.member[2].tolist() == [True, False, False, False, False, False]      # WM lane 0 = SUMO lane 2
    assert ctl.next_ok[1] == 90 and ctl.next_ok[2] == 90
    assert float(ctl.v_w[1, 0]) == 15.0 and float(ctl.dmax[1, 0]) == 1.5
    ctl._update(st, 1)                                 # odd step: no rule decision
    st2 = _rule_state([-900.0, 80.0, -900.0, -500, 10, -2000], lane, present=[True, True, False, True, True, True])
    ctl._update(st2, 20)                               # t = 10 s: slot 1 at the drop, slot 2 gone
    assert ctl.member[1].tolist() == [True, False, False, False, False, False]
    ctl._update(st, 178)                               # t = 89 s: still held, cooldown not passed
    assert bool(ctl.member[1, 0]) and len(ctl.acts[1]) == 1
    ctl._update(st, 180)                               # t = 90 s: hold end, cooldown passed -> new wall at once
    assert len(ctl.acts[1]) == 2 and ctl.acts[1][1][0] == 90.0
    assert float(ctl.start[1, 0]) == 90.0


# ------------------------------------------------------------------------------------------ entries
def test_grouped_entries_share_arrivals_within_a_draw_and_keep_cav_identity():
    torch = pytest.importorskip('torch')
    from wms.core import road as RD
    with RD.road_scope(RD.load_road(str(MAP_SPEC))):
        counter = np.zeros((2, 30))
        counter[:, :12] = 3.0                          # 0.3 veh/s per entrance
        n = 40
        st = dict(s_m=torch.zeros(4, n), v_mps=torch.zeros(4, n), a_mps2=torch.zeros(4, n),
                  lane=torch.zeros(4, n, dtype=torch.long), length_m=torch.full((4, n), 5.0),
                  present=torch.zeros(4, n, dtype=torch.bool), is_cav=torch.zeros(4, n, dtype=torch.bool))
        st['present'][:, 0] = True
        st['s_m'][:, 0] = 500.0
        st['v_mps'][:, 0] = 20.0
        ep = R.GroupedEntryProcess([0, 0, 1, 1], {g: dict(counter=counter, cav_share=0.5, seed=11 + g)
                                                  for g in (0, 1)})
        masks = []
        for _ in range(60):
            masks.append(ep.insert(st).clone())
            st['s_m'] = torch.where(st['present'], st['s_m'] + 10.0, st['s_m'])
        m = torch.stack(masks)
        assert torch.equal(m[:, 0], m[:, 1]) and torch.equal(m[:, 2], m[:, 3])
        assert torch.equal(st['is_cav'][0], st['is_cav'][1])
        assert not torch.equal(m[:, 0], m[:, 2])                                        # other draw, other arrivals
        assert len(ep.backlog_log) == 60 and ep.stats[0].inserted > 0 and ep.stats[0].inserted_cav > 0


# ------------------------------------------------------------------------------------------ WM (WM2, tiny)
def _fixture(wall=True):
    """Synthetic observable state on wz3to2 (global x): 3 CAVs abreast in the zone (wall) + HDVs + counter."""
    cavs = [dict(id='c0', edge='up', lane=0, x=1500.0, v=20.0), dict(id='c1', edge='up', lane=1, x=1520.0, v=22.0),
            dict(id='c2', edge='up', lane=2, x=1480.0, v=24.0), dict(id='c3', edge='up', lane=1, x=2400.0, v=18.0)]
    if not wall:
        cavs[0]['x'] = 300.0                           # lane-0 CAV outside the zone: no wall
    det = [dict(id=f'h{i}', edge='up', lane=i % 3, x=1300.0 + 60.0 * i, v=19.0, is_cav=False) for i in range(8)]
    counter = [[3.0] * 12 + [0.0] * 18, [2.0] * 12 + [0.0] * 18]
    return dict(sid='fixture_D3200_p60_s989200_t300', cell='D3200_p60', seed=989200, t0=300, valid=True,
                all_cavs_t0=cavs, observed=dict(vehicles=det, counter=counter, cav_entry_share=0.5))


@pytest.fixture(scope='module')
def wm():
    if not (WM2.exists() and U0.exists()):
        pytest.skip('WM2 / u0 checkpoint not present')
    import torch
    torch.set_num_threads(2)
    os.environ.setdefault('COOPMWM_MAP_SPEC', str(MAP_SPEC))
    from wms import planapi as P
    return P.load(str(WM2), device='cpu', max_rows=8), R.load_policy(str(U0))


@needs_wm
def test_wm2_rollout_crn_divergence_after_activation_and_determinism(wm):
    h, exe = wm
    m = _fixture(wall=True)
    arms = ['u0', 'wall_h_solo2', 'wall_h']
    tr = {}
    a = R.evaluate_state(h, exe, m, arms=arms, K=2, H=6.0, slots=48, max_rows=8, trace=tr)
    b = R.evaluate_state(h, exe, m, arms=arms, K=2, H=6.0, slots=48, max_rows=8)
    for arm in arms:
        assert a['arms'][arm]['road_w'] == b['arms'][arm]['road_w']                    # deterministic
        for k in range(2):
            assert len(a['arms'][arm]['road_w'][k]) == 1 and math.isfinite(a['arms'][arm]['road_w'][k][0])
    for arm in ('wall_h_solo2', 'wall_h'):
        for k in range(2):
            fa, fd = a['arms'][arm]['first_act_s'][k], a['arms'][arm]['first_div_s'][k]
            assert fa == 0.0 and a['arms'][arm]['activations'][k] == 1
            assert fd is None or fd >= fa
    assert a['arms']['wall_h']['road_w'] != a['arms']['u0']['road_w']                  # the rule acts
    acts = dict(zip(tr['rows'], tr['acts']))
    assert len(acts[('wall_h', 0)][0][1]) == 3 and len(acts[('wall_h_solo2', 0)][0][1]) == 1


@needs_wm
def test_wm2_multi_state_call_matches_single_state_calls(wm):
    """Packing several start states into one call (fixed slot count) reproduces the per-state results."""
    h, exe = wm
    m1, m2 = _fixture(wall=True), dict(_fixture(wall=False), sid='fixture2_D2400_p40_s989201_t300', cell='D2400_p40',
                                       seed=989201)
    kw = dict(arms=['auto'], K=2, H=4.0, slots=64, max_rows=64)
    both = R.evaluate_states(h, exe, [m1, m2], **kw)
    one = [R.evaluate_states(h, exe, [m], **kw)[0] for m in (m1, m2)]
    assert both[0]['batch'] == 2 and one[0]['batch'] == 1
    for x, y in zip(both, one):
        assert set(x['arms']) == set(y['arms'])
        for arm in x['arms']:
            np.testing.assert_allclose(np.array(x['arms'][arm]['road_w']), np.array(y['arms'][arm]['road_w']),
                                       rtol=1e-4)
            assert x['arms'][arm]['activations'] == y['arms'][arm]['activations']


@needs_wm
def test_wm2_exact_crn_on_a_real_state_when_the_rule_never_fires(wm, monkeypatch):
    """One arm per call: a strategy whose trigger never fires reproduces the u0 rows BIT-identically on a real
    D2400 start state over 40 s with entries (identical rows inside one batched call diverge within ~15-30 s)."""
    h, exe = wm
    st = R.load_states(ROOT / 'wmev_data/u0_STATES_t300.json', sids=['rgu0_D2400_p40_s989200_t300'])
    monkeypatch.setattr(R, 'ZONE_S', (1e9, 1e9 + 1.0))
    tr = {}
    r = R.evaluate_states(h, exe, st, arms=['u0', 'wall_h_solo2', 'wall_h'], K=2, H=40.0, slots=112, max_rows=8,
                          trace=tr)[0]
    base = tr['per_step'][0]
    for j in (1, 2):
        assert np.array_equal(tr['per_step'][j], base)
    for arm in ('wall_h_solo2', 'wall_h'):
        assert r['arms'][arm]['road_w'] == r['arms']['u0']['road_w'] and r['arms'][arm]['first_div_s'] == [None, None]


@needs_wm
def test_wm2_no_feasible_wall_means_identical_rows(wm):
    h, exe = wm
    r = R.evaluate_state(h, exe, _fixture(wall=False), arms=['u0', 'wall_h_solo2', 'wall_h'], K=1, H=4.0, slots=48,
                         max_rows=8)
    assert r['arms']['wall_h']['activations'] == [0] and r['arms']['wall_h']['first_div_s'] == [None]
    assert r['arms']['wall_h']['road_w'] == r['arms']['u0']['road_w'] == r['arms']['wall_h_solo2']['road_w']


@needs_wm
def test_wm_moment_carries_no_world_labels():
    mom = R.wm_moment(_fixture())
    assert set(mom) == {'map', 't0', 'cavs', 'vehicles', 'counter'}
    blob = json.dumps(mom)
    for label in ('D3200', 'p60', '989200', 'penetration', 'demand', 'cell'):
        assert label not in blob


# ------------------------------------------------------------------------------------------ guards / states
def test_seed_and_path_guards(tmp_path):
    bad = [dict(_fixture(), seed=989650, sid='x')]
    p = tmp_path / 'states.json'
    p.write_text(json.dumps(bad))
    with pytest.raises(G.ForbiddenSeedError):
        R.load_states(p)
    with pytest.raises(G.ForbiddenPathError):
        R.load_states(tmp_path / 'holdout_states.json')
    rows = [dict(_fixture(), sid=f's{i}', cell=c) for i, c in enumerate(['A', 'A', 'B', 'A'])]
    p2 = tmp_path / 'states2.json'
    p2.write_text(json.dumps(rows))
    assert [m['sid'] for m in R.load_states(p2, max_per_cell=2)] == ['s0', 's1', 's2']


@pytest.mark.skipif(not TAPES.exists(), reason='WM-D tapes not present')
def test_states_from_one_tape_are_observable_only(tmp_path):
    rep = R.build_states(TAPES, tmp_path / 'st.json', cells=['D2400_p40'], max_worlds=1, ref_out=tmp_path / 'ref.json')
    assert rep['states'] == 1
    m = json.loads((tmp_path / 'st.json').read_text())[0]
    assert m['t0'] == 300 and 0.0 <= m['observed']['cav_entry_share'] <= 1.0
    assert len(m['observed']['counter']) == 2 and len(m['observed']['counter'][0]) == 30
    ref = json.loads((tmp_path / 'ref.json').read_text())['ref'][m['sid']]
    assert len(ref) == 9 and all(x is not None and x >= 0 for x in ref)


# ------------------------------------------------------------------------------------------ SUMO + analysis
@pytest.mark.skipif(not SUMO_WIN.exists(), reason='SUMO windows not extracted')
def test_sumo_windows_reproduce_the_registered_tables():
    """diff = strategy - u0 (veh-s, team, 300-1200 s) means from REGIMEU0_REPORT / LOWSHARE_U0_REPORT."""
    sumo = C.read_sumo_windows(SUMO_WIN)
    want = {('D3200_p60', 'wall_h_solo2'): -19981, ('D3200_p60', 'wall_h'): -17810, ('D3200_p40', 'wall_h_solo2'):
            -23406, ('D2800_p40', 'wall_h_solo2'): -354, ('D2400_p40', 'wall_h'): 9413,
            ('D3200_p30', 'wall_g'): -10699, ('D3200_p20', 'wall_h_solo2'): -7517}
    for (cell, arm), v in want.items():
        g = C._sumo_unit_gains(sumo, cell, arm, 9, 'team')
        assert -np.mean([x[0] for x in g.values()]) == pytest.approx(v, abs=1.0), (cell, arm)
    assert len(sumo['D3200_p20']['u0']) == 36 and len(sumo['D2400_p60']['wall_h']) == 24


def _scores_from_sumo(sumo, path, noise=0.0, flip=None):
    """Fake WM score file whose 'states' are the SUMO seeds (K=1): a perfect (or perturbed) predictor."""
    rng = np.random.default_rng(0)
    with open(path, 'w') as fh:
        for cell, arms in sumo.items():
            for seed in sorted(arms['u0']):
                out = {}
                for arm, d in arms.items():
                    w = np.array(d[seed]['road_w'], float)
                    if arm != 'u0':
                        w = w + noise * rng.standard_normal(len(w))
                        if flip and (cell, arm) in flip:
                            base = np.array(arms['u0'][seed]['road_w'], float)
                            w = 2 * base - w                    # benefit sign flipped
                    out[arm] = dict(road_w=[w.tolist()], src_w=[[0.0] * 9], n_w=[[100.0] * 9], activations=[3],
                                    first_act_s=[0.0 if arm != 'u0' else None], first_div_s=[0.5 if arm != 'u0'
                                                                                             else None], no_slot=[0])
                fh.write(json.dumps(dict(sid=f'{cell}_{seed}', cell=cell, seed=seed, t0=300, arms=out)) + '\n')


@pytest.mark.skipif(not SUMO_WIN.exists(), reason='SUMO windows not extracted')
def test_analysis_perfect_and_flipped_predictors(tmp_path):
    sumo = C.read_sumo_windows(SUMO_WIN)
    _scores_from_sumo(sumo, tmp_path / 'perfect.jsonl')
    res = C.analyze([tmp_path / 'perfect.jsonl'], SUMO_WIN, H=900, boot=50)
    p = res['point']
    assert p['n_pairs'] == 20 and p['sign_agreement'] == 1.0 and p['spearman_pooled'] == pytest.approx(1.0)
    assert p['calib_mae'] == pytest.approx(0.0) and all(r['class_agree'] for r in res['table'])
    assert all(r['wm_says_solo_dominates'] == r['sumo_solo_dominates'] for r in res['dominance'])
    pat = {(r['cell'], r['strategy']): r for r in res['pattern']}
    assert pat[('D2400_p40', 'wall_h')]['sumo'] == 'hurt' and pat[('D3200_p60', 'wall_h_solo2')]['sumo'] == 'help'
    _scores_from_sumo(sumo, tmp_path / 'flip.jsonl', flip={('D2400_p40', 'wall_h'), ('D2400_p60', 'wall_h')})
    res2 = C.analyze([tmp_path / 'flip.jsonl'], SUMO_WIN, H=900, boot=50)
    assert res2['point']['sign_agreement'] == pytest.approx(18 / 20)
    md = C.markdown(res)
    assert 'Regime pattern' in md and 'Group minus best solo' in md
    hz = C.horizon_check([tmp_path / 'perfect.jsonl'], SUMO_WIN)
    assert hz['H_star'] == 900 and hz['crn']['violations'] == 0


def test_cost_estimate_counts_rows(tmp_path):
    rows = [dict(_fixture(), sid=f's{i}', cell=c) for i, c in enumerate(['D3200_p60', 'D2400_p40', 'D3200_p60'])]
    p = tmp_path / 'st.json'
    p.write_text(json.dumps(rows))
    # one arm per call: D3200_p60 4 arms, D2400_p40 3 arms; 'auto' states/call = 32 // 8 = 4 but batches never mix
    # slot counts or arm lists -> batches [s0], [s1], [s2]
    est = R.cost_estimate(p, K=8, H=900, max_rows=32, s_per_step=0.1, workers=2, slots=448)
    assert est['calls'] == 4 + 3 + 4 and est['row_steps'] == (4 + 3 + 4) * 8 * 1800
    assert est['serial_minutes'] == pytest.approx(11 * 1800 * 0.1 / 60, abs=0.1)
    auto = R.cost_estimate(p, K=8, H=900, max_rows=32, s_per_step=0.1)
    assert auto['calls_by_slots'] == {448: 8, 256: 3}
    same = R.cost_estimate(p, K=8, H=900, max_rows=32, s_per_step=0.1, sids=['s0', 's2'])
    assert same['calls'] == 4                       # both D3200_p60 states share each arm's call (2 x 8 = 16 rows)
    k4 = R.cost_estimate(p, K=4, H=900, max_rows=32, s_per_step=0.1, sids=['s0', 's2'], states_per_call=1)
    assert k4['calls'] == 8


@needs_wm
@pytest.mark.skipif(not SUMO_WIN.exists(), reason='SUMO windows not extracted')
def test_end_to_end_score_cli_resume_and_analyze(tmp_path, wm):
    """score CLI on two fixture states (tiny H) -> resumable JSONL + meta -> analyze/markdown on the real SUMO side."""
    states = [_fixture(wall=True), dict(_fixture(wall=True), sid='fixture_D3200_p60_s989202_t300', seed=989202)]
    sp = tmp_path / 'states.json'
    sp.write_text(json.dumps(states))
    out = tmp_path / 'SCORES.jsonl'
    args = ['score', '--states', str(sp), '--checkpoint', str(WM2), '--u0', str(U0), '--K', '2', '--H', '4',
            '--max-rows', '16', '--states-per-call', '2', '--slots', '48', '--threads', '2', '--out', str(out)]
    R.main(args)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert [r['sid'] for r in rows] == [m['sid'] for m in states] and all('error' not in r for r in rows)
    meta = json.loads(Path(str(out) + '.meta.json').read_text())
    assert meta['u0_sha256'].startswith('d9042cc5') and meta['K'] == 2 and meta['slots'] == '48'
    R.main(args)                                                        # resume: nothing re-scored
    assert len(out.read_text().splitlines()) == 2
    with pytest.raises(ValueError):                                     # changed settings refused
        R.main([a if a != '2' else '3' for a in args])
    res = C.analyze([out], SUMO_WIN, H=100, boot=20)
    assert {r['strategy'] for r in res['table']} == {'wall_h_solo2', 'wall_h', 'wall_h_solo0'}
    assert all(r['wm_states'] == 2 for r in res['table'])
    assert 'Regime pattern' in C.markdown(res, C.horizon_check([out], SUMO_WIN))


@pytest.mark.skipif(not SUMO_WIN.exists(), reason='SUMO windows not extracted')
def test_analyze_cli_horizon_rule_and_by_h_table(tmp_path):
    sumo = C.read_sumo_windows(SUMO_WIN)
    _scores_from_sumo(sumo, tmp_path / 'perfect.jsonl')
    C.main(['analyze', '--scores', str(tmp_path / 'perfect.jsonl'), '--sumo-windows', str(SUMO_WIN), '--label', 'P',
            '--boot', '40', '--out', str(tmp_path / 'R.json')])
    res = json.loads((tmp_path / 'R.json').read_text())
    assert res['H'] == 900 and res['horizon_flag'].startswith('stable') and len(res['all_H']) == 9
    assert res['all_H']['300']['sign_agreement'] == 1.0          # a perfect predictor agrees at every horizon
    md = (tmp_path / 'R.md').read_text()
    assert 'By horizon H' in md and 'Horizon check' in md
    # a WM whose late windows blow up (x10 delay from window 5 on, every arm) loses the stable horizon there
    rows = [json.loads(x) for x in (tmp_path / 'perfect.jsonl').read_text().splitlines()]
    with open(tmp_path / 'blow.jsonl', 'w') as fh:
        for r in rows:
            for o in r['arms'].values():
                o['road_w'] = [[x * (10.0 if j >= 4 else 1.0) for j, x in enumerate(w)] for w in o['road_w']]
            fh.write(json.dumps(r) + '\n')
    hz = C.horizon_check([tmp_path / 'blow.jsonl'], SUMO_WIN)
    assert hz['H_star'] == 400

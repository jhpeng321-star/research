"""clean5c bank -> guidance converter: CPU contracts (synthetic bank rows + the real BANK_D3200_p60 when present)."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import json
from pathlib import Path

import numpy as np
import pytest

from . import fullpol_bank as fb, fullpol_ev as fe, fullpol_learn as fl
from .test_fullpol_ev_cpu import ev_cfg
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)

PROV = 'clean5c CPU fixture (MAIN 2026-11-08)'
RUN = Path('E:/research-dense/runs/active/C4-V7FT-20261017')
BANK = RUN/'wmverify_d3200p60'/'ANALYSIS_D3200_p60'


def bank_row(cell, fam, seed, diff, worst, members=2):
    groups = {g: -5. for g in fb.GROUPS}
    groups['origin:L2'] = worst
    return dict(schema='wmverify.bank.v1', cell=cell, family=fam, seed=seed, members=[f'c{i}' for i in range(members)],
                diff_vs_u0=diff, cell_groups_vs_u0=groups, executor='plant', strategy_key=fam, template=None,
                template_sha256=None, offer_condition=None, lcexec_parity=None, n2_parity_verdict='identical')


def fallback(cells=('D3200_p60', 'D2800_p60', 'D2400_p60')):
    rows = [dict(arm='B', regime=c, family='wall_g_solo2_gate0', baseline='mappo', n=24, mean_veh_s=900.,
                 lb95_veh_s=100., fair=True) for c in cells]
    return fl.guidance_doc(rows, sources=[dict(name='F8S', sha256='1'*64)], provenance=PROV)


ARMS = [{c: dict(wm=[dict(name='over'), dict(name='fairpair')], wm_fairer=['p'], structured=['struct'],
                random=['rnd']) for c in ('D3200_p60', 'D2800_p60', 'D2400_p60')}]
VERIFY = dict(families={f: dict(n2_parity=dict(verdict='identical')) for f in ('over', 'fairpair', 'p', 'struct', 'rnd')})


def write(tmp_path, rows):
    p = tmp_path/'BANK.jsonl'
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    return p


def test_priorities_multi_cav_only_and_fallback(cfg, tmp_path):
    rows = []
    for s in range(6):
        rows += [bank_row('D3200_p60', 'over', 989950+s, -5000.-100*s, 14.),        # congested, over 10 s: labelled
                 bank_row('D2800_p60', 'over', 989950+s, -5000.-100*s, 14.),        # less congested: rejected
                 bank_row('D2800_p60', 'fairpair', 989950+s, -3000.-50*s, 6.),
                 bank_row('D3200_p60', 'solo', 989950+s, -9000., 1., members=1),   # single CAV: never a target
                 bank_row('D3200_p60', 'struct', 989950+s, -9000., 1.),            # structured: never a target
                 bank_row('D3200_p60', 'rnd', 989950+s, -9000., 1.)]               # random: never a target
    doc = fb.build([write(tmp_path, rows)], fallback=fallback(), provenance=PROV, arms=ARMS, verify=VERIFY)
    by = {(r['regime'], r['family'], r['source']): r for r in doc['rows']}
    assert by[('D3200_p60', 'over', 'wm_bank')]['fair'] and by[('D3200_p60', 'over', 'wm_bank')]['fair_label'] == 'over_fair_bound'
    assert ('D2800_p60', 'over', 'wm_bank') not in by and by[('D2800_p60', 'fairpair', 'wm_bank')]['fair']
    assert doc['clean5c']['rejected']['D2800_p60/over']['reason'] == 'over_fair_bound'
    assert {doc['clean5c']['rejected'][f'D3200_p60/{f}']['reason'] for f in ('struct', 'rnd')} == {'not_wm_sourced'}
    assert all(r.get('fallback') for r in doc['rows'] if r['source'] == 'f8s_rule')
    assert ('D3200_p60', 'solo', 'wm_bank') not in by and 'D3200_p60/solo' in doc['clean5c']['skipped']
    assert doc['clean5c']['verified_cells'] == ['D2800_p60', 'D3200_p60']
    assert {k[0] for k in by if k[2] == 'f8s_rule'} == {'D2400_p60'}                # fallback only where unverified
    assert by[('D3200_p60', 'over', 'wm_bank')]['mean_veh_s'] == pytest.approx(5250.)   # gain = -(strategy - u0)
    on = ev_cfg(cfg, guidance=doc, parity=fb.eligible_families(doc))
    table, _ = fl.ev_table(on, 'B')
    assert fl.qualifying(table['D3200_p60']) == ['over'] and fl.qualifying(table['D2800_p60']) == ['fairpair']
    assert fl.qualifying(table['D2400_p60']) == ['wall_g_solo2_gate0']


def test_b_and_r_identical_except_guidance_and_cli(tmp_path):
    rows = [bank_row('D3200_p60', 'p', 989950+s, -4000.-10*s, 3.) for s in range(4)]
    path = write(tmp_path, rows)
    b = fb.build([path], fallback=fallback(), provenance=PROV, arm='B', arms=ARMS, verify=VERIFY)
    r = fb.build([path], fallback=fallback(), provenance=PROV, arm='R', arms=ARMS, verify=VERIFY)
    strip = lambda d: [{k: v for k, v in x.items() if k != 'arm'} for x in d['rows'] if x['source'] == 'wm_bank']  # noqa: E731
    assert strip(b) == strip(r) and {x['arm'] for x in r['rows']} == {'R'}       # R: no B fallback rows
    fpath, apath, vpath = tmp_path/'F.json', tmp_path/'A.json', tmp_path/'V.json'
    fpath.write_text(json.dumps(fallback()))
    apath.write_text(json.dumps(ARMS[0]))
    vpath.write_text(json.dumps(VERIFY))
    out = tmp_path/'G.json'
    args = ['--bank', str(path), '--fallback', str(fpath), '--arms', str(apath), '--verify', str(vpath),
            '--provenance', PROV, '--out', str(out)]
    fb.main(args)
    assert json.loads(out.read_text())['clean5c']['verified_cells'] == ['D3200_p60']
    with pytest.raises(SystemExit):
        fb.main(args)


def test_wm_groups_from_plan_json():
    plan = dict(cells=['D3200_p60'], families=dict(a=dict(cells=dict(D3200_p60='wm')),
                                                   b=dict(cells=dict(D3200_p60='structured'))))
    assert fb.wm_families([plan]) == {'D3200_p60': {'a'}}


def test_parity_gate_drops_and_logs(tmp_path):
    rows = [bank_row('D3200_p60', 'p', 989950+s, -4000.-10*s, 3.) for s in range(4)]
    rows = [dict(r, n2_parity_verdict=None) for r in rows]
    bad = dict(families={'p': dict(n2_parity=dict(verdict='not_representable'))})
    doc = fb.build([write(tmp_path, rows)], fallback=fallback(), provenance=PROV, arms=ARMS, verify=bad)
    assert doc['clean5c']['verified_cells'] == []                                   # falls back to F8S
    assert doc['clean5c']['rejected']['D3200_p60/p']['reason'] == 'n2_parity_not_representable'
    doc = fb.build([write(tmp_path, rows)], fallback=fallback(), provenance=PROV, arms=ARMS, verify=dict(families={}))
    assert doc['clean5c']['rejected']['D3200_p60/p']['reason'] == 'n2_parity_None'


def test_offer_condition_enforced_on_the_slate(monkeypatch):
    """A bank target is offered (hence an EV-BC target) only where its verified trigger/offer condition holds."""
    from . import fullpol_templates as ft
    cond = dict(trigger='lanes', own_wm_lanes=[0], zone_s=[-1800., -400.], spread_m=300.,
                roles=[dict(wm_lane=0), dict(wm_lane=1)], gate=None, start_gate=None)

    def r(v, s, ln, cav=True):
        return dict(vid=v, s_m=float(s), lane=ln, v_mps=20., is_cav=cav)
    own = r('e', -1000, 0)
    assert fb.offer_ok(cond, own, [own, r('p', -1100, 1)])
    assert not fb.offer_ok(cond, own, [own, r('h', -1100, 1, cav=False)])            # partner lane needs a CAV
    assert not fb.offer_ok(cond, own, [own, r('p', -1500, 1)])                        # partner beyond spread
    assert not fb.offer_ok(cond, r('e', -1000, 1), [r('p', -1100, 0)])                 # own lane not verified
    assert not fb.offer_ok(cond, r('e', -200, 0), [r('p', -250, 1)])                   # outside the zone
    assert not fb.offer_ok(dict(cond, gate={'any': 1}), own, [own, r('p', -1100, 1)])  # unevaluable gate: closed
    calls, real = [], fb.offer_ok
    monkeypatch.setattr(fb, 'offer_ok', lambda c, o, rows: calls.append(len(rows)) or real(c, o, rows))
    fp_cls = type('FP', (), {'from_dict': staticmethod(lambda d: d)})
    monkeypatch.setattr(ft, '_lc', lambda: type('LC', (), {'FreeProgram': fp_cls}))
    monkeypatch.setattr(ft, '_view', lambda frame: None)
    tpl = {'x': 1, 'source': dict(variant='wm_x')}
    slot = dict(classes=['verified'], templates=[dict(template=tpl, offer_condition=dict(cond, slots=1, cooldown_s=120))])

    def state(rows, ledger, now=500):
        return dict(frame=None, rows=rows, fullpol=slot, now=now, ledger=ledger)
    pair = [own, r('p', -1100, 1)]
    rng = np.random.default_rng(0)
    assert ft.propose(state(pair, {}), own, rng) == [tpl]
    assert ft.propose(state([own], {}), own, rng) == []                                # no partner: no target
    assert calls == [1, 0]
    # slots / cooldown (the bank's decision rule): no target while a session of the family runs or cools down
    assert ft.propose(state(pair, {'wm_x': dict(active=1, last_start_s=300)}), own, rng) == []
    assert ft.propose(state(pair, {'wm_x': dict(active=0, last_start_s=400)}), own, rng) == []    # 500 < 520
    assert ft.propose(state(pair, {'wm_x': dict(active=0, last_start_s=380)}), own, rng) == [tpl]  # 500 >= 500
    assert ft.propose(state(pair, None), own, rng) == []                                  # no ledger: closed
    assert calls == [1, 0, 1]


def test_offer_wall_member_rule_exact():
    """wall: one in-zone CAV per lane 0/1/2, all members (acting CAV included) in ONE spread_m window."""
    cond = dict(trigger='wall', own_wm_lanes=[0, 1], zone_s=[-1724.56, -324.56], spread_m=200.,
                roles=[dict(wm_lane=1), dict(wm_lane=0)], gate=None, start_gate=None, slots=1, cooldown_s=120)

    def r(v, s, ln):
        return dict(vid=v, s_m=float(s), lane=ln, v_mps=20., is_cav=True)
    own = r('e', -1000, 0)
    assert fb.offer_ok(cond, own, [r('a', -950, 1), r('b', -850, 2)])                 # spread 150: offered
    assert fb.offer_ok(cond, own, [r('a', -1100, 1), r('b', -900, 2)])                # spread 200: offered
    assert not fb.offer_ok(cond, own, [r('a', -1150, 1), r('b', -850, 2)])            # 150 behind + 150 ahead
    assert fb.offer_ok(cond, own, [r('a', -1150, 1), r('b', -850, 2), r('c', -950, 1)])   # another lane-1 CAV fits
    assert not fb.offer_ok(cond, own, [r('a', -950, 1), r('b', -1800, 2)])            # lane-2 partner out of zone
    assert not fb.offer_ok(cond, r('e', -400, 0), [r('a', -350, 1), r('b', -300, 2)])  # partner past the zone end
    assert not fb.offer_ok(cond, own, [r('a', -950, 1)])                               # lane 2 missing
    assert not fb.offer_ok(cond, r('e', -1000, 2), [r('a', -950, 1), r('b', -900, 0)])  # own lane 2 not verified
    assert fb.session_ok(cond, None, 300) and not fb.session_ok(cond, False, 300)


def test_real_bank_d3200_p60(cfg):
    if not (BANK/'BANK_D3200_p60.jsonl').exists():
        pytest.skip('WMVERIFY bank not present')
    arms = RUN/'wmc_strategy_v3'/'ARMS_D3200_p60.json'
    doc = fb.build([BANK/'BANK_D3200_p60.jsonl'], fallback=fallback(), analysis_paths=[BANK/'ANALYSIS_ALLOWED.json'],
                   provenance=PROV, arms=[json.loads(arms.read_text())], verify=json.loads((BANK/'VERIFY.json').read_text()))
    rej = doc['clean5c']['rejected']
    # this pilot bank's WM arm has lb95 <= 0; structured / random winners are never WM targets
    assert {rej[f'D3200_p60/{f}']['reason'] for f in ('wall_g_duo12_stagger40', 'wall_g_relay21',
                                                         'rand_25071e07bf236408')} == {'not_wm_sourced'}
    assert rej['D3200_p60/mut_59e2cfa0ff49510a']['reason'] == 'lb95<=0'
    assert doc['clean5c']['verified_cells'] == [] and all(r['source'] == 'f8s_rule' for r in doc['rows'])
    assert fe.settings(ev_cfg(cfg, guidance=doc, parity=fb.eligible_families(doc)))


def test_real_bank_mc(cfg):
    mc = RUN/'wmverify_mc'/'ANALYSIS_MC'
    if not (mc/'BANK_MC.jsonl').exists():
        pytest.skip('wmv_mc bank not present')
    doc = fb.build([mc/'BANK_MC.jsonl'], fallback=fallback(), analysis_paths=[mc/'ANALYSIS_ALLOWED.json'],
                   provenance=PROV, arms=[json.loads((mc/'PLAN.json').read_text())],
                   verify=json.loads((mc/'VERIFY.json').read_text()))
    bank = [r for r in doc['rows'] if r['source'] == 'wm_bank']
    assert bank and all(r['members'] >= 2 and r['n2_parity'] == 'identical' and r['lb95_veh_s'] > 0 for r in bank)
    assert all(t['offer_condition'] for t in doc['clean5c']['templates'])
    assert not any(r['fair_label'] == 'over_fair_bound' for r in bank if not r['regime'].startswith('D3200'))
    assert fe.settings(ev_cfg(cfg, guidance=doc, parity=fb.eligible_families(doc)))


def test_protocol_ledger_counts_sessions_per_variant():
    """LCEXEC Protocol.ledger(): active sessions + last start per template source variant (generator gate input)."""
    from . import lc_families as lf
    proto = lf.Protocol.__new__(lf.Protocol)
    proto.reset()
    tpl = type('T', (), {'source': dict(variant='wm_x')})()
    proto.sessions = {'a': dict(template=tpl)}
    proto.started = {'wm_x': [300, 420]}
    assert proto.ledger() == {'wm_x': dict(active=1, last_start_s=420)}
    proto.sessions = {}
    assert proto.ledger() == {'wm_x': dict(active=0, last_start_s=420)}

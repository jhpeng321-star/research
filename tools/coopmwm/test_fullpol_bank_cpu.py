"""clean5c bank -> guidance converter: CPU contracts (synthetic bank rows + the real BANK_D3200_p60 when present)."""
# ruff: noqa: F811 (imported pytest fixtures are test parameters)
import json
from pathlib import Path

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
                 bank_row('D3200_p60', 'solo', 989950+s, -9000., 1., members=1)]   # single CAV: never a bank target
    doc = fb.build([write(tmp_path, rows)], fallback=fallback(), provenance=PROV)
    by = {(r['regime'], r['family'], r['source']): r for r in doc['rows']}
    assert by[('D3200_p60', 'over', 'wm_bank')]['fair'] and by[('D3200_p60', 'over', 'wm_bank')]['fair_label'] == 'over_fair_bound'
    assert ('D2800_p60', 'over', 'wm_bank') not in by and by[('D2800_p60', 'fairpair', 'wm_bank')]['fair']
    assert doc['clean5c']['rejected']['D2800_p60/over']['reason'] == 'over_fair_bound'
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
    b = fb.build([path], fallback=fallback(), provenance=PROV, arm='B')
    r = fb.build([path], fallback=fallback(), provenance=PROV, arm='R')
    strip = lambda d: [{k: v for k, v in x.items() if k != 'arm'} for x in d['rows'] if x['source'] == 'wm_bank']  # noqa: E731
    assert strip(b) == strip(r) and {x['arm'] for x in r['rows']} == {'R'}       # R: no B fallback rows
    fpath = tmp_path/'F.json'
    fpath.write_text(json.dumps(fallback()))
    out = tmp_path/'G.json'
    fb.main(['--bank', str(path), '--fallback', str(fpath), '--provenance', PROV, '--out', str(out)])
    assert json.loads(out.read_text())['clean5c']['verified_cells'] == ['D3200_p60']
    with pytest.raises(SystemExit):
        fb.main(['--bank', str(path), '--fallback', str(fpath), '--provenance', PROV, '--out', str(out)])


def test_real_bank_d3200_p60(cfg):
    if not (BANK/'BANK_D3200_p60.jsonl').exists():
        pytest.skip('WMVERIFY bank not present')
    doc = fb.build([BANK/'BANK_D3200_p60.jsonl'], fallback=fallback(), analysis_paths=[BANK/'ANALYSIS_ALLOWED.json'],
                   provenance=PROV)
    bank = [r for r in doc['rows'] if r['source'] == 'wm_bank']
    assert bank and all(r['members'] >= 2 for r in bank)
    raw = json.loads((BANK/'ANALYSIS_ALLOWED.json').read_text())['cells']['D3200_p60']
    for r in bank:
        assert r['lb95_veh_s'] == pytest.approx(-raw[r['family']]['ci95'][1])
    elig = {r['family'] for r in bank if r['fair']}
    assert elig and all(r['lb95_veh_s'] > 0 and r['fair'] for r in bank)
    assert 'wall_g_solo2' not in {r['family'] for r in bank}                          # single-CAV: excluded
    tpl = doc['clean5c']['templates']
    assert {t['family'] for t in tpl} <= elig and all(t['template']['source']['variant'] == t['family'] for t in tpl)
    assert fe.settings(ev_cfg(cfg, guidance=doc, parity=fb.eligible_families(doc)))

import json
from types import SimpleNamespace

import pytest

from wms.control.auditfix_authority import build, digest, LEGACY_THRESHOLD, CCT_SIGMA
from wms.control.coopw import HorizonConfig
from wms.control.looplong import read_boundary


def inputs(tmp_path):
    def write(name, doc):
        path = tmp_path / name
        path.write_text(json.dumps(doc), encoding='utf-8')
        return path
    wm = write('new-wm.pt', {'new': 'weights'})
    cfg = write('wm.json', {'new': 'config'})
    threshold = write('registered.json', {'threshold': LEGACY_THRESHOLD})
    gate = dict(schema='coopw_horizon_gate/1', wm_sha256=digest(wm),
                wm_config_sha256=digest(cfg), threshold=LEGACY_THRESHOLD,
                threshold_source={'sha256': digest(threshold)},
                entrant_control=True, deterministic_crn=True, samples=8,
                horizons={str(h): {'sign_agreement': .9, 'verdict': 'BASE' if h == 10 else 'PASS'}
                          for h in (10, 30, 60, 90)})
    gate_path = write('gate.json', gate)
    cct = dict(schema='gated300_cct1/1', wm_sha256=digest(wm), wm_config_sha256=digest(cfg),
               panel='pairs300', policy_authorized=True, source_sha256=digest(gate_path),
               legacy_threshold=LEGACY_THRESHOLD, legacy_horizons=gate['horizons'],
               H_C=0, entrant_control=True, deterministic_crn=True, samples=8,
               bootstrap=dict(unit='world', draws=2000, seed=0, ci=.95),
               horizons={str(h): dict(sigma=sigma, threshold=2*sigma, sign_agreement=.5,
                                      coverage=1., wrong_mean_abs_real=100., verdict='FAIL')
                         for h, sigma in CCT_SIGMA.items()})
    template = write('old-boundary.json', dict(schema='looplong_boundary/1',
        horizon_s=90, imagination_horizon_s=90, previous_horizon_s=90,
        flags={'auto_horizon': True, 'long_credit': True}, short_nonregression=True,
        pairs='unchanged-pairs', native_returns='unchanged-native'))
    acceptance = write('ACCEPT.json', dict(decision='ACCEPT', inputs={
        'candidate_sha256': digest(wm), 'config_sha256': digest(cfg)}))
    return SimpleNamespace(boundary_template=template, wm_ckpt=wm, wm_config=cfg,
        gate=gate_path, cct1=write('cct1.json', cct), threshold=threshold,
        acceptance=acceptance, authority_out=tmp_path / 'AUTHORITY.json', out=tmp_path / 'BOUNDARY.json')


@pytest.mark.parametrize('hc', [0, 300])
def test_new_wm_measured_authority_binds_boundary_and_horizon(tmp_path, monkeypatch, hc):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    args = inputs(tmp_path)
    if hc:
        cct = json.loads(args.cct1.read_text())
        cct['H_C'] = hc
        for row in cct['horizons'].values():
            row.update(sign_agreement=1., coverage=1., wrong_mean_abs_real=0., verdict='PASS')
        args.cct1.write_text(json.dumps(cct))
    result = build(args)
    assert result['pairs'] == 'unchanged-pairs'
    assert read_boundary(args.out, swap=True) == result
    h = HorizonConfig(enabled=True, horizons_s=[10, 90, 120, 300],
                      curriculum_updates=[0, 1, 2, 3], gate_file=str(args.authority_out))
    assert h.bind([digest(args.wm_ckpt)])['measured_gate']
    assert h.allowed_stage == 1  # CCT never relabels raw120/300 credit PASS.
    with pytest.raises(ValueError, match='NEW files'):
        build(args)
    with pytest.raises(ValueError, match='WM mismatch'):
        h.bind(['aa' * 32])
    args.cct1.write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        read_boundary(args.out)


@pytest.mark.parametrize('mutation', ['wm', 'config', 'accept', 'short', 'threshold',
                                    'cct_source', 'cct_panel', 'cct_hc', 'cct_metrics'])
def test_refuses_unbound_or_failed_measurements(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    args = inputs(tmp_path)
    if mutation in ('wm', 'config'):
        getattr(args, 'wm_ckpt' if mutation == 'wm' else 'wm_config').write_text('changed')
    else:
        path = args.acceptance if mutation == 'accept' else args.gate if mutation in ('short', 'threshold') else args.cct1
        doc = json.loads(path.read_text())
        if mutation == 'accept':
            doc['decision'] = 'REJECT'
        elif mutation == 'short':
            doc['horizons']['30']['sign_agreement'] = .1
        elif mutation == 'threshold':
            doc['threshold'] = .5
        elif mutation == 'cct_source':
            doc['source_sha256'] = 'bb' * 32
        elif mutation == 'cct_hc':
            doc['H_C'] = 300
        elif mutation == 'cct_metrics':
            doc['horizons'] = {}
        else:
            doc['panel'] = 'validation'
        path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        build(args)
    assert not args.out.exists() and not args.authority_out.exists()

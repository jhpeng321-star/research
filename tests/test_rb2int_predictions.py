import copy
import json

import pytest

from tools.rb2int.predictions import read_model
from tools.rbeval.core import ARMS, HS, REGIONS, breakdown, digest


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def fixture(root, *, futures=8):
    real = []
    bins = {r: dict(speed=[20.]*66, post_counts=[1]*60,
        events={str(h): breakdown([20.]*(6+h//10)) for h in HS}) for r in REGIONS}
    for arm in ARMS:
        real.append(dict(seed=969550, arm=arm, initial_sha256='initial', bins=copy.deepcopy(bins)))
    source = {'/runtime/src/wms/model.py': 'frozen'}
    identity = dict(model='wm2', K=8, arms=list(ARMS), t0=300, horizon=600, mix=True,
        native_lanes=True, wm_sha256='wm', config_sha256='config', candidate_sha256='candidate',
        export_sha256='export', seed=965140, update=190, switches={}, sources=source,
        real_receipt_sha256='real')
    job = root/'run'
    write(job/'IDENTITY.json', identity)
    for k in range(futures):
        episodes = [dict(seed=969550, future=k, arm=arm, initial_sha256='initial',
            bins=copy.deepcopy(bins), clocks={str(h): dict(team_delay_vs=3., road_delay_vs=2., source_wait_vs=1.) for h in HS})
            for arm in ARMS]
        write(job/f'w969550_k{k}.json', dict(schema='rbeval/1', seed=969550, future=k,
            identity_sha256=digest(job/'IDENTITY.json'), arms=list(ARMS), episodes=episodes))
    reg = root/'registration.json'
    write(reg, dict(final_comparison_worlds=[969550]))
    bridge = root/'bridge.json'
    write(bridge, dict(passed=True))
    release = root/'release.json'
    write(release, dict(status='READY', exposure_audit_passed=True, registration_sha256=digest(reg),
        execution_purposes=['final_comparison'],
        models=dict(wm2=dict(sources=source, config_sha256='config'))))
    manifest = root/'runs.json'
    write(manifest, dict(schema='rb2int-approved-runs/1', registration_path=str(reg),
        registration_sha256=digest(reg), release_path=str(release), release_sha256=digest(release),
        models=dict(WM2=dict(identity_name='wm2', wm_sha256='wm', config_sha256='config',
            candidate_sha256='candidate', export_sha256='export', frozen_model_sources=source,
            development_bridge_passed=True, bridge_receipts={str(bridge): digest(bridge)},
            runs=[dict(root=str(job), identity_sha256=digest(job/'IDENTITY.json'), real_receipt_sha256='real')]))))
    return manifest, real, job


def test_partial_ensemble_is_reported_but_never_scored_as_k8(tmp_path):
    manifest, real, job = fixture(tmp_path, futures=7)
    pred, audit = read_model(manifest, 'WM2', real, [969550], partial=True)
    assert pred == {} and audit['complete_worlds'] == []
    assert audit['missing_tasks'] == [(969550, 7)]
    with pytest.raises(ValueError, match='incomplete registered'):
        read_model(manifest, 'WM2', real, [969550], partial=False)


def test_original_and_extended_sources_need_explicit_approval(tmp_path):
    manifest, real, job = fixture(tmp_path)
    pred, audit = read_model(manifest, 'WM2', real, [969550])
    assert len(pred) == 5 and audit['complete_worlds'] == [969550]
    identity = json.loads((job/'IDENTITY.json').read_text())
    identity['config_sha256'] = 'changed'
    write(job/'IDENTITY.json', identity)
    with pytest.raises(ValueError, match='approved run identity changed'):
        read_model(manifest, 'WM2', real, [969550])


def test_duplicate_forecast_roots_are_not_pooled(tmp_path):
    manifest, real, job = fixture(tmp_path)
    data = json.loads(manifest.read_text())
    data['models']['WM2']['runs'] *= 2
    write(manifest, data)
    with pytest.raises(ValueError, match='duplicate forecast task'):
        read_model(manifest, 'WM2', real, [969550])


def test_bad_delay_ledger_or_future_identity_cannot_enter_scores(tmp_path):
    manifest, real, job = fixture(tmp_path)
    path = job/'w969550_k0.json'
    task = json.loads(path.read_text())
    task['episodes'][0]['clocks']['600']['team_delay_vs'] = 999
    write(path, task)
    with pytest.raises(ValueError, match='inconsistent source/road delay'):
        read_model(manifest, 'WM2', real, [969550])

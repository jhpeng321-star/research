"""Opt-in parity against the committed track-base, loaded in memory only."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import torch

from tools.coopmwm import rerun_cpus as rc


@pytest.fixture
def baseline(monkeypatch):
    import tools.coopmwm as package
    modules = {}
    for name in ('rerun_cpus', 'v2v_runner', 'v2v_launch3'):
        frozen = os.environ.get('COOPMWM_TRACK_BASELINE_DIR')
        raw = ((Path(frozen)/(name+'.py')).read_bytes() if frozen else subprocess.check_output(
            ['git', 'show', f'13f184b5:tools/coopmwm/{name}.py']))
        assert hashlib.sha256(raw).hexdigest() == {
            'rerun_cpus':'75d6970dc9bd1aa8b94caab22b0c40507e3c302b39142b76b340ad9a7c8deb05',
            'v2v_runner':'ed3b610723b66022250f667498cfd06065aa2d77e1c5382f021bd638941d7a98',
            'v2v_launch3':'e027c75e8adbec98545fbd3a27889f71915598bfdee121afb42c58101fb02ecf'}[name]
        code = raw.decode('utf-8')
        module = types.ModuleType('tools.coopmwm._t2c_baseline_'+name)
        module.__package__ = 'tools.coopmwm'
        exec(compile(code, '<committed-track-base:'+name+'>', 'exec'), module.__dict__)
        modules[name] = module
    module = modules['rerun_cpus']
    original = module.runtime_config
    def runtime_config(*args, **kwargs):
        # The historical producer must also use its historical V2V producer
        # and pins. Restore current imports before testing the current runtime.
        with monkeypatch.context() as patch:
            for name in ('v2v_runner', 'v2v_launch3'):
                patch.setattr(package, name, modules[name], raising=False)
                patch.setitem(sys.modules, 'tools.coopmwm.'+name, modules[name])
            return original(*args, **kwargs)
    module.runtime_config = runtime_config
    return module


def test_absent_and_false_opt_in_match_track_base_without_opt_in_only_imports(tmp_path, monkeypatch, baseline):
    # practice_weighting is now a shared legacy practice-config dependency;
    # the absent contract covers opt-in-only components, not that validator.
    for name in ('moves', 'trigger_policy', 'coop_layer', 'group_credit', 'merge_features',
                 'dense_teacher', 'plan_augment', 'coop_reward'):
        monkeypatch.setitem(sys.modules, 'tools.coopmwm.'+name, None)
    kwargs = dict(root=str(tmp_path/'run'), reference_root=str(tmp_path/'refs'),
                  source_root=str(tmp_path/'src'), common_checkpoint_sha256='a'*64)
    base = dict(workers=2, planner=dict(lookahead_s=720))
    expected = baseline.runtime_config(base, **kwargs)
    for opt in ({}, dict(track2=False)):
        before = torch.get_rng_state().clone()
        actual = rc.runtime_config(base, **kwargs, **opt)
        assert torch.equal(before, torch.get_rng_state())
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        assert rc.check_pins(actual)
        assert rc.track2_components(actual, 'B', adapters={}) == {}
        assert rc.episode_options(actual, 'B') == {}
        assert rc.run_tracks(actual, None, object(), legacy_run=lambda c: c) is actual


def test_public_preparation_and_cli_match_committed_baseline(baseline):
    import sys
    for opt in ({}, dict(track2=False)):
        assert rc.quote(**opt) == baseline.quote()
        assert rc.budget(**opt) == baseline.budget()
        assert rc.wall_clock(**opt) == baseline.wall_clock()
    for args in ([], ['--track2']):
        value = json.loads(subprocess.check_output(
            [sys.executable, '-B', '-m', 'tools.coopmwm.rerun_cpus', 'quote', *args], text=True))
        module = rc if args else baseline
        opt = dict(track2=True) if args else {}
        assert value == dict(quote=module.quote(**opt), budget=module.budget(**opt),
                             wall_clock=module.wall_clock(**opt))
    assert rc.quote(track2=True) != baseline.quote()


def test_baseline_runtime_isolates_historical_preparation_dependencies(tmp_path, monkeypatch, baseline):
    from tools.coopmwm import v2v_runner
    def unexpected(*args, **kwargs):
        pytest.fail('historical baseline must not use current V2V preparation')
    monkeypatch.setattr(v2v_runner, 'prepare_intent', unexpected)
    cfg = baseline.runtime_config({}, root=str(tmp_path/'run'), reference_root=str(tmp_path/'refs'),
                                  source_root=str(tmp_path/'src'), common_checkpoint_sha256='a'*64)
    assert cfg['practice_starts'] is True
    assert 'practice_weighting' not in cfg


def test_default_producer_request_and_preflight_flow(tmp_path, baseline):
    prep, stage, _ = legacy_preparation(tmp_path, baseline)
    request = prep/rc.BUDGET_REQUEST
    request.write_text(json.dumps(dict(quote=rc.quote(), budget=rc.budget())), encoding='utf-8')
    app_path = prep/rc.APPROVAL
    app = json.loads(app_path.read_text(encoding='utf-8'))
    app['budget_request_sha256'] = rc.sha(request)
    app_path.write_text(json.dumps(app), encoding='utf-8')
    assert rc.preflight_dry(prep, stage_dir=stage)['ready_for_registered_launch']


def test_clean_profile_has_fair_enabled_consumers_and_costs(tmp_path):
    cfg = rc.runtime_config({}, root=str(tmp_path/'run'), reference_root=str(tmp_path/'refs'),
        source_root=str(tmp_path/'src'), common_checkpoint_sha256='a'*64, track2=True)
    assert cfg['launch_profile'] == 'track2_clean_run'
    assert cfg['teacher']['B']['kind'] == 'wm_dense'
    assert cfg['teacher']['R']['kind'] == 'random'
    for name in ('track2_reward', 'plan_augment', 'practice_weighting'):
        assert cfg[name]['enabled'] and cfg[name]['arms'] == ['B', 'R']
    assert not cfg['coop_reward']['enabled']
    assert not cfg['dense_teacher']['enabled'] and not cfg['group_credit']['enabled']
    assert rc.budget(track2=True)['phases']['teacher_verify'] == 0
    assert rc.budget(track2=True)['phases']['augment_verify'] > 0
    assert rc.budget(track2=True)['phases']['early_probes'] == 18*rc.ISOLATED_SIM_S_MEASURED


@pytest.mark.parametrize('opt', [{}, dict(track2=False)], ids=['absent', 'false'])
def test_off_root_authority_and_legacy_ledger_match_baseline(tmp_path, monkeypatch, baseline, opt):
    from tools.coopmwm import accel_storage
    from tools.coopmwm.accel_protocol import SUPERSEDED_CHARGE
    monkeypatch.setattr(accel_storage, '_reserve', accel_storage._reserve)
    def unexpected():
        pytest.fail('OFF root/ledger must not consult the Track 2 criterion')
    monkeypatch.setattr(rc, 'criterion_document', unexpected)
    expected_root, actual_root = tmp_path/'baseline', tmp_path/'current'
    baseline.init_root(expected_root, amendment_shas=('c'*64,))
    rc.init_root(actual_root, amendment_shas=('c'*64,), **opt)
    for name in ('AUTHORITY.md', 'ACCOUNTING.claim', 'CHARGES.jsonl'):
        assert (actual_root/name).read_bytes() == (expected_root/name).read_bytes()
    # Admit the very same root created by the committed baseline, not only a new root.
    assert rc.own_ledger(expected_root, **opt) == baseline.own_ledger(expected_root)
    cfg = rc.runtime_config({}, root=str(expected_root), reference_root=str(tmp_path/'refs'),
                            source_root=str(tmp_path/'src'), common_checkpoint_sha256='a'*64, **opt)
    expected = baseline.bind_ledger(expected_root, cfg)
    expected_cap = accel_storage._reserve.__globals__['CAP']
    assert rc.bind_ledger(expected_root, cfg) == expected
    assert accel_storage._reserve.__globals__['CAP'] == expected_cap == cfg['budget']['total']-SUPERSEDED_CHARGE
    assert accel_storage._reserve.__globals__['CAPS'] == cfg['budget']['phases']


@pytest.mark.parametrize('opt', [{}, dict(track2=False)], ids=['absent', 'false'])
def test_off_charges_match_baseline_at_total_phase_and_duplicate_boundaries(tmp_path, monkeypatch, baseline, opt):
    from tools.coopmwm import accel_storage
    from tools.coopmwm.accel_protocol import SUPERSEDED_CHARGE
    monkeypatch.setattr(accel_storage, '_reserve', accel_storage._reserve)
    # Phase caps permit the whole total, exposing the historical total-CAP subtraction.
    b = dict(total=SUPERSEDED_CHARGE+10, phases=dict(train=SUPERSEDED_CHARGE+10, lesson=2))
    roots = [tmp_path/'baseline', tmp_path/'current']
    for module, root, kwargs in ((baseline, roots[0], {}), (rc, roots[1], opt)):
        module.init_root(root, **kwargs)
        module.charge(root, [dict(id='first', sim_s=6)], 'train', budget=b, **kwargs)
        before = (root/'CHARGES.jsonl').read_bytes()
        for jobs, phase, error in (([dict(id='first', sim_s=1)], 'train', 'duplicate charge'),
                                  ([dict(id='phase', sim_s=3)], 'lesson', 'whole-wave budget blocked'),
                                  ([dict(id='too-large', sim_s=5)], 'train', 'whole-wave budget blocked')):
            with pytest.raises(ValueError, match=error):
                module.charge(root, jobs, phase, budget=b, **kwargs)
            assert (root/'CHARGES.jsonl').read_bytes() == before
        module.charge(root, [dict(id='exact-cap', sim_s=4)], 'train', budget=b, **kwargs)
        with pytest.raises(ValueError, match='whole-wave budget blocked'):
            module.charge(root, [dict(id='over-cap', sim_s=1)], 'train', budget=b, **kwargs)
    assert (roots[0]/'CHARGES.jsonl').read_bytes() == (roots[1]/'CHARGES.jsonl').read_bytes()


def legacy_preparation(tmp_path, baseline):
    from tools.coopmwm.exact_grade import _seal
    prep, stage = tmp_path/'prep', tmp_path/'stage'
    prep.mkdir()
    dest = stage/'home'/'ubuntu'/'legacy'/'B_checkpoint_000.pt'
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b'synthetic legacy checkpoint')
    # Genuine legacy schema: no required-role inventory or Track 2 bindings.
    man = _seal(dict(schema='coopmwm.rerun_cpus.staging_manifest.v1', home=baseline.HOME,
                    entries={'home/ubuntu/legacy/B_checkpoint_000.pt':
                             dict(sha256=baseline.sha(dest), bytes=dest.stat().st_size,
                                  kind='checkpoint_u0')}), 'manifest_sha256')
    (prep/rc.MANIFEST_FILE).write_text(json.dumps(man), encoding='utf-8')
    (prep/rc.BUDGET_REQUEST).write_text(json.dumps(dict(quote=baseline.quote(), budget=baseline.budget())),
                                      encoding='utf-8')
    (prep/rc.OWNER_QUOTE).write_text('synthetic owner quote', encoding='utf-8')
    (prep/rc.VERDICT_FILE).write_text(json.dumps(dict(verdict='positive')), encoding='utf-8')
    app = dict(owner_approved=True, effective_imitation_commit=baseline.EFFECTIVE_IMITATION_COMMIT,
               budget_request_sha256=baseline.sha(prep/rc.BUDGET_REQUEST),
               owner_quote_sha256=baseline.sha(prep/rc.OWNER_QUOTE),
               verdict_file_sha256=baseline.sha(prep/rc.VERDICT_FILE))
    (prep/rc.APPROVAL).write_text(json.dumps(app), encoding='utf-8')
    return prep, stage, dest


@pytest.mark.parametrize('opt', [{}, dict(track2=False)], ids=['absent', 'false'])
def test_off_approved_preflight_matches_baseline_receipt_and_refusals(tmp_path, monkeypatch, baseline, opt):
    prep, stage, dest = legacy_preparation(tmp_path, baseline)
    def unexpected(*a, **kw):
        pytest.fail('legacy preflight must not use Track 2 registration or inventory checks')
    for name in ('criterion_document', 'budget_request_document', 'validate_manifest', 'own_ledger'):
        monkeypatch.setattr(rc, name, unexpected)
    expected = baseline.preflight_dry(prep, stage_dir=stage)
    assert expected['ready_for_registered_launch'] and expected['new_sim_s'] == 0
    assert rc.preflight_dry(prep, stage_dir=stage, **opt) == expected
    before = {p.name: p.read_bytes() for p in prep.iterdir()}
    original_app = before[rc.APPROVAL]
    # Both the approved pass and every legacy refusal must remain read-only.
    for field, value in (('owner_approved', False), ('budget_request_sha256', '0'*64),
                         ('owner_quote_sha256', '0'*64), ('verdict_file_sha256', '0'*64),
                         ('effective_imitation_commit', '0'*40)):
        app = json.loads(original_app)
        app[field] = value
        (prep/rc.APPROVAL).write_text(json.dumps(app), encoding='utf-8')
        with pytest.raises(SystemExit) as old:
            baseline.preflight_dry(prep, stage_dir=stage)
        with pytest.raises(SystemExit) as new:
            rc.preflight_dry(prep, stage_dir=stage, **opt)
        assert str(new.value) == str(old.value)
    (prep/rc.APPROVAL).write_bytes(original_app)
    dest.write_bytes(b'corrupt synthetic checkpoint')
    with pytest.raises(SystemExit) as old:
        baseline.preflight_dry(prep, stage_dir=stage)
    with pytest.raises(SystemExit) as new:
        rc.preflight_dry(prep, stage_dir=stage, **opt)
    assert str(new.value) == str(old.value)
    assert {p.name: p.read_bytes() for p in prep.iterdir()} == before


def test_on_ledger_refuses_legacy_authority_and_keeps_full_new_cap(tmp_path, monkeypatch, baseline):
    from tools.coopmwm import accel_storage
    monkeypatch.setattr(accel_storage, '_reserve', accel_storage._reserve)
    root = tmp_path/'legacy'
    baseline.init_root(root)
    b = dict(total=10, phases=dict(train=10))
    for call in (lambda: rc.own_ledger(root, track2=True),
                 lambda: rc.bind_ledger(root, dict(budget=b, track2=True)),
                 lambda: rc.charge(root, [dict(id='blocked', sim_s=1)], 'train', budget=b, track2=True)):
        with pytest.raises(ValueError, match='pre-registered.*criterion'):
            call()
    assert (root/'CHARGES.jsonl').read_bytes() == b''
    root = tmp_path/'track2'
    rc.init_root(root, track2=True)
    rc.charge(root, [dict(id='full-new-cap', sim_s=10)], 'train', budget=b, track2=True)
    assert accel_storage._reserve.__globals__['CAP'] == 10
    with pytest.raises(ValueError, match='whole-wave budget blocked'):
        rc.charge(root, [dict(id='over', sim_s=1)], 'train', budget=b, track2=True)


def test_on_preflight_cannot_admit_approved_legacy_preparation(tmp_path, baseline):
    prep, stage, _ = legacy_preparation(tmp_path, baseline)
    with pytest.raises(SystemExit, match='differs from the 3-arm quote'):
        rc.preflight_dry(prep, stage_dir=stage, track2=True)


def test_track2_runtime_explicitly_selects_strict_preflight(tmp_path, monkeypatch):
    from tools.coopmwm.rerun_tracks_runtime import Runtime
    cfg = rc.runtime_config({}, root=str(tmp_path/'run'), reference_root=str(tmp_path/'refs'),
                            source_root=str(tmp_path/'src'), common_checkpoint_sha256='a'*64, track2=True)
    rc.init_root(cfg['run_root'], track2=True)
    class AdmissionReached(Exception):
        pass
    def preflight(prep, *, stage_dir, track2=False):
        assert track2 is True
        raise AdmissionReached
    monkeypatch.setattr(rc, 'preflight_dry', preflight)
    backend = types.SimpleNamespace(cpu_plant=False, prep_dir=tmp_path/'prep', stage_dir=tmp_path/'stage')
    with pytest.raises(AdmissionReached):
        Runtime(cfg, {}, backend)


def test_track2_runtime_ledger_charge_requires_criterion(tmp_path, monkeypatch, baseline):
    import threading
    from tools.coopmwm import accel_storage
    from tools.coopmwm.rerun_tracks_runtime import Runtime
    monkeypatch.setattr(accel_storage, '_reserve', accel_storage._reserve)
    runtime = Runtime.__new__(Runtime)
    runtime.root = tmp_path/'run'
    baseline.init_root(runtime.root)
    runtime.cfg = dict(budget=dict(total=10, phases=dict(train=10)))
    runtime.smoke = runtime.synthetic = False
    runtime.reserve_lock = threading.Lock()
    runtime.spend, runtime.seen_jobs = dict(train=0), set()
    with pytest.raises(ValueError, match='pre-registered.*criterion'):
        runtime._reserve_batch([dict(id='first', sim_s=10)], 'train')
    assert runtime.spend == dict(train=0) and not runtime.seen_jobs
    assert (runtime.root/'CHARGES.jsonl').read_bytes() == b''
    with (runtime.root/'AUTHORITY.md').open('a', encoding='utf-8') as f:
        f.write('isolated-test criterion '+rc.criterion_document()['criterion_sha256']+'\n')
    runtime._reserve_batch([dict(id='first', sim_s=10)], 'train')
    assert runtime.spend == dict(train=10) and runtime.seen_jobs == {'first'}
    assert accel_storage._reserve.__globals__['CAP'] == 10

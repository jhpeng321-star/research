"""Restart identities and conservative old-ledger admission, without physics."""
from copy import deepcopy

import pytest

from . import launch_clean2 as clean2, rerun_cpus as rc, launch_production as production
from .protocol import digest
from .test_clean2_budget_cpu import args


def test_restart_resets_attempt_and_root_bindings_but_preserves_design():
    old = rc.runtime_config({}, root='/home/ubuntu/coopmwm20261105_clean1',
        reference_root='/home/ubuntu/legacy', source_root='/home/ubuntu/oldsource',
        common_checkpoint_sha256='a'*64, track2=True)
    old['plan_augment']['enabled'] = False
    old['launch_augmentation_decision'] = rc.AUGMENTATION_OFF_DECISION
    old.update(launch_optimization=deepcopy(production.OPTIMIZATION), gate_attempt=2,
        smoke_attempt=2, training_seed=20261101, worlds_by_update={'1':['registered-world']},
        augmentation_prelaunch=dict(ledger_root=old['run_root']))
    original = deepcopy(old)
    manifest = dict(commit='new-qualified-commit', files={})
    cfg = clean2.restart_config(old, clean2.ROOT, '/home/ubuntu/newsource', manifest)
    assert old == original
    assert rc.check_pins(cfg)
    assert cfg['run_root'] == cfg['accounting_root'] == cfg['bank_source_root'] == clean2.ROOT
    assert cfg['control_arm']['root'] == cfg['a_results']['root'] == cfg['single_ledger']['root'] == clean2.ROOT
    assert cfg['augmentation_prelaunch']['ledger_root'] == clean2.ROOT
    assert cfg['source_manifest_sha256'] == digest(manifest)
    assert cfg['gate_attempt'] == cfg['smoke_attempt'] == 1
    assert cfg['resume_update'] == cfg['A_resume_update'] == 0
    for key in ('budget','worlds_by_update','training_seed','launch_optimization',
                'isolated_test','diagnostic_schedule','final_evaluation','criterion_sha256'):
        assert cfg[key] == old[key]


def test_launch_admission_allows_only_budgeted_plan_fix_appends(args, tmp_path, monkeypatch):
    from . import clean2_budget as cb, storage
    value = cb.build(**args)
    value['clean1_ledger_sha256'] = digest(args['clean1_rows'])
    from .exact_grade import _seal
    value = _seal({k:v for k,v in value.items() if k != 'receipt_sha256'}, 'receipt_sha256')
    cfg = dict(budget=args['canonical_budget'], clean2_budget=value,
        clean2_restart=dict(clean1_root=str(tmp_path)))
    monkeypatch.setattr(clean2, 'stopped_clean1', lambda root: args['clean1_stop_sha256'])
    rows = deepcopy(args['clean1_rows'])
    monkeypatch.setattr(storage, 'ledger', lambda root: rows)
    clean2.verify_previous(cfg)
    rows.append(dict(id='more_plan', phase='plan_fix', sim_s=1000))
    clean2.verify_previous(cfg)
    rows.append(dict(id='unexpected_train', phase='v2v_train', sim_s=1200))
    with pytest.raises(ValueError, match='unexpected clean1'):
        clean2.verify_previous(cfg)
    rows.pop()
    rows[0]['sim_s'] += 1
    with pytest.raises(ValueError, match='prefix changed'):
        clean2.verify_previous(cfg)


def test_smoke_uses_exact_inherited_owner_safety_rules_and_never_changes_A(tmp_path):
    from . import composite_health as safety, launch_qualification as qualification
    from .exact_grade import _seal
    from .launch_training import write
    root=tmp_path/'run';root.mkdir();output=root/'smoke';output.mkdir()
    write(root/rc.RULE_FILE,dict(original_rule=True))
    amendment='a'*64
    (root/'AUTHORITY.md').write_text('amendmentSHA'+amendment)
    names=[]
    for arm,name,pause,authority in [('B',safety.B_RULE,safety.B_PAUSE,safety.B_AUTHORITY),
                                    ('R',safety.R_RULE,safety.R_PAUSE,safety.R_AUTHORITY)]:
        write(root/pause,dict(arm=arm,original_clean1_evidence=True))
        write(root/name,_seal(dict(authority=authority,arm=arm,composite_action='report_only',
            collision_teleport_action='pause_on_any',original_pause_sha256=rc.sha(root/pause),
            amendment_sha256=amendment),'receipt_sha256'))
        names += [name,pause]
    cfg=dict(run_root=str(root),clean2_budget={},clean2_restart=dict(
        inherited_safety_receipts={n:rc.sha(root/n) for n in names}))
    qualification.copy_safety_authority(cfg,output)
    assert safety.report_only(output,'B') is True
    assert safety.report_only(output,'R') is True
    assert safety.report_only(output,'A') is False
    assert all((output/n).read_bytes()==(root/n).read_bytes() for n in names)
    (root/safety.B_RULE).write_text('{}')
    with pytest.raises(ValueError,match='changed before smoke'):
        qualification.copy_safety_authority(cfg,output)


def test_prospective_quotes_create_artifacts_and_expand_all_full_native_grids(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from .test_track2_monitor_cpu import diagnostic_config
    cfg, backend = diagnostic_config(tmp_path)
    backend_adapter = SimpleNamespace(registration=cfg['isolated_registration'],
        resolve_states=backend.resolve_states,weights=backend.weights,
        reserve=backend.reserve,run_branch=backend.run_branch,close=lambda:None)
    monkeypatch.setattr(clean2,'QuoteBackend',lambda cfg,dry:backend_adapter)
    monkeypatch.setattr(production,'monitor_config',lambda cfg,root:dict(cfg,run_root=str(root)))
    quotes=clean2.full_diagnostic_quotes(cfg)
    assert set(quotes)=={'probe_u002','probe_u004','replicated_u006','replicated_u012','replicated_u024'}
    assert (tmp_path/'prospective_quotes').is_dir()
    assert not backend.calls
    for name,row in quotes.items():
        assert row['status']=='dry_run' and row['new_sim_s']==0
        assert row['cells']==(3 if name.startswith('probe') else 10)
        assert all(not b['window'].get('truncated') for b in row['branch_manifest'])

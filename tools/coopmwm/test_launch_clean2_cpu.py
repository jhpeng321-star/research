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

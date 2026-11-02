import json
from unittest.mock import patch

import pytest

from .ls_label_runtime import SmokeBranchRunner, sumo_environment, void_environment_failure, label_path


def test_environment_keeps_existing_paths_and_requires_real_executable(tmp_path):
    home=tmp_path/'sumo';(home/'bin').mkdir(parents=True)
    with pytest.raises(ValueError):sumo_environment(home,{'PATH':'/existing'})
    binary=home/'bin'/'sumo';binary.write_text('fixture');binary.chmod(0o755)
    env=sumo_environment(home,{'PATH':'/existing','KEPT':'yes'})
    assert env['SUMO_HOME']==str(home) and env['PATH'].startswith(str(home/'bin'))
    assert env['PATH'].endswith('/existing') and env['KEPT']=='yes'


def test_only_all_nine_zero_physics_environment_failures_can_retry(tmp_path):
    v=dict(status='failed',actual_sim_s=0,branches=[dict(status='failed',reason='SUMO binary not found on PATH') for _ in range(9)])
    assert void_environment_failure(dict(verification=v))
    for change in (dict(actual_sim_s=1),dict(status='accepted'),dict(branches=v['branches'][:8])):
        assert not void_environment_failure(dict(verification=dict(v,**change)))
    v['branches'][0]['reason']='prefix mismatch'
    assert not void_environment_failure(dict(verification=v))
    assert label_path(tmp_path,'query','SUMOENV1') != label_path(tmp_path,'query')


def test_real_policy_branch_is_first_and_reused_without_extra_execution(tmp_path):
    runner=SmokeBranchRunner(tmp_path,smoke_path=tmp_path/'SMOKE.json')
    specs=[dict(branch='policy',verification_id='test',branch_sha256='fixture'),dict(branch='cand1')]
    snap=dict(t0=300,prefixes={'300':{'state':'identical'}})
    result=dict(actual_sim_s=480,prefixes=snap['prefixes'])
    with patch('tools.coopmwm.sumo_verify.SubprocessBranchRunner.prepare') as prep, patch(
            'tools.coopmwm.sumo_verify.SubprocessBranchRunner.run',return_value=result) as run:
        with pytest.raises(ValueError):runner.run(specs[1],snap)
        assert run.call_count==0
        runner.prepare(specs,snap)
        assert run.call_args.args[0]['branch']=='policy'
        assert runner.run(specs[0],snap) is result and run.call_count==1
        runner.run(specs[1],snap)
        assert run.call_count==2 and prep.call_count==1
    assert json.loads((tmp_path/'SMOKE.json').read_text())['extra_sim_s']==0


def test_prefix_failure_never_opens_candidate_gate(tmp_path):
    runner=SmokeBranchRunner(tmp_path,smoke_path=tmp_path/'SMOKE.json')
    with patch('tools.coopmwm.sumo_verify.SubprocessBranchRunner.prepare'), patch(
            'tools.coopmwm.sumo_verify.SubprocessBranchRunner.run',return_value=dict(actual_sim_s=300,prefixes={})):
        with pytest.raises(Exception,match='fingerprint'):
            runner.prepare([dict(branch='policy')],dict(t0=300,prefixes={'300':{}}))
        with pytest.raises(ValueError):runner.run(dict(branch='cand1'),{})
    assert not (tmp_path/'SMOKE.json').exists()


def test_recovered_label_preserves_void_and_precharges_new_attempt(tmp_path):
    from .ls_collection import label
    (tmp_path/'labels').mkdir();(tmp_path/'admissions').mkdir()
    old=dict(verification=dict(status='failed',actual_sim_s=0,receipt_sha256='old',
        branches=[dict(status='failed',reason='SUMO binary not found on PATH') for _ in range(9)]))
    original=tmp_path/'labels'/'query.json';original.write_text(json.dumps(old));saved=original.read_bytes()
    (tmp_path/'admissions'/'query.json').write_text(json.dumps(dict(charged_sim_s=900)))
    envelope=dict(record=dict(state_id='query'),world_id=986016)
    config=dict(label_generation='SUMOENV1',run_root=str(tmp_path))
    def verify(snapshot,candidates,runner,charge,*args,**kwargs):
        charge('fixture',[dict(sim_s=100) for _ in range(9)])
        return dict(status='rejected')
    with patch('tools.coopmwm.ls_collection.complete_snapshot',return_value={}), patch(
            'tools.coopmwm.sumo_verify.verify',side_effect=verify), patch(
            'tools.coopmwm.ls_label_runtime.grading_disk_ready',return_value=True), patch(
            'tools.coopmwm.accel_storage.reserve') as reserve:
        label(tmp_path,envelope,dict(collection_top8=[{} for _ in range(8)]),config)
    assert reserve.call_count==1 and reserve.call_args.args[1][0]['sim_s']==900
    assert original.read_bytes()==saved
    assert json.loads((tmp_path/'label_voids'/'query.json').read_text())['charges_retained']
    assert label_path(tmp_path,'query','SUMOENV1').exists()


def test_low_disk_waits_before_claim_and_resumes_at_existing_guard(tmp_path):
    from types import SimpleNamespace
    from .ls_label_runtime import grading_disk_ready,MIN_FREE_BYTES
    with patch('tools.coopmwm.ls_label_runtime.shutil.disk_usage',return_value=SimpleNamespace(free=MIN_FREE_BYTES-1)):
        assert not grading_disk_ready(tmp_path,'SUMOENV1')
        receipt=tmp_path/'LABEL_DISK_WAIT_SUMOENV1.json';saved=receipt.read_bytes()
        assert not grading_disk_ready(tmp_path,'SUMOENV1') and receipt.read_bytes()==saved
        assert not json.loads(saved)['new_verification_claim']
    with patch('tools.coopmwm.ls_label_runtime.shutil.disk_usage',return_value=SimpleNamespace(free=MIN_FREE_BYTES)):
        assert grading_disk_ready(tmp_path,'SUMOENV1')

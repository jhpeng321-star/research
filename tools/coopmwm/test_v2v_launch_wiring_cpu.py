"""Actual supervisor dispatch must consume the reviewed launch3 gate."""
from pathlib import Path
from unittest.mock import patch

import pytest

from . import v2v_live as live
from . import storage
from . import replay_start, v2v_launch3


def authority(tmp_path):
    root=tmp_path/'new';root.mkdir()
    legacy=tmp_path/'old';legacy.mkdir()
    checkpoints={}
    from .protocol import sha
    for a in ('B','R'):
        p=legacy/f'{a}_checkpoint_020.pt';p.write_bytes(a.encode())
        checkpoints[a]=dict(path=str(p),update=20,sha256=sha(p))
    record=dict(binding_path=str(root/'binding.json'),binding_sha256='a'*64,
        control_root=str(root/'control'),mover_root=str(tmp_path/'mover'),
        legacy_boundary=dict(owners={a:dict(pid=i+1) for i,a in enumerate(('B','R'))},
            checkpoints=checkpoints))
    storage.write(root/'LAUNCH3_AUTHORITY.json',record)
    (root/'AUTHORITY.md').write_text(sha(root/'LAUNCH3_AUTHORITY.json'))
    return root,dict(legacy_root=str(legacy),gpu_deadline=10**12,v2v_intent=True),record


@pytest.mark.parametrize('stage,arm,cost',[
    (None,None,None),('smoke','B',{'v2v_train':28800}),
    ('train','R',{'v2v_train':662400}),('test8',None,{'v2v_interim':345600}),
    ('test24',None,{'v2v_final':345600})])
def test_registered_gate_and_exact_child_cost(tmp_path,stage,arm,cost):
    root,cfg,record=authority(tmp_path)
    with patch.object(replay_start,'alive',return_value=False), \
            patch.object(v2v_launch3,'wait_for_preflight',return_value={'ready':True}) as check:
        assert live.launch3_check(root,cfg,stage,arm)=={'ready':True}
    assert check.call_args.args==(record['binding_path'],)
    assert check.call_args.kwargs['binding_sha256']==record['binding_sha256']
    assert check.call_args.kwargs['phase_costs']==cost


@pytest.mark.parametrize('fault',['authority','live_owner','checkpoint'])
def test_boundary_or_authority_failure_never_reaches_preflight(tmp_path,fault):
    root,cfg,record=authority(tmp_path)
    if fault=='authority':(root/'AUTHORITY.md').write_text('')
    if fault=='checkpoint':Path(record['legacy_boundary']['checkpoints']['B']['path']).write_bytes(b'changed')
    with patch.object(replay_start,'alive',return_value=fault=='live_owner'), \
            patch.object(v2v_launch3,'wait_for_preflight') as check:
        with pytest.raises(ValueError):live.launch3_check(root,cfg,'smoke','B')
    check.assert_not_called()


def test_failed_gate_prevents_actual_popen(tmp_path):
    root=tmp_path/'new';root.mkdir()
    with patch.object(live,'configuration',return_value={'config':{'v2v_intent':True}}), \
            patch.object(live,'launch3_check',side_effect=ValueError('gate failed')) as check, \
            patch('subprocess.Popen') as popen:
        with pytest.raises(ValueError,match='gate failed'):live.spawn(root,'smoke','B')
    check.assert_called_once_with(root,{'v2v_intent':True},'smoke','B')
    popen.assert_not_called()


def test_hold_created_during_resource_wait_refuses_dispatch(tmp_path):
    root,cfg,record=authority(tmp_path)
    def wait(*args,**kwargs):
        (root/v2v_launch3.HOLD_NAME).write_text('hold')
        return {'ready':True}
    with patch.object(replay_start,'alive',return_value=False), \
            patch.object(v2v_launch3,'wait_for_preflight',side_effect=wait):
        with pytest.raises(v2v_launch3.Refused,match='HOLD'):live.launch3_check(root,cfg,'smoke','B')


def test_legacy_restarted_during_resource_wait_refuses_dispatch(tmp_path):
    root,cfg,record=authority(tmp_path)
    with patch.object(replay_start,'alive',side_effect=[False,False,True]), \
            patch.object(v2v_launch3,'wait_for_preflight',return_value={'ready':True}):
        with pytest.raises(ValueError,match='legacy arm still running'):live.launch3_check(root,cfg,'smoke','B')


def test_existing_dispatch_is_adopted_without_new_gate_or_popen(tmp_path):
    d=dict(pid=7);storage.write(tmp_path/'smoke_B_DISPATCH.json',d)
    with patch.object(replay_start,'alive',return_value=True), \
            patch.object(live,'launch3_check') as check,patch('subprocess.Popen') as popen:
        assert live.spawn(tmp_path,'smoke','B')==d
    check.assert_not_called();popen.assert_not_called()

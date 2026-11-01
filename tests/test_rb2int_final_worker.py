import json

import pytest

from tools.rb2int.final_worker import child_argv, completed_world, sha


def put(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj),encoding='utf-8')


def test_final_child_owns_one_world_but_keeps_complete_registered_k8(tmp_path):
    panel = tmp_path/'panel.json'
    put(panel,dict(worlds=[969550,969551]))
    release = dict(catalog={'FULL-v5':dict(identity_name='WM-PHYS-FULL-v5',binding=dict(
        variant='FULL',runtime_root='/frozen',config='/frozen/config.toml'))})
    argv = child_argv(tmp_path,tmp_path/'job',dict(model='FULL-v5',seed=969551),release,panel)
    assert argv[argv.index('--shard')+1] == '1'
    assert argv[argv.index('--shards')+1] == '2'
    assert '--max-tasks' not in argv and '--endpoint-only' not in argv
    assert argv[argv.index('--runtime-root')+1] == '/frozen'


def test_world_completion_requires_all_eight_immutable_task_files(tmp_path):
    out = tmp_path/'tasks'
    put(out/'IDENTITY.json',dict(model='test'))
    hashes = {}
    for k in range(8):
        p = out/f'w969550_k{k}.json'
        put(p,dict(seed=969550,future=k))
        hashes[p.name] = sha(p)
    put(out/'COMPLETE.json',dict(tasks=8,identity_sha256=sha(out/'IDENTITY.json'),hashes=hashes))
    assert completed_world(out,969550)
    put(out/'w969550_k7.json',dict(seed=969550,future=7,changed=True))
    with pytest.raises(ValueError,match='hash differs'):
        completed_world(out,969550)


def test_another_world_or_partial_ensemble_is_not_complete(tmp_path):
    out = tmp_path/'tasks'
    put(out/'COMPLETE.json',dict(tasks=7,identity_sha256='unused',hashes={}))
    with pytest.raises(ValueError,match='complete K8'):
        completed_world(out,969550)

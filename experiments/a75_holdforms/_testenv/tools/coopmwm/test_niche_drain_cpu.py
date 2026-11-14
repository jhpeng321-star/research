import json
from .ls_broker import drain_request
from .protocol import digest
from .ls_fast_worker import owner_key


def test_generation_drain_preserves_legacy_and_never_drains_other_owner(tmp_path):
    old=dict(boot_id='boot',pid=1,start_ticks='10')
    new=dict(boot_id='boot',pid=2,start_ticks='20')
    slot='gpuA@incident1'
    legacy=tmp_path/'gpu_drains'/(digest(slot)+'.json')
    legacy.parent.mkdir();legacy.write_text(json.dumps(dict(owner=old)))
    assert drain_request(tmp_path,slot,new) is None
    key=digest(dict(slot=slot,owner=owner_key(new)))
    target=tmp_path/'gpu_drains_v2'/(key+'.json')
    target.parent.mkdir();target.write_text(json.dumps(dict(owner=new)))
    assert drain_request(tmp_path,slot,new)==tmp_path/'gpu_drained_v2'/target.name
    assert drain_request(tmp_path,slot,old)==tmp_path/'gpu_drained'/legacy.name
    assert json.loads(legacy.read_text())['owner']==old

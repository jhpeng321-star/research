"""Read-only S-C2H60B handoff checks; never train, collect, or signal a process."""
import hashlib
import json

from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.c2h60b import FILES
from tools.loopresume.operate import ROOT, remote


def main():
    folder = ROOT / 'tools/loopresume/receipts_c2h60b'
    docs = {name: json.loads((folder / (name + '.json')).read_text()) for name in (
        'SOURCE', 'TESTS', 'RECOVERY', 'BINDINGS', 'LAUNCH', 'VERIFIED')}
    for name in FILES:
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == docs['SOURCE'][name], name
    assert docs['TESTS']['exit'] == 0
    assert docs['RECOVERY']['saved_update'] == 241
    assert docs['RECOVERY']['best_update'] == 240
    verified = docs['VERIFIED']
    assert verified['new_updates'] >= 15 and verified['memory_watch_s'] >= 1200
    assert verified['charged'] == 176473 and verified['finite_losses_and_gradients']
    assert docs['BINDINGS']['amendment']['gpu_cap'] == 0.92
    hashes = {name: hashlib.sha256((folder / (name + '.json')).read_bytes()).hexdigest() for name in docs}
    remote('gpuA', PRELUDE + f'''
from tools.loopresume.c2h60_box import verify_updates
for name,sha in {hashes!r}.items():
 assert loop.sha256(r/('C2H60B_'+name+'.json'))==sha,name
binding=loop.read_json(r/'C2H60B_BINDINGS.json')
for name,sha in binding['source'].items(): assert loop.sha256(root/name)==sha,name
recovery=loop.read_json(r/'C2H60B_RECOVERY.json')
for key in ('summary','checkpoint','config','gate_panel','best_checkpoint','cause_summary'):
 assert loop.sha256(recovery[key])==recovery[key+'_sha256'],key
cfg=loop.load_config(a/'CONFIG.json'); state=loop.load_state(cfg); reg=loop.check_registration(cfg)
assert state['sim_s_charged']==176473
assert cfg['policy']['env']['C4_GPU_MEM_FRACTION']=='0.92'
assert cfg['policy']['env']['PYTORCH_CUDA_ALLOC_CONF']=='expandable_segments:True'
doc=loop.read_json(r/'C2H60B_PROGRESS.json')
if doc['status']=='POLICY_RUNNING':
 assert previous.live(loop.read_json(r/'C2H60B_LAUNCH.json')) and previous.live(doc['child'])
 out=pathlib.Path(doc['output'])
 rows=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
 verify_updates(rows,241,len(rows))
 assert len(rows)>=15
 assert all(state['cycles'][2]['phases'][name]['status']=='pending' for name in ('collect','convert','wm_ft','check'))
else:
 assert doc['status']=='WAIT_OWNER_BUDGET' and state['stopped']=='WAIT_OWNER_BUDGET'
print(json.dumps({{'verification':'PASS','status':doc['status'],'charged':state['sim_s_charged'],
 'cap':reg['budget']['sim_s_total'],'dispatch':False}}))
''')


if __name__ == '__main__':
    main()

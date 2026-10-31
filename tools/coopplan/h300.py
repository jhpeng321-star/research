"""MAIN02:21 prospective H300 continuation; never mutates completed roots."""
import argparse
import base64
import json
from pathlib import Path
from unittest.mock import patch

from tools.coopbandit.ops import remote
from . import ops, extension
from .protocol import h300_worlds, h300_budget, digest, sha, checked, REGISTRATION, approval

LOCAL = ops.LOCAL
EVIDENCE = LOCAL/'docs/paper_results/coopplan/h300'
DESIGN = LOCAL/'docs/paper_results/COOPPLAN_H300_DESIGN.md'
HOSTS = {h:dict(c,root=c['root'].replace('_v1','_v3')) for h,c in ops.HOSTS.items()}
WORLD_CAP = 9780


def save(name,value):
    with patch.object(extension,'EVIDENCE',EVIDENCE):
        return extension.save(name,value)


def allocation(host):
    panel = [w for i,w in enumerate(h300_worlds()) if i%3 == tuple(HOSTS).index(host)]
    return dict(worlds=panel,cap=h300_budget()[host],
                reserved={f's{w["seed"]}_{a}':1200 for w in panel for a in ('mappo','planner')},
                world_caps={f's{w["seed"]}':WORLD_CAP for w in panel})


def manifest():
    return dict(prior_actual_and_charged=565140,continuation_cap=sum(h300_budget().values()),
                global_upper=799860,unused=140,world_cap=WORLD_CAP,
                mandatory_pairs=24*2400,adaptive_predictions_and_all_solos=24*(WORLD_CAP-2400),
                panel_sha256=digest(h300_worlds()),design_sha256=sha(DESIGN),
                host_caps=h300_budget(),horizon_s=300,new_sim_s=0)


def free_snapshot():
    # The peer handoff explicitly supersedes the legacy MAPPO-credit exports.
    directory=Path('E:/agent-work/c4-coopfree/tools/coopfree/publications/hold')
    path=directory/'MANIFEST.json'
    if not path.exists():
        return dict(available=False,reason='corrected HOLD publication absent')
    m=json.loads(checked(path).read_text(encoding='utf-8'))
    for checkpoint in sorted(directory.glob('policy_u*.pt'),reverse=True):
        update=int(checkpoint.stem.split('_u')[1]);receipt=directory/f'UPDATE_{update:03d}.json'
        if update<1 or not receipt.exists():continue
        r=json.loads(checked(receipt).read_text(encoding='utf-8'))
        actual=sha(checkpoint)
        assert r['checkpoint_sha256']==actual==m['files'][checkpoint.name]
        assert m['files'][receipt.name]==sha(receipt)
        binding_path=directory/'BINDING.json'
        assert m['files']['BINDING.json']==sha(binding_path)
        binding=json.loads(binding_path.read_text(encoding='utf-8'))
        for name in ('policy.py','protocol.py'):
            assert sha(Path('E:/agent-work/c4-coopfree/tools/coopfree')/name)==binding['sources']['tools/coopfree/'+name]
        return dict(available=True,checkpoint=str(checkpoint),sha256=actual,update=update,
                    credit_reference='HOLD',manifest_sha256=sha(path),receipt_sha256=sha(receipt))
    return dict(available=False,reason='corrected HOLD learner has no complete learned update',manifest_sha256=sha(path))


def probe(host):
    with patch.object(extension,'HOSTS',HOSTS):return extension.probe(host)


def audit(host):
    with patch.object(ops,'worlds',h300_worlds):result=ops.audit(host)
    return save('AUDIT_'+host+'.json',result)


def stage(host):
    with patch.object(extension,'HOSTS',HOSTS),patch.object(extension,'DESIGN',DESIGN):
        return extension.stage(host)


def prepare(host):
    c=HOSTS[host];a=allocation(host)
    text=checked(REGISTRATION).read_text(encoding='utf-8');owner=approval(text)
    main=next(s for s in text.splitlines() if '02:21 OWNER' in s and 'H=300' in s)
    registered=next(s for s in text.splitlines() if 'S-COOPPLAN PRECOMPUTE DESIGN REGISTERED H300' in s)
    assert all(json.loads((EVIDENCE/f'AUDIT_{h}.json').read_text())['pass_'] for h in HOSTS)
    free=json.loads((EVIDENCE/'FREE_FREEZE.json').read_text())
    data=base64.b64encode(checked(free['checkpoint']).read_bytes()).decode() if free['available'] else None
    old=ops.HOSTS[host]['root']
    return remote(host,f'''import pathlib,sys,os,json,base64
root=pathlib.Path({c['root']!r});assert root.is_dir();assert not (root/'DISPATCH_ONCE.json').exists()
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools')]+{c['deps']!r}
from tools.coopplan.storage import read,write
from tools.coopplan.protocol import sha,h300_budget,h300_worlds,guard
free=json.loads({json.dumps(free)!r});artifacts={{}}
if free['available']:
 p=root/'FREE.pt';data=base64.b64decode({data!r})
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)
 assert sha(p)==free['sha256'];artifacts['FREE.pt']=sha(p)
files=json.loads({json.dumps(list(ops.sources()))!r})
binding=dict(phase='h300',host={host!r},runtime={c['runtime']!r},checkpoint={c['checkpoint']!r},deps={c['deps']!r},
 design_sha256=sha(root/'COOPPLAN_DESIGN.md'),sources={{n:sha(root/'pytools'/n) for n in files}},
 budget=h300_budget(),workers={c['workers']},load_max={c['load_max']},all_credits=True,horizon_s=300,
 free_checkpoint=str(root/'FREE.pt') if free['available'] else None,free_provenance=free,artifacts=artifacts)
allocation=json.loads({json.dumps(a)!r})
for name,value in [('BINDING.json',binding),('ALLOCATION.json',allocation),
 ('APPROVAL.json',dict(owner_line={owner!r},main_line={main!r},registration_line={registered!r},design_sha256=binding['design_sha256']))]:
 if (root/name).exists():assert read(root/name)==value
 else:write(root/name,value)
for name in ('jobs','states'):(root/name).mkdir(exist_ok=True)
reference=read(pathlib.Path({old!r})/'MECHANICS.json');assert reference['pass_']
if (root/'MECHANICS.json').exists():assert read(root/'MECHANICS.json')==reference
else:write(root/'MECHANICS.json',reference)
from tools.coopplan.plant import setup
setup(root)
for world in h300_worlds():guard(world)
import sumo,subprocess
binary=pathlib.Path(sumo.__file__).parent/'bin/sumo'
assert '1.27.1' in subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True).stdout
print(json.dumps(dict(host={host!r},guards=24,cap=allocation['cap'],design_sha256=binding['design_sha256'],
 source_sha256s=binding['sources'],free=free,new_sim_s=0)))
''')


def cpu(host):
    c=HOSTS[host]
    return remote(host,f'''import pathlib,sys,os,unittest,json
from unittest.mock import patch
root=pathlib.Path({c['root']!r});os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools')]+{c['deps']!r}
from tools.coopplan.plant import setup
setup(root)
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in
 ('tools.coopplan.test_cpu','tools.coopplan.test_extension','tools.coopplan.test_h300'))
with patch('subprocess.Popen',side_effect=AssertionError('SUMO disabled')):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
from tools.coopplan.predictor import SumoOraclePredictor
from tools.coopplan.planner import wmdecide_check,decide
from tools.coopplan.test_cpu import STATE,ACTIONS,toy
p=SumoOraclePredictor(lambda s,a,h:toy(a)).predict(STATE,ACTIONS,300,1)
assert wmdecide_check(p)['selected']==decide(p)['selected']
receipt=dict(host={host!r},pass_=result.wasSuccessful(),tests=result.testsRun,skipped=len(result.skipped),wmdecide_parity=True,sim_s=0)
assert receipt['pass_'] and not receipt['skipped']
from tools.coopplan.storage import write
write(root/'CPU.json',receipt)
print(json.dumps(receipt))
''')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('manifest','probe','audit','freeze','stage','prepare','cpu','launch','status'))
    p.add_argument('--host',choices=tuple(HOSTS),default='cpuQ');a=p.parse_args()
    if a.mode=='manifest':result=manifest()
    elif a.mode=='freeze':
        result=save('FREE_FREEZE.json',free_snapshot());save('MANIFEST.json',manifest())
    elif a.mode in ('launch','status'):
        if a.mode=='launch':
            for host in HOSTS:
                assert json.loads((EVIDENCE/f'CPU_{host}.json').read_text())['pass_']
        with patch.object(ops,'HOSTS',HOSTS):result=getattr(ops,a.mode)(a.host)
        if a.mode=='launch':save('DISPATCH_'+a.host+'.json',json.loads(result))
    else:
        result=globals()[a.mode](a.host)
        if a.mode in ('prepare','cpu'):save(a.mode.upper()+'_'+a.host+'.json',json.loads(result))
    print(json.dumps(result) if not isinstance(result,str) else result)


if __name__=='__main__':main()

"""01:28 approved extension: separate immutable roots, allocations, and public receipts."""
import argparse
import base64
import json
from pathlib import Path

from tools.coopbandit.ops import remote
from . import ops
from .protocol import extension_worlds, extension_budget, approval, sha, digest, checked, REGISTRATION

LOCAL = ops.LOCAL
EVIDENCE = LOCAL/'docs/paper_results/coopplan/extension'
DESIGN = LOCAL/'docs/paper_results/COOPPLAN_EXTENSION_DESIGN.md'
HOSTS = {h:dict(c,root=c['root'].replace('_v1','_v2')) for h,c in ops.HOSTS.items()}


def first_snapshot():
    return json.loads((LOCAL/'docs/paper_results/coopplan/final/RECEIPTS.json').read_text(encoding='utf-8'))


def missing(host):
    h = next(h for h in first_snapshot() if h['host'] == host)
    result = []
    for r in h['records']:
        sampled = {c['time_s'] for c in r['credits']}
        for e in r['history']:
            if e['group'] and e['time_s'] not in sampled:
                result.append(dict(world=r['world'],event=e))
    return result


def allocation(host):
    panel = [w for i,w in enumerate(extension_worlds()) if i%3 == tuple(HOSTS).index(host)]
    reserved = {f's{w["seed"]}_{arm}':1200 for w in panel for arm in ('mappo','planner')}
    for r in missing(host):
        e = r['event'];key=f's{r["world"]["seed"]}_t{e["time_s"]}_CG'
        reserved[key] = 2*(e['time_s']+120)
    return dict(worlds=panel,cap=extension_budget()[host],reserved=reserved,backfill=missing(host))


def manifest(host=None):
    original = sum(h['charged'] for h in first_snapshot())
    assert original == 220980
    backfill = sum(2*(r['event']['time_s']+120) for h in HOSTS for r in missing(h))
    return dict(original_charged=original,extension_cap=sum(extension_budget().values()),
                global_cap=original+sum(extension_budget().values()),paired_episodes=57600,
                original_missing_credits=sum(len(missing(h)) for h in HOSTS),original_solo_reserve=backfill,
                adaptive_predictions_and_all_solos=579020-57600-backfill,
                host_allocations={h:dict(cap=allocation(h)['cap'],mandatory=sum(allocation(h)['reserved'].values())) for h in HOSTS},
                panel_sha256=digest(extension_worlds()),design_sha256=sha(DESIGN),new_sim_s=0)


def save(name,value):
    assert EVIDENCE.parent.is_dir()
    EVIDENCE.mkdir(exist_ok=True)
    path = EVIDENCE/name
    if path.exists():
        assert json.loads(path.read_text(encoding='utf-8')) == value
    else:
        from .storage import write
        write(path,value)
    return value


def probe(host):
    """Read only exact public roots before admitting new source or processes."""
    c = HOSTS[host]
    return json.loads(remote(host,f'''import pathlib,json,os
root=pathlib.Path({c['root']!r})
names=('BINDING.json','DISPATCH_ONCE.json','COMPLETE.json','STOP.json')
print(json.dumps(dict(host={host!r},extension_root_exists=root.exists(),
 receipts={{n:json.loads((root/n).read_text()) for n in names if (root/n).exists()}},
 load=os.getloadavg())))
'''))


def free_snapshot():
    """Verify a completed TRAIN update only; no peer job creation or intervention."""
    p = Path('E:/agent-work/c4-coopfree/tools/coopfree/publications/MANIFEST.json')
    if not p.exists():
        return dict(available=False,reason='no published learned checkpoint')
    m = json.loads(checked(p).read_text(encoding='utf-8'))
    checkpoints = sorted(p.parent.glob('policy_u*.pt'))
    for checkpoint in reversed(checkpoints):
        update = int(checkpoint.stem.split('_u')[1])
        receipt = p.parent/f'UPDATE_{update:03d}.json'
        if not receipt.exists() or update < 1:
            continue
        r = json.loads(receipt.read_text(encoding='utf-8'))
        actual = sha(checkpoint)
        assert r['checkpoint_sha256'] == actual
        assert m['files'][checkpoint.name] == actual
        assert m['files'][receipt.name] == sha(receipt)
        binding_path=p.parent/'BINDING.json'
        assert m['files']['BINDING.json'] == sha(binding_path)
        binding=json.loads(binding_path.read_text(encoding='utf-8'))
        for name in ('policy.py','protocol.py'):
            source=Path('E:/agent-work/c4-coopfree/tools/coopfree')/name
            assert sha(source) == binding['sources']['tools/coopfree/'+name]
        return dict(available=True,checkpoint=str(checkpoint),sha256=actual,update=update,
                    receipt_sha256=sha(receipt),manifest_sha256=sha(p))
    return dict(available=False,reason='publisher has no complete learned update',manifest_sha256=sha(p))


def audit(host):
    from unittest.mock import patch
    with patch.object(ops,'worlds',extension_worlds):
        result = ops.audit(host)
    return save('AUDIT_'+host+'.json',result)


def stage(host):
    c = HOSTS[host]
    files = ops.sources()
    design = DESIGN.read_text(encoding='utf-8')
    assert len(design.splitlines()) <= 70
    return remote(host,f'''import pathlib,json
root=pathlib.Path({c['root']!r});assert root.parent.is_dir();root.mkdir(exist_ok=True)
assert not (root/'DISPATCH_ONCE.json').exists(),'never restage a dispatched source'
files=json.loads({json.dumps(files)!r})
for name,text in files.items():
 p=root/'pytools'/name;p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_text()==text,('immutable source',name)
 else:p.write_text(text)
p=root/'COOPPLAN_DESIGN.md'
if p.exists():assert p.read_text()=={design!r}
else:p.write_text({design!r})
print(json.dumps(dict(host={host!r},root=str(root),files=len(files),new_sim_s=0)))
''')


def prepare(host):
    c = HOSTS[host]
    text = checked(REGISTRATION).read_text(encoding='utf-8')
    owner = approval(text)
    main = next(s for s in text.splitlines() if '01:28 MAIN REVIEW' in s and 'EXTEND S-COOPPLAN' in s)
    registered = next(s for s in text.splitlines() if 'S-COOPPLAN PRECOMPUTE DESIGN REGISTERED EXTENSION' in s)
    a = allocation(host)
    audits = [json.loads((EVIDENCE/f'AUDIT_{h}.json').read_text()) for h in HOSTS]
    assert all(r['pass_'] and not r['hits'] for r in audits)
    # One shared snapshot, frozen before ANY host dispatch, so model availability is not outcome-dependent.
    free = json.loads((EVIDENCE/'FREE_FREEZE.json').read_text())
    checkpoint = base64.b64encode(checked(free['checkpoint']).read_bytes()).decode() if free['available'] else None
    oldroot = ops.HOSTS[host]['root']
    return remote(host,f'''import pathlib,sys,os,json,base64,shutil
root=pathlib.Path({c['root']!r});assert root.is_dir();assert not (root/'DISPATCH_ONCE.json').exists()
os.nice(19);os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools')]+{c['deps']!r}
from tools.coopplan.storage import read,write
from tools.coopplan.protocol import sha,extension_budget,guard,extension_worlds
free=json.loads({json.dumps(free)!r});artifacts={{}}
if free['available']:
 p=root/'FREE.pt';data=base64.b64decode({checkpoint!r})
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)
 assert sha(p)==free['sha256'];artifacts['FREE.pt']=free['sha256']
files=json.loads({json.dumps(list(ops.sources()))!r})
binding=dict(phase='extension',host={host!r},runtime={c['runtime']!r},checkpoint={c['checkpoint']!r},deps={c['deps']!r},
 design_sha256=sha(root/'COOPPLAN_DESIGN.md'),sources={{n:sha(root/'pytools'/n) for n in files}},
 budget=extension_budget(),workers={c['workers']},load_max={c['load_max']},all_credits=True,
 free_checkpoint=str(root/'FREE.pt') if free['available'] else None,free_provenance=free,artifacts=artifacts)
allocation=json.loads({json.dumps(a)!r})
for name,value in [('BINDING.json',binding),('ALLOCATION.json',allocation),
 ('APPROVAL.json',dict(owner_line={owner!r},main_line={main!r},registration_line={registered!r},design_sha256=binding['design_sha256']))]:
 if (root/name).exists():assert read(root/name)==value
 else:write(root/name,value)
for name in ('jobs','states','original'):(root/name).mkdir(exist_ok=True)
old=pathlib.Path({oldroot!r});assert (old/'COMPLETE.json').exists() and not (old/'STOP.json').exists()
for r in allocation['backfill']:
 e=r['event'];key=f's{{r["world"]["seed"]}}_t{{e["time_s"]}}'
 src=old/'states'/(key+'.json');dst=root/'states'/src.name
 if dst.exists():assert dst.read_bytes()==src.read_bytes()
 else:shutil.copyfile(src,dst)
 for action in ('O0',e['executed_action']['id']):
  d=root/'original'/(key+'_'+action);d.mkdir(exist_ok=True)
  for name in ('RESULT.json','observable.jsonl.gz'):
   src=old/'jobs'/(key+'_'+action)/name;dst=d/name
   if dst.exists():assert dst.read_bytes()==src.read_bytes()
   else:shutil.copyfile(src,dst)
reference=read(old/'MECHANICS.json')
if (root/'MECHANICS.json').exists():assert read(root/'MECHANICS.json')==reference
else:write(root/'MECHANICS.json',reference)
from tools.coopplan.plant import setup
setup(root)
for world in extension_worlds():guard(world)
import sumo,subprocess
binary=pathlib.Path(sumo.__file__).parent/'bin/sumo'
assert '1.27.1' in subprocess.run([str(binary),'--version'],capture_output=True,text=True,check=True).stdout
print(json.dumps(dict(host={host!r},guards=24,cap=allocation['cap'],mandatory=sum(allocation['reserved'].values()),
 design_sha256=binding['design_sha256'],source_sha256s=binding['sources'],free=free,new_sim_s=0)))
''')


def cpu(host):
    c = HOSTS[host]
    return remote(host,f'''import pathlib,sys,os,unittest,json
from unittest.mock import patch
root=pathlib.Path({c['root']!r});os.nice(19)
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
sys.path[:0]=[str(root/'pytools')]+{c['deps']!r}
from tools.coopplan.plant import setup
setup(root)
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in ('tools.coopplan.test_cpu','tools.coopplan.test_extension'))
with patch('subprocess.Popen',side_effect=AssertionError('SUMO disabled')):
 result=unittest.TextTestRunner(verbosity=2).run(suite)
from tools.coopplan.predictor import SumoOraclePredictor
from tools.coopplan.planner import wmdecide_check,decide
from tools.coopplan.test_cpu import STATE,ACTIONS,toy
p=SumoOraclePredictor(lambda s,a,h:toy(a)).predict(STATE,ACTIONS,120,1)
assert wmdecide_check(p)['selected']==decide(p)['selected']
receipt=dict(host={host!r},pass_=result.wasSuccessful(),tests=result.testsRun,skipped=len(result.skipped),wmdecide_parity=True,new_sim_s=0)
assert receipt['pass_'] and not receipt['skipped']
from tools.coopplan.storage import write
write(root/'CPU.json',dict(receipt,sim_s=0))
print(json.dumps(receipt))
''')


def backfill(root):
    """Add missing original-panel twins only; never re-execute original live episodes."""
    from copy import deepcopy
    from . import plant
    from .storage import read,write
    from .predictor import Prediction,Outcome
    from .planner import cooperation_credit
    plant.setup(root)
    credits = []
    for r in read(root/'ALLOCATION.json')['backfill']:
        e = r['event'];key=f's{r["world"]["seed"]}_t{e["time_s"]}'
        chosen = e['executed_action'];solos=[];outcomes={}
        for name in ('O0',chosen['id']):
            outcomes[name] = read(root/'original'/(key+'_'+name)/'RESULT.json')['outcome']
        for i,member in enumerate(chosen['members']):
            a = deepcopy(chosen);a.update(id=chosen['id']+f'_solo{i}',members=[member])
            solos.append(a['id']);outcomes[a['id']] = plant.branch(root,key,a,120)
        prediction = Prediction({a:[Outcome('exact',**o)] for a,o in outcomes.items()},True,'SUMO-ORACLE')
        cg = cooperation_credit(prediction,chosen['id'],solos)['exact']
        j0=outcomes['O0']['team_delay']
        credits.append(dict(seed=r['world']['seed'],time_s=e['time_s'],action=chosen,members=e['group'],
                            C_G=cg,D_G=j0-outcomes[chosen['id']]['team_delay'],
                            member_alone_D=[j0-outcomes[s]['team_delay'] for s in solos],
                            horizon_s=120,semantics='all_executed_groups_backfill'))
    write(root/'BACKFILL.json',credits)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('manifest','probe','audit','freeze','stage','prepare','cpu','launch','status'))
    p.add_argument('--host',choices=tuple(HOSTS),default='cpuQ');a=p.parse_args()
    if a.mode == 'freeze':
        result=save('FREE_FREEZE.json',free_snapshot());save('MANIFEST.json',manifest())
    elif a.mode in ('launch','status'):
        from unittest.mock import patch
        with patch.object(ops,'HOSTS',HOSTS):
            result=getattr(ops,a.mode)(a.host)
        if a.mode == 'launch':save('DISPATCH_'+a.host+'.json',json.loads(result))
    else:
        result=globals()[a.mode](a.host)
        if a.mode in ('prepare','cpu'):save(a.mode.upper()+'_'+a.host+'.json',json.loads(result))
    print(json.dumps(result) if not isinstance(result,str) else result)


if __name__ == '__main__':
    main()

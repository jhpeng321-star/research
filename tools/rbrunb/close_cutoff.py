"""Verify and record the stopped partial B2 panel; never dispatch or score."""
from datetime import datetime
import json
import time

import watch


def main():
    assert time.time() >= datetime.fromisoformat('2026-11-01T11:30:00+08:00').timestamp()
    claim = json.loads((watch.SHARED/'WM-PHYS-B2.claim').read_text())
    assert claim['owner'] == watch.OWNER
    result = json.loads(watch.ssh(r'''
from pathlib import Path
import hashlib,json,time
r=Path('/root/rbeval20261031/runb/WM-PHYS-B2')
def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
failed=load(r/'FAILED.json')
assert failed['owned_pids']==[801887,801888,801889]
assert 'Owned workers exited [1, 0, 0]' in failed['error']
assert all(not (Path('/proc')/str(pid)).exists() for pid in failed['owned_pids'])
assert not Path('/proc/762343').exists()
assert not (r/'FINALIZED.json').exists() and not (r/'reports').exists()
owned=load(next(r.glob('SUPERVISOR_762343_*.json')))
assert owned['owner']=='S-RB-RUNB' and owned['deadline_unix']==1793503800.0
assert owned['manifest']==MANIFEST
assert all(sha(Path(p))==h for p,h in owned['sources'].items())
hashes={}; futures={}; counts={}; complete_shards=[]; stop_times={}
for i in range(3):
 d=r/('shard'+str(i)); identity=sha(d/'IDENTITY.json')
 assert (d/'STOP').is_file()
 stop_times[d.name]=(d/'STOP').stat().st_mtime
 expected={'w%s_k%s.json'%(s,k) for j,s in enumerate(range(969420,969440)) if j%3==i for k in range(8)}
 files=list(d.glob('w*_k*.json')); assert {p.name for p in files}<=expected
 counts[d.name]=len(files)
 for p in files:
  t=load(p); assert t['identity_sha256']==identity
  assert p.name=='w%s_k%s.json'%(t['seed'],t['future'])
  hashes[d.name+'/'+p.name]=sha(p)
  futures.setdefault(str(t['seed']),[]).append(t['future'])
 if (d/'COMPLETE.json').exists():
  c=load(d/'COMPLETE.json'); assert set(c['hashes'])==expected
  assert c['tasks']==len(expected)==len(files) and c['identity_sha256']==identity
  assert all(sha(d/n)==h for n,h in c['hashes'].items())
  complete_shards.append(d.name)
assert counts==dict(shard0=53,shard1=56,shard2=48)
assert complete_shards==['shard1','shard2']
complete_worlds=sorted(int(s) for s,ks in futures.items() if sorted(ks)==list(range(8)))
incomplete={s:sorted(ks) for s,ks in futures.items() if sorted(ks)!=list(range(8))}
assert len(complete_worlds)==19 and incomplete=={'969429':[0,1,2,3,4]}
note=dict(owner='S-RB-RUNB',variant='WM-PHYS-B2',state='OWNER_DEADLINE_PARTIAL_NO_RETRY',
 remote_output=str(r),tasks=len(hashes),expected_tasks=160,complete_worlds=complete_worlds,
 incomplete_worlds=incomplete,shards=counts,complete_shards=complete_shards,
 task_hashes=hashes,stop_unix=stop_times,failed=failed,all_owned_gpu_processes_exited=True,
 sources_unchanged=True,automatic_retry=False,scored=False,done=False,verified_unix=time.time())
with (r/'CUTOFF_VERIFIED_1130.json').open('x') as f: json.dump(note,f,indent=2)
print(json.dumps(note))
'''.replace('MANIFEST', repr(claim['manifest']))))
    target = watch.WORK/'runs'/'rbrunb'/'B2_CUTOFF_VERIFIED_1130.json'
    watch.save_new(target, result)
    watch.save_new(watch.SHARED/'WM-PHYS-B2.partial.json', result)
    failure = watch.SHARED/'WM-PHYS-B2.failed.json'
    if not failure.exists():
        watch.save_new(failure, dict(owner=watch.OWNER, variant='WM-PHYS-B2',
                                    remote_output=result['remote_output'], result=result['failed'],
                                    tasks=result['tasks'], automatic_retry=False,
                                    reason='Owner11:30 cooperative deadline; completed tasks preserved'))
    assert not (watch.SHARED/'WM-PHYS-B2.done').exists()
    print(json.dumps({k:v for k,v in result.items() if k!='task_hashes'}))


if __name__ == '__main__':
    main()

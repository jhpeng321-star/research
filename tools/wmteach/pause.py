"""Owner-requested cooperative stop and read-only, hash-bound resume inventory.

This does not launch or resume anything. Immutable worker code already checks
STOP between bounded tasks; its resulting FAILURE is an intentional owner stop.
"""
import json

from tools.wmteach.ops import ROOT, remote

HERE = ROOT / 'docs/paper_results/wmteach'
HOLD = HERE / 'OWNER1735_PAUSE.json'
JOBS = (
    ('v4scale', '/root/autodl-tmp/wmteach_scale20261101_v3', 453587, '889174202'),
    ('v5scale', '/root/autodl-tmp/wmteach_scale_v5_20261101_v2', 453608, '889174685'),
    ('v5quality', '/root/autodl-tmp/wmteach_v5quality20261101_v2', 453566, '889173543'),
)
BOOT = 'a81555d7-176e-4ed8-bc7d-896fc89cc914'


def inspect(request=False):
    assert HOLD.is_file(), 'durable local owner hold required'
    source = '''import pathlib,json,time,hashlib,os,subprocess
jobs=JOBS_VALUE
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
out=[]
for name,location,pid,ticks in jobs:
 root=pathlib.Path(location)
 assert root.is_dir() and root.parent==pathlib.Path('/root/autodl-tmp')
 owner=json.loads((root/'DISPATCH.json').read_text())
 assert owner['pid']==pid and owner['start_ticks']==ticks and owner['boot_id']==BOOT_VALUE
 proc=pathlib.Path(f'/proc/{pid}')
 st=(proc/'stat').read_text().split() if (proc/'stat').is_file() else None
 matches=bool(st and st[21]==ticks and boot==owner['boot_id'])
 active=matches and st[2]!='Z'
 cmd=(proc/'cmdline').read_bytes().replace(b'\\0',b' ').decode() if matches else None
 cwd=str((proc/'cwd').resolve()) if active else None
 if active:
  assert cwd==str(root) and str(root) in cmd and 'tools.wmteach.' in cmd
 if REQUEST_VALUE and active:
  marker=root/'OWNER1735_PAUSE.json'
  if not marker.exists():
   with marker.open('x') as f:json.dump(dict(unix=time.time(),owner=owner,
    authority='MAIN 2026-11-01 17:35',resume_requires='S-WMTEACH RESUME',
    checkpoint='existing STOP guard at next bounded task boundary',retain_all=True),f)
  if not (root/'STOP').exists():
   with (root/'STOP').open('x') as f:f.write('MAIN1735 PAUSE; retain all work; resume only after S-WMTEACH RESUME\\n')
 files={}
 for item in ('IDENTITY.json','DISPATCH.json','OWNER1735_PAUSE.json','QUALITY.json',
  'QUALITY_RECORDS.json','QUALITY_COMPLETE.json','QUALITY_PROVENANCE.json',
  'HELPFUL_QUALITY.json','CORPUS.json','COMPLETE.json','FAILURE.json'):
  p=root/item
  if p.is_file():files[item]=json.loads(p.read_text())
 states=sorted((root/'states').glob('*/*_t*.json'))
 labels120=sorted((root/'labels').glob('*/LABEL_H120.json'))
 labels300=sorted((root/'labels').glob('*/LABEL_H300.json'))
 futures=sorted((root/'labels').glob('*/FUTURES.json'))
 # Hash all completed cache outputs, excluding mutable logs/locks and source data.
 cached=states+sorted((root/'states').glob('*/COMPLETE.json'))+futures+labels120+labels300
 quality_tasks=sorted((root/'tasks').glob('*/*.json'))
 cached+=quality_tasks
 manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,
  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in cached}
 done120={p.parent.name for p in labels120};done300={p.parent.name for p in labels300}
 done=done120 & done300
 record=dict(name=name,root=str(root),owner=owner,process=dict(identity_matches=matches,
  active=active,state=st[2] if matches else None,command=cmd,cwd=cwd),
  stop_present=(root/'STOP').exists(),files=files,generated_states=len(states),
  complete_lineages=len(list((root/'states').glob('*/COMPLETE.json'))),
  complete_label_pairs=len(done),h120_labels=len(labels120),h300_labels=len(labels300),
  future_caches=len(futures),completed_state_ids=sorted(done),
  quality_futures=sum(p.name.startswith('k') for p in quality_tasks),
  quality_labels=sum(p.name.startswith('LABEL_H') for p in quality_tasks),
  next_unlabelled_state=next((p.stem for p in states if p.stem not in done),None),
  partial_lineages=[dict(name=p.name,states=len(list(p.glob('*_t*.json'))),
   last_saved_time_s=max(int(q.stem.rsplit('_t',1)[1])/2 for q in p.glob('*_t*.json')))
   for p in sorted((root/'states').iterdir()) if p.is_dir() and not (p/'COMPLETE.json').exists()]
    if (root/'states').is_dir() else [],
  unfinished_label_dirs=sorted(p.name for p in (root/'labels').iterdir() if p.is_dir() and p.name not in done)
    if (root/'labels').is_dir() else [],cache_manifest=manifest,
  log_tail=(root/'RUN.log').read_text()[-2400:] if (root/'RUN.log').is_file() else '')
 out.append(record)
print(json.dumps(dict(unix=time.time(),request=REQUEST_VALUE,jobs=out,
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
 live_root_exists=pathlib.Path('/root/autodl-tmp/wmteach_live20261101_v1').exists(),new_sumo_sim_s=0)))
'''
    source = source.replace('JOBS_VALUE', repr(JOBS)).replace('BOOT_VALUE', repr(BOOT))
    source = source.replace('REQUEST_VALUE', repr(request))
    data = json.loads(remote(source, timeout=90))
    dest = HERE / f'OWNER1735_{"REQUEST" if request else "CHECKPOINT"}_{int(data["unix"])}.json'
    with dest.open('x', encoding='utf-8') as f:
        json.dump(data, f, sort_keys=True, allow_nan=False)
        f.write('\n')
    summary = dict(path=str(dest), jobs=[dict(name=j['name'], root=j['root'], process=j['process'],
        generated=j['generated_states'], labelled=j['complete_label_pairs'],
        complete_lineages=j['complete_lineages'], partial_lineages=j['partial_lineages'],
        next_unlabelled_state=j['next_unlabelled_state'],
        quality_futures=j['quality_futures'], quality_labels=j['quality_labels'],
        stop=j['stop_present'], unfinished=j['unfinished_label_dirs'],
        failure=j['files'].get('FAILURE.json'),
        cache_files=len(j['cache_manifest'])) for j in data['jobs']],
        gpu=data['gpu'], live_root_exists=data['live_root_exists'])
    with dest.with_name(dest.stem + '_SUMMARY.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(summary))
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', action='store_true')
    inspect(parser.parse_args().request)

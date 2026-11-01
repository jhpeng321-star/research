"""Published v5 teacher-quality rerun, separate immutable root and ownership."""
import json
from pathlib import Path,PurePosixPath

from tools.coopbandit.protocol import sha
from tools.wmteach.ops import remote,ROOT,REMOTE

QUALITY='/root/autodl-tmp/wmteach_v5quality20261101_v1'
PHYS='/root/autodl-tmp/c4/rb2int20261101/phys_code_b2df7458a41f'
SOURCE='b2df7458a41f5f4d3a048532bc40d8d4435ec381'
RECIPE_SHA='023fa74a4d3059115ec0a5f28d39030a76137d871db5798eb1d23a13fc1e32c5'
HERE=ROOT/'docs/paper_results/wmteach'


def stage():
    exposure=json.loads(Path('E:/agent-work/c4-rb2-int/docs/rb2int/EXPOSURE_B2DF7458_GPUA_V2.json').read_text())
    canary=json.loads(Path('E:/agent-work/c4-rb2-int/docs/rb2int/CANARY_V5_B2DF7458_STATUS03.json').read_text())
    assert exposure['fit_separation_passed'] and canary['receipts']['tasks/DEV_CANARY_PASS.json']['passed']
    assert canary['receipts']['tasks/DEV_CANARY_PASS.json']['recipe_sha256']==RECIPE_SHA
    overrides={f'tools/wmteach/{p.name}':p.read_text(encoding='utf-8')
               for p in (ROOT/'tools/wmteach').glob('*.py')}
    artifacts={str(PurePosixPath(PHYS)/r['path']):r['sha256'] for r in exposure['artifacts']}
    manifest={**exposure['source_manifest'],**artifacts,str(PurePosixPath(PHYS)/'configs/wm_phys_v5.toml'):RECIPE_SHA}
    design=(ROOT/'docs/paper_results/WMTEACH_SCALE_V2_DESIGN.md').read_text(encoding='utf-8')
    return remote(f'''import pathlib,json,hashlib
parent=pathlib.Path({REMOTE!r});root=pathlib.Path({QUALITY!r});assert root.parent.is_dir() and not root.exists()
sources=json.loads({json.dumps(manifest)!r})
for n,d in sources.items():assert hashlib.sha256(pathlib.Path(n).read_bytes()).hexdigest()==d,n
old=json.loads((parent/'IDENTITY.json').read_text())
files={{str(pathlib.Path(n).relative_to(parent)):pathlib.Path(n).read_text()
       for n in old['sources'] if pathlib.Path(n).is_relative_to(parent)}}
files.update(json.loads({json.dumps(overrides)!r}));files['WMTEACH_SCALE_DESIGN.md']={design!r}
root.mkdir()
for n,text in files.items():
 p=root/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:f.write(text)
(root/'deps').symlink_to(parent/'deps',target_is_directory=True);(root/'tasks').mkdir()
sources.update({{str(root/n):hashlib.sha256((root/n).read_bytes()).hexdigest() for n in files}})
identity=dict(schema='wmteach/1',backend='FULL-v5',source={SOURCE!r},sources=sources,
 K=8,horizons=[120,300],fit_worlds=list(range(969400,969432)),settings=[0,1,2,3],
 quality_states=old['quality_states'],new_sumo_sim_s=0,quality_only=True,
 design_sha256=hashlib.sha256((root/'WMTEACH_SCALE_DESIGN.md').read_bytes()).hexdigest(),
 readiness=dict(recipe_sha256={RECIPE_SHA!r},full_fivearm_smoke_pass=True))
with (root/'IDENTITY.json').open('x') as f:json.dump(identity,f,sort_keys=True)
print(json.dumps(dict(root=str(root),sources=len(sources),identity_sha256=hashlib.sha256((root/'IDENTITY.json').read_bytes()).hexdigest())))
''',timeout=90)


def cpu():
    return remote(f'''import pathlib,json,sys,os,unittest
root=pathlib.Path({QUALITY!r});sys.path[:0]=[str(root),str(root/'deps'),{PHYS!r}+'/src',{PHYS!r}]
import tools;tools.__path__.append({PHYS!r}+'/tools')
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',WMTEACH_BACKEND='FULL-v5')
result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromNames([
 'tools.wmteach.test_cpu','tools.wmteach.test_scale','tools.wmteach.test_marginal']))
assert result.wasSuccessful() and not result.skipped
receipt=dict(pass_=True,tests=result.testsRun,new_sumo_sim_s=0)
with (root/'CPU.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''',timeout=90)


def register():
    design=sha(ROOT/'docs/paper_results/WMTEACH_SCALE_V2_DESIGN.md')
    line=(f'- S-WMTEACH FULL-v5 QUALITY RERUN REGISTERED BEFORE DISPATCH: source{SOURCE}/recipeSHA{RECIPE_SHA}/'
          f'designSHA{design}/immutable{QUALITY}. Published FULL-v5 READY + two-world allfivearm600 smoke + '
          '406file exposure audit verified, no oldv4probabilitymap. Same fixed12EXAMstates/K8/H120,H300, '
          'roster-only/sourceprior/coldmemory limitations retained; preserve new physical LC population, '
          'CALlatent wrapper/arrivalspread and registered CAVsettings. Quality-only job cannot enter native '
          'FIT exporter path. Exact targets read only after forecasts; report choice/sign/rank/regret/helpful '
          'agreement before v5scale fitting. Latest releasedv5 fixed as eventual live teacher BEFORE this '
          'newquality forecast, not chosen by exact/live outcomes. Own one-use root/process/strict8GiB/'
          'nice19/oneCPUthread/GPUAoutputs; CPU LC-CAL wrapper regression required; no newSUMO/peerPIDaction/'
          'retries/deletion/newenv/push.\n')
    registry=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
    if line not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(line);f.flush()
    dest=HERE/'V5_QUALITY_REGISTRATION.json'
    if not dest.exists():
        with dest.open('x',encoding='utf-8') as f:json.dump(dict(line=line,design_sha256=design),f);f.write('\n')
    return json.dumps(dict(path=str(dest),design_sha256=design))


def launch():
    registration=json.loads((HERE/'V5_QUALITY_REGISTRATION.json').read_text())
    return remote(f'''import pathlib,json,subprocess,os,shutil,time
root=pathlib.Path({QUALITY!r});assert not (root/'LAUNCH_ONCE.json').exists()
assert json.loads((root/'CPU.json').read_text())['pass_'] and not list((root/'tasks').iterdir())
assert json.loads((root/'IDENTITY.json').read_text())['design_sha256']=={registration['design_sha256']!r}
free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
assert free>12*1024 and shutil.disk_usage(root).free>4*2**30 and os.getloadavg()[0]<24
with (root/'LAUNCH_ONCE.json').open('x') as f:json.dump(dict(unix=time.time()),f)
with (root/'RUN.log').open('x') as log:
 p=subprocess.Popen(['/root/miniconda3/bin/python','-B','-u','-m','tools.wmteach.run',str(root),'--quality-only'],
  cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
  env={{**os.environ,'PYTHONPATH':str(root/'deps'),'WMTEACH_BACKEND':'FULL-v5',
   'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
receipt=dict(pid=p.pid,start_ticks=pathlib.Path(f'/proc/{{p.pid}}/stat').read_text().split()[21],
 boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),root=str(root),new_sumo_sim_s=0)
with (root/'DISPATCH.json').open('x') as f:json.dump(receipt,f)
print(json.dumps(receipt))
''')


def collect():
    result=json.loads(remote(f'''import pathlib,json,time
root=pathlib.Path({QUALITY!r});files={{}}
for name in ('IDENTITY.json','DISPATCH.json','CPU.json','QUALITY.json','QUALITY_RECORDS.json','QUALITY_COMPLETE.json'):
 p=root/name
 if p.is_file():files[name]=json.loads(p.read_text())
owner=files.get('DISPATCH.json');process=None
if owner:
 p=pathlib.Path(f'/proc/{{owner["pid"]}}/stat');st=p.read_text().split() if p.exists() else None
 process=dict(pid=owner['pid'],state=st[2] if st else None,identity_matches=bool(st and st[21]==owner['start_ticks'] and
  pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id']))
print(json.dumps(dict(unix=time.time(),root=str(root),files=files,process=process,
 completed_futures=len(list((root/'tasks').glob('*/k*.json'))),
 completed_labels=len(list((root/'tasks').glob('*/LABEL_H*.json'))),
 log_tail=(root/'RUN.log').read_text()[-8000:] if (root/'RUN.log').exists() else '',new_sumo_sim_s=0)))
''',timeout=90))
    if 'QUALITY_COMPLETE.json' in result['files']:
        from tools.wmteach.labels import helpful_quality
        helpful=helpful_quality([r for r in result['files']['QUALITY_RECORDS.json'] if r['horizon_s']==300])
        result['helpful_H300']=helpful
        q=result['files']['QUALITY.json']['300']
        line=(f'- S-WMTEACH FULL-v5 EXACT QUALITY COMPLETE BEFORE SCALE FITTING: source{SOURCE}/'
              f'recipe{RECIPE_SHA}/12fixedEXAMstates/K8/H120,H300; H300choice{q["choice_agreement"]}/'
              f'sign{q["sign_agreement"]}/rank{q["mean_within_state_spearman"]}/'
              f'exactchosenD{q["exact_chosen_saving"]}/regret{q["exact_regret"]}veh-s/state; '
              f'helpful{helpful["exact_argmax_matches"]}/{helpful["helpful_states"]}argmaxhits/'
              f'{helpful["helpful_positive_choices"]}positivechoices/livequalityPASS{helpful["pass_"]}. '
              'Roster-only/sourceprior/coldmemory diagnostics, not full descendant qualification or '
              'calibrated confidence; no efficacy claim. Model version chosen by published release '
              'BEFORE these forecasts, no exact-result-based policy/snapshot selection. Student awaits '
              '5120distinctstates/worldCV/portableparity/CPU24guards/finalbudget registration; newSUMO0.\n')
        registry=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
        if '- S-WMTEACH FULL-v5 EXACT QUALITY COMPLETE BEFORE SCALE FITTING:' not in registry.read_text(encoding='utf-8'):
            with registry.open('a',encoding='utf-8') as f:f.write(line);f.flush()
        result['registration_line']=line
    dest=HERE/f'V5_QUALITY_SNAPSHOT_{int(result["unix"])}.json'
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    return json.dumps(dict(path=str(dest),process=result['process'],futures=result['completed_futures'],
                          labels=result['completed_labels'],log_tail=result['log_tail'][-2000:]))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','cpu','register','launch','collect'])
    print(globals()[parser.parse_args().action]())

"""CPU release gates for the landed niche module; no activation or SUMO dispatch."""
import argparse
import base64
import json
import zlib
from . import ops
from .ls_validity_ops import remote
from .protocol import checked,sha,REGISTRY
from .storage import read,write

SOURCE='/home/ubuntu/coopmwm20261103_nichenullcpu5'
BASE='/home/ubuntu/coopmwm20261103_lsfastcpu5'


def qualify():
    import subprocess
    names=('mppi_search','mppi_runtime','plan_service','niche_collect','ls_gpu','ls_collection',
        'sumo_verify','feasible','nullfix_geometry','test_nullfix_cpu','test_nullfix_geometry_cpu',
        'ls_feasible','test_ls_feasible_cpu','ls_broker','niche_worker','test_niche_drain_cpu',
        'test_niche_search_cpu','test_niche_collect_cpu')
    files={'pytools/tools/coopmwm/'+n+'.py':checked(ops.LOCAL/'tools/coopmwm'/(n+'.py')).read_text(encoding='utf-8') for n in names}
    fixture='pytools/tools/coopmwm/fixtures/niche1_golden.json'
    files[fixture]=checked(ops.LOCAL/'tools/coopmwm/fixtures/niche1_golden.json').read_text(encoding='utf-8')
    files['FROZEN_GEOMETRY_SOURCE.py']=checked(ops.LOCAL/'traffic_shepherd/c4/geometry.py').read_text(encoding='utf-8')
    files['FROZEN_PRE_NICHE_SEARCH.py']=subprocess.check_output(
        ['git','show','35d31859:tools/coopmwm/mppi_search.py'],cwd=ops.LOCAL).decode('utf-8')
    packed=base64.b64encode(zlib.compress(json.dumps(files).encode())).decode()
    network=base64.b64encode(checked(ops.LOCAL/'assets/research/highway_merge_shepherd_long/osm.net.xml.gz').read_bytes()).decode()
    print(remote('cpuQ',f'''from pathlib import Path
import json,os,shutil,hashlib,subprocess,time,base64,zlib
r=Path({SOURCE!r});base=Path({BASE!r});assert r.parent.is_dir()
files=json.loads(zlib.decompress(base64.b64decode({packed!r})))
if not r.exists():
 r.mkdir();shutil.copytree(base/'pytools',r/'pytools',ignore=shutil.ignore_patterns('__pycache__'))
 for name in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py'):shutil.copy2(base/name,r/name)
 for n,text in files.items():
  p=r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
 (r/'FROZEN_NETWORK.net.xml.gz').write_bytes(base64.b64decode({network!r}))
 hashes={{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'pytools').rglob('*.py')}}
 hashes[{fixture!r}]=hashlib.sha256((r/{fixture!r}).read_bytes()).hexdigest()
 hashes['FROZEN_PRE_NICHE_SEARCH.py']=hashlib.sha256((r/'FROZEN_PRE_NICHE_SEARCH.py').read_bytes()).hexdigest()
 hashes['FROZEN_GEOMETRY_SOURCE.py']=hashlib.sha256((r/'FROZEN_GEOMETRY_SOURCE.py').read_bytes()).hexdigest()
 hashes['FROZEN_NETWORK.net.xml.gz']=hashlib.sha256((r/'FROZEN_NETWORK.net.xml.gz').read_bytes()).hexdigest()
 with (r/'LS_SOURCE_HASHES.json').open('x') as f:json.dump(hashes,f)
 tests=json.loads((base/'LS_RELEASE_TESTS.json').read_text())+['test_niche_search_cpu','test_niche_collect_cpu','test_nullfix_cpu','test_nullfix_geometry_cpu','test_ls_feasible_cpu','test_niche_drain_cpu']
 with (r/'LS_RELEASE_TESTS.json').open('x') as f:json.dump(sorted(set(tests)),f)
for n,text in files.items():assert (r/n).read_text()==text,n
done=r/'LS_CPU_RESULT.json';dispatch=r/'LS_CPU_DISPATCH.json'
if done.exists():print(done.read_text())
elif dispatch.exists():print(dispatch.read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',COOPMWM_NICHE_BASELINE=str(r/'FROZEN_PRE_NICHE_SEARCH.py'),COOPMWM_NULLFIX_GEOMETRY=str(r/'FROZEN_GEOMETRY_SOURCE.py'),COOPMWM_NULLFIX_NETWORK=str(r/'FROZEN_NETWORK.net.xml.gz'))
 with (r/'LS_CPU.log').open('x') as log:
  code='from tools.coopmwm.nullfix_geometry import install;install('+repr(str(r/'FROZEN_GEOMETRY_SOURCE.py'))+');from tools.coopmwm.ls_cpu_test import run;run('+repr(str(r))+')'
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-c',code],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with dispatch.open('x') as f:json.dump(d,f)
 print(json.dumps(d))'''))


def result():
    from . import ls_label_ops
    ls_label_ops.SOURCE=SOURCE
    ls_label_ops.result()


def register():
    cpu=read(ops.EVIDENCE/('LS_LABEL_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    ready_path=ops.LOCAL.parent/'c4-niche/NICHE_READY.json'
    ready=read(ready_path);assert ready['commit']=='5df3e853'
    null_path=ops.LOCAL.parent/'c4-nulldiag/NULLFIX_READY.json'
    null_ready=read(null_path);assert null_ready['commit']=='21390262'
    p=ops.EVIDENCE/'NICHE_NULLFIX_COMPARISON_REGISTRATION.json'
    if not p.exists():
        line=('\nS-COOPMWM MAIN03:06/04:26 JOINT NICHE+NULLFIX CPU RELEASE: '
            'niche5df3e853 integrated1a36cd4e; nullfix21390262 integrateddd6368a4. '
            'Search actuator-feasibility clamp/policy-duplicate exclusion; LS label '
            'adapter calls the delegated actionable gate BEFORE snapshot materialization '
            'or any SUMO branch. A provably inactionable slate gets a sealed INACTIONABLE '
            'receipt, NULL exact grade, zero actual physics, original precharge retained. '
            'Partial slates verify only actionable candidates plus policy, preserving '
            'the raw8 search pool and all discarded IDs; no new charge or look-ahead. '
            'Allowed speeds from subscriptions or SHA-bound preserved road network; '
            'additive geometry helper checks original LANE_TABLE/HERE_OFFSETS equality, '
            'explicit network path only/no global data-root change. Mirror remains OFF. '
            'B-only prospective new admissions alternate niches1,4 '
            'from ordinal0 at the recorded activation count, independent of outcomes. '
            'Same it2_K2 full dimension-scaled N, screenK2H300, finalK8/Hfinal and '
            'up-to8 actionable+policy SUMO lists in existing LS envelope; NO extra SUMO. '
            'No known-family seeds, no R change. Metadata config is explicitly bound '
            'and whitelisted, signed into request/label hashes. Keep all eight final '
            'candidates including negative predictions; preserve niche IDs even if '
            'all candidates lose. Record pairwise distance, niche occupancy, exact '
            'acceleration sign mix and observable member positions. Report VARIATION '
            'including missing/declined lists, timing and best SUMO margins. '
            'Linux M1 byte identity compares SAME-RUNTIME immutable pre-change Git '
            'module35d31859 (SHA in CPU receipt); Windows golden retained, cross-BLAS '
            'roundoff failures preserved, NO tolerance widened. Activation remains '
            'blocked until source-bound per-slot GPU gates pass and the LS adapter is '
            'deployed. Existing four-slot it2_K2 handover complete; qualify ONE slot '
            'at a time while other workers serve. All failed qualification receipts '
            'retained; frozen-net binding tested separately from missing-net fixtures. '
            'New append-only per-owner drain namespace keeps old drain receipts. '
            'Registered metadata broker handover adopts original assignments/queries; '
            'never repeat inference. Original numerical files backed up with exact '
            'hashes before the retired slot gets new sources; original GPU_BINDING '
            'retained and new effective binding records current hashes. ALL8fresh '
            'startup gates required before admission. gpuR uses batch width1 to bound '
            'qualification/serving memory while peers serve; A widths unchanged. '
            'All horizons/K/population/iterations remain registered; no mirror activation. '
            'Conservative6M total guard/phase caps unchanged; no PPO restart.\n')
        write(p,dict(line=line,cpu=cpu,delegated_ready=ready,ready_sha256=sha(ready_path),
            nullfix_ready=null_ready,nullfix_ready_sha256=sha(null_path),
            authority='MAIN2026-11-03 01:33',activation=False,new_sumo_sim_s=0,
            prerequisite='source-bound joint GPU qualification and LS adapter deployment'))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+sha(p)+'\n')
    print(json.dumps(dict(amendment_sha256=sha(p),cpu_tests=cpu['tests'],activation=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('qualify','result','register'))
    globals()[p.parse_args().action]()

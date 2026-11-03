"""One-use, MAIN-only interim analysis; never controls training or physics."""
import argparse
import base64
import json
import time
from pathlib import Path
from . import ops
from .ls_validity_ops import remote
from .ls_collect_ops import ROOT
from .protocol import checked, sha, REGISTRY
from .storage import read, write

BASE = '/home/ubuntu/coopmwm20261103_compositereleasecpu1'
SOURCE = '/home/ubuntu/coopmwm20261103_interimanalysis1'
CONTROL = ROOT+'/interim_analysis_1235_v1'
REG = 'INTERIM_U8_ANALYSIS_REGISTRATION_1235.json'
CPU = 'INTERIM_U8_ANALYSIS_CPU_1235.json'
DISPATCH = 'INTERIM_U8_ANALYSIS_DISPATCH_1235.json'
DELIVERED = 'INTERIM_U8_ANALYSIS_DELIVERED_1235.json'
AUTHORITY = 'MAIN2026-11-03 12:35'
MAIN_TARGET = 'E:/agent-work/_sol/COOPMWM_INTERIM_U8_MAIN.json'
PUBLISH_PUBLIC = False


def qualify():
    files = {n: checked(ops.LOCAL/'tools/coopmwm'/n).read_text(encoding='utf-8') for n in
             ('interim_u8_analysis.py', 'test_interim_u8_analysis_cpu.py')}
    v = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,os,subprocess,time,hashlib
r=Path({SOURCE!r});assert r.parent.is_dir();r.mkdir(exist_ok=True)
files=json.loads({json.dumps(files)!r})
for n,text in files.items():
 p=r/n
 if p.exists():assert p.read_text()==text,n
 else:
  with p.open('x') as f:f.write(text)
cpu=r/'CPU.json'
if not cpu.exists():
 env=dict(os.environ,PYTHONPATH={BASE!r}+'/pytools',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 start=time.time();p=subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B','-m','unittest','discover','-s',str(r),'-p','test_interim_u8_analysis_cpu.py','-v'],env=env,capture_output=True,text=True,timeout=60)
 v=dict(pass_=p.returncode==0,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,wall_s=time.time()-start,new_sim_s=0,new_inference=0,source_hashes={{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in files}})
 with cpu.open('x') as f:json.dump(v,f)
print(cpu.read_text())''', timeout=75))
    p = ops.EVIDENCE/CPU
    if not p.exists():write(p, v)
    print(json.dumps(dict(receipt=str(p), **v)))


def register():
    p = ops.EVIDENCE/REG
    if not p.exists():
        cpu = read(ops.EVIDENCE/CPU)
        assert cpu['pass_'] and cpu['new_sim_s'] == cpu['new_inference'] == 0
        live = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,sys
sys.path.insert(0,{BASE!r}+'/pytools');sys.path.insert(0,{SOURCE!r})
from interim_u8_analysis import metadata
from tools.coopmwm.accel_protocol import worlds
from tools.coopmwm.protocol import digest
r=Path({ROOT!r});panel=worlds('final_test');m=metadata(r,panel)
cpu=json.loads(Path({BASE!r},'LS_CPU_RESULT.json').read_text());assert cpu['pass_'] and cpu['sim_s']==cpu['skips']==0
print(json.dumps(dict(panel_sha256=digest(panel),public_manifest_sha256=digest(m),episodes=len(m),frozen_dependencies=cpu['source_hashes'])))''', timeout=45))
        assert live['episodes'] == 480
        line = ('\nS-COOPMWM MAIN2026-11-03 12:35 COMPLETED FIXED-U8 INTERIM ANALYSIS: '
            'A/B/R each96 public completions and both completion certificates verified before '
            'any outcome decoding; exact checkpoint8 hashes and registered panel pinned. '
            'Read ONLY interim008 A/B/R and reused fixedfinal MAPPO/native episode results '
            'through checked paths, no sealed directory, no A24/A48/B24/R24/branch outcomes. '
            'Existing qualified world-cluster bootstrap10000/RNG20261101/95CI; J600/J900 '
            'B-A/B-MAPPO/B-R plus A/R-MAPPO, all96 and focus demand2400-2700/share.2-.4; '
            'percent reductions, per-cell descriptives, all existing episode safety metrics. '
            'Interim episode Holm analogue alpha.05 over{all,focus} separately for each '
            'B-A/B-MAPPO comparator AND endpoint; no across-endpoint/interim-final adjusted '
            'claim, final registered branch families unchanged. MAIN-only result, public '
            'hash/completion metadata only. No training/search/loss/checkpoint/endpoint '
            'change conditioned on results; primaryu24 unchanged. One-use detached nice19 '
            'CPU analysis, no SUMO/inference/reservation, zero additional sim-s.\n')
        if PUBLISH_PUBLIC:
            line += ('MAIN13:35 extension: add B-native to the same completed episode grid; '
                'also report B-R/B-native two-scope Holm families separately per endpoint, '
                'without changing original B-A/B-MAPPO families. Publish INTERIM_U8_RESULT.json '
                'and authorised summary. Original analysis/source/receipts retained; zero '
                'new physics/inference. Training and V2V decisions never conditioned on interim.\n')
        write(p, dict(authority=AUTHORITY, line=line, cpu=cpu,
            source_root=BASE, driver_root=SOURCE, control_root=CONTROL,
            driver_sha256=cpu['source_hashes']['interim_u8_analysis.py'], **live))
    v = read(p);h = sha(p);registry = checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a', encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    print(remote('cpuQ', f'''from pathlib import Path
import json
r=Path({ROOT!r});v=json.loads({json.dumps(dict(v,amendment_sha256=h))!r});p=r/{REG!r}
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
a=r/'AUTHORITY.md'
if {h!r} not in a.read_text():
 with a.open('a') as f:f.write(v['line']+'amendmentSHA'+{h!r}+chr(10))
print(json.dumps(dict(registered=True,sha256={h!r},new_sim_s=0,outcomes_opened=False)))''', timeout=40))


def launch():
    v = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,os,subprocess,fcntl,time,hashlib
r=Path({ROOT!r});c=Path({CONTROL!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
lock=(c/'DISPATCH.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (c/'DISPATCH.json').exists():print((c/'DISPATCH.json').read_text());raise SystemExit
v=json.loads((r/{REG!r}).read_text());assert v['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
assert hashlib.sha256(Path({SOURCE!r},'interim_u8_analysis.py').read_bytes()).hexdigest()==v['driver_sha256']
env=dict(os.environ,PYTHONPATH={BASE!r}+'/pytools',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (c/'RUN.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u',{SOURCE!r}+'/interim_u8_analysis.py',str(r),str(c),str(r/{REG!r})],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
d=dict(pid=p.pid,start_ticks=Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),control_root=str(c),amendment_sha256=v['amendment_sha256'],analysis_only=True,new_sim_s=0)
with (c/'DISPATCH.json').open('x') as f:json.dump(d,f)
print(json.dumps(d))''', timeout=45))
    p = ops.EVIDENCE/DISPATCH
    if not p.exists():write(p, v)
    print(json.dumps(v))


def status():
    v = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,time
c=Path({CONTROL!r});d=json.loads((c/'DISPATCH.json').read_text());p=Path('/proc',str(d['pid']))
s=(p/'stat').read_text().split() if p.exists() else None
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==d['boot_id']
assert s is None or s[21]==d['start_ticks']
print(json.dumps(dict(unix=time.time(),owner=d,state=s[2] if s else 'ABSENT',complete=json.loads((c/'COMPLETE.json').read_text()) if (c/'COMPLETE.json').exists() else None,failed=json.loads((c/'FAILED.json').read_text()) if (c/'FAILED.json').exists() else None)))''', timeout=40))
    p = ops.EVIDENCE/f'INTERIM_U8_ANALYSIS_STATUS_{time.time_ns()}.json';write(p, v)
    print(json.dumps(dict(receipt=str(p), **v)))


def deliver():
    value = json.loads(remote('cpuQ', f'''from pathlib import Path
import json,base64,hashlib
c=Path({CONTROL!r});v=json.loads((c/'COMPLETE.json').read_text());assert v['pass_'] and not (c/'FAILED.json').exists()
payload=(c/'RESULT.json').read_bytes();assert hashlib.sha256(payload).hexdigest()==v['result_sha256']
print(json.dumps(dict(complete=v,payload=base64.b64encode(payload).decode())))''', timeout=45))
    # Private MAIN delivery; result values are never printed in this operator CLI.
    target = checked(Path(MAIN_TARGET))
    assert target.parent.is_dir()
    blob = base64.b64decode(value['payload'])
    if target.exists():assert target.read_bytes() == blob
    else:
        with target.open('xb') as f:f.write(blob)
    report = json.loads(blob)
    lines = ['# S-COOPMWM fixed-u8 interim — MAIN only', '',
        'Fixed A/B/R u8,96 paired worlds,32 focus worlds. MAPPO/native reused; zero new SUMO.',
        'Interim only; primary u24 unchanged. No training/search/checkpoint changes may follow these results.',
        'Differences below are treatment minus comparator; negative delay is better.',
        '95% intervals are10000 whole-world paired resamples,RNG20261101.', '']
    for horizon in ('600', '900'):
        for scope in ('all', 'focus'):
            lines.append(f'## J{horizon} — {scope}')
            for comparator in (('A', 'mappo', 'R', 'native') if PUBLISH_PUBLIC else ('A', 'mappo', 'R')):
                key=f'{horizon}:{scope}:B-{comparator}:'
                x=report['results'][key+'team_delay_vs'];lo,hi=x['ci95']
                pct=x['pct_delay_reduction'];plo,phi=x['pct_ci95']
                adj=x.get('holm_within_endpoint_comparator')
                inference=(f"; Holm p={adj['p_holm']:.6g},reject={adj['reject']}" if adj else '; descriptive secondary contrast')
                lines.append(f"- B-{comparator}: {x['mean_diff_vs']:+.3f}veh-s95%CI[{lo:+.3f},{hi:+.3f}]; delay reduction{pct:+.3f}%[{plo:+.3f},{phi:+.3f}]{inference}.")
                safety=[]
                for metric in ('collisions','teleports','ttc_below_3_vs','hard_brake'):
                    m=report['results'][key+metric];l,u=m['ci95']
                    safety.append(f"{metric} {m['mean_diff_vs']:+.4f}[{l:+.4f},{u:+.4f}]")
                lines.append('  - Safety mean differences95%CI: '+'; '.join(safety)+'.')
            lines.append('')
    lines += ['## Inference and provenance', report['holm'],
        'Episode-level interim only; no branch outcomes or final-policy outcomes decoded.',
        'Per-cell means,all arm contrasts and safety metrics are in the adjacent JSON.',
        'ResultSHA256 '+value['complete']['result_sha256'],
        'RegistrationSHA256 '+report['amendment_sha256'], '']
    markdown = target.with_suffix('.md');text='\n'.join(lines)
    if markdown.exists():assert markdown.read_text(encoding='utf-8') == text
    else:
        with markdown.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    receipt = dict(value['complete'], main_report=str(target), main_summary=str(markdown),
        local_sha256=sha(target), main_summary_sha256=sha(markdown))
    if PUBLISH_PUBLIC:
        public = ops.EVIDENCE/'INTERIM_U8_RESULT.json'
        if public.exists():assert read(public) == report
        else:write(public, report)
    p = ops.EVIDENCE/DELIVERED
    if not p.exists():write(p, receipt)
    print(json.dumps(dict(receipt=str(p), **receipt)))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=('qualify','register','launch','status','deliver'))
    globals()[p.parse_args().action]()

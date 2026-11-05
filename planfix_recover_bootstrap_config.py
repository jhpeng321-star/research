"""Recover exact printed checkpoint config/export metadata from preserved ops logs.

The legacy candidate's policy/optimizer tensors are not needed by WMTEACH:
bootstrap consumes only its saved config. Independent original inspections must
agree; this is recovery of recorded inputs, never a default/config substitute.
"""
import hashlib
import json
from pathlib import Path
from planfix_ops import remote, note, TEMP
from planfix_cpu_prepare import blob

HERE=Path(__file__).parent
LOGS=Path('E:/agent-work/_sol')

def recover():
    configs=[];sources=[]
    for name in ('setupaudit.log','setupaudit2.log'):
        path=LOGS/name;raw=path.read_bytes();lines=raw.decode(errors='replace').splitlines()
        found=[]
        for i,line in enumerate(lines):
            if line.startswith('CONFIG {') and 'CHECKPOINT de235a787024bb0f79d43937132eefcea69267b38205bc020f95bc98d4e8c7d1' in lines[max(0,i-3):i]:
                cfg=json.loads(line[len('CONFIG '):]);configs.append(cfg);found.append(i+1)
        assert found, 'missing independent original checkpoint inspection'
        sources.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),config_lines=found))
    assert len(configs)>=3 and all(c==configs[0] for c in configs)
    path=LOGS/'speedhead.log';raw=path.read_bytes();exports=[]
    for line in raw.decode(errors='replace').splitlines():
        marker='S-SPEEDHEAD FINAL EXPORT BOUND '
        if marker in line:
            value=line.split(marker,1)[1]
            if value.startswith('{'):exports.append(json.loads(value))
    assert exports and all(x==exports[0] for x in exports)
    export=exports[0]
    assert export['sha256']=='d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
    assert export['source']=='/home/ubuntu/mappospeed/production2/policy_latest.pt'
    sources.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),record='S-SPEEDHEAD FINAL EXPORT BOUND'))
    return dict(config=configs[0],original_candidate_sha256='de235a787024bb0f79d43937132eefcea69267b38205bc020f95bc98d4e8c7d1',
        independent_config_observations=len(configs),original_export=export,source_receipts=sources,
        new_sim_s=0,config_recovered_not_substituted=True)

if __name__=='__main__':
    import argparse,base64,zlib,subprocess
    parser=argparse.ArgumentParser();parser.add_argument('--stage',action='store_true');a=parser.parse_args()
    recovered=recover();p=HERE/'PLANFIX_BOOTSTRAP_CONFIG_RECOVERED.json'
    if p.exists():assert json.loads(p.read_text())==recovered
    else:
        with p.open('x') as out:json.dump(recovered,out,indent=2,sort_keys=True)
    print(json.dumps(dict(independent_config_observations=recovered['independent_config_observations'],config_fields=len(recovered['config']),
                         cav_exec_cap=recovered['config']['cav_exec_cap'],export_sha256=recovered['original_export']['sha256'])),flush=True)
    if not a.stage:raise SystemExit(0)
    driver=blob('22cc32f6','configs/execparity_cav_idm_v1.json')
    files={'RECOVERED.json':base64.b64encode(p.read_bytes()).decode(),
           'execparity_cav_idm_v1.json':base64.b64encode(driver).decode()}
    packed=zlib.compress(json.dumps(files,sort_keys=True).encode(),9);sha=hashlib.sha256(packed).hexdigest()
    package=TEMP/f'planfix_bootstrap_{sha[:12]}.z'
    if package.exists():assert package.read_bytes()==packed
    else:
        with package.open('xb') as out:out.write(packed)
    destination='/home/ubuntu/coopmwm20261105_clean1/plan_fix/'+package.name
    check=remote('cpuV',f"from pathlib import Path; import hashlib; p=Path({destination!r}); print('PACKAGE',p.exists(),hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else '')",timeout=25)
    if 'PACKAGE True '+sha not in check:
        assert 'PACKAGE False' in check
        result=subprocess.run(['wsl','-d','Ubuntu-22.04','--','python3','-B',
            '/mnt/e/agent-work/c4-variety-collect/planfix_cpuv_transport.py','--timeout','35',
            '--put','/mnt/e/Temp/opencode/'+package.name,destination],capture_output=True,timeout=95)
        print(result.stdout.decode(errors='replace'),result.stderr.decode(errors='replace'),flush=True)
        assert result.returncode==0,'inspect content-addressed package before any repeat'
    code=r'''
import base64,hashlib,json,os,zlib
from pathlib import Path
root=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_bootstrap_recovered');root.mkdir(exist_ok=True)
packed=Path(__ARCHIVE__).read_bytes();assert hashlib.sha256(packed).hexdigest()==__SHA__
files=json.loads(zlib.decompress(packed))
for name,text in files.items():
 raw=base64.b64decode(text);p=root/name
 if p.exists():assert p.read_bytes()==raw
 else:
  with p.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
d=json.loads((root/'RECOVERED.json').read_text());physical=Path(d['original_export']['source'])
assert hashlib.sha256(physical.read_bytes()).hexdigest()==d['original_export']['sha256']
print('RECOVERED_BOOTSTRAP_VERIFIED',json.dumps(dict(root=str(root),physical_path=str(physical),physical_sha256=d['original_export']['sha256'],
 config_fields=len(d['config']),independent_config_observations=d['independent_config_observations'],new_sim_s=0)),flush=True)
'''.replace('__ARCHIVE__',repr(destination)).replace('__SHA__',repr(sha))
    text=remote('cpuV',code,timeout=25)
    receipt=json.loads(next(x[len('RECOVERED_BOOTSTRAP_VERIFIED '):] for x in text.splitlines() if x.startswith('RECOVERED_BOOTSTRAP_VERIFIED ')))
    with (HERE/'PLANFIX_CPU_BOOTSTRAP_ASSETS_VERIFIED.json').open('x') as out:json.dump(receipt,out,indent=2,sort_keys=True)
    note('cpuV','legacy checkpoint CONFIG recovered from independent SHA-bound original inspections; physical export matches original d9042cc5; no model/scoring substitution')

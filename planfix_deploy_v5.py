"""New namespace for unmeasured inputs, corrected qualified runtime import order."""
import ast
from pathlib import Path
from planfix_ops import remote

here=Path(__file__).parent
template=next(node.value for node in ast.walk(ast.parse((here/'planfix_deploy_v4.py').read_text())) if
 isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.startswith('import ast,hashlib,json,os,subprocess,time'))
code=template.replace('BANK_DISPATCH_v4','BANK_DISPATCH_v5').replace('BANK_DISPATCH_v3','BANK_DISPATCH_v4')
code=code.replace("'PLANFIX own-wave hold' in (w/'BANK_JOB_v3.log')", "'FileNotFoundError' in (w/'BANK_JOB_v4.log')")
code=code.replace("==2400", "==4800").replace("w/'bank_verification_v3'", "w/'bank_verification_v4'")
code=code.replace("not (w/'bank_verification_v4').exists()", "not (w/'bank_verification_v5').exists()")
code=code.replace("old=w/'ops_4ea920bda5d1';ops=w/'ops_v4_ownerstopped'", "old=w/'ops_v4_ownerstopped';ops=w/'ops_v5_qualified_runtime'")
code=code.replace("target=ops/'planfix_bank_v4.py'", "target=ops/'planfix_bank_v5.py'")
code=code.replace("str(ops)+':'+source+'/pytools:'+source+'/runtime_src:'+source+'/WM_PACKAGE:/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime'",
 "str(ops)+':'+source+'/pytools:/home/ubuntu/e3final/runtime:/home/ubuntu/e3final/runtime/src:'+source+'/runtime_src:'+source+'/WM_PACKAGE'")
code=code.replace("BANK_JOB_v4.log').open('x')", "BANK_JOB_v5.log').open('x')").replace('BANK_DISPATCH_V4','BANK_DISPATCH_V5')
code=code.replace('previous_physics_sim_s=2400','previous_physics_sim_s=4800')
start=code.index('with (r/\'AUTHORITY.md\').open(\'a\') as out:')
end=code.index("with (w/('BANK_JOB",start) if "with (w/('BANK_JOB" in code[start:] else code.index("with (w/'BANK_JOB_v5",start)
authority=r'''with (r/'AUTHORITY.md').open('a') as out:
 out.write('\nPLANFIX v5 prospective remaining-input continuation: v4 own264975 exited on FileNotFoundError before SUMO start; phase4800 retained, no scientific result. Zero-SUMO spawned-process diagnosis reproduced exact missing runtime_src/baselines/circles/adapter.json, then proved physical e3final/runtime first in PYTHONPATH reaches the same qualified SUMO1.27.1 Popen with subprocess creation blocked. Qualified physical runtime/checkpoint and full-prefix checks stay unchanged. Preserve all v2/v3/v4 artifacts; failed986052 and986066sourceCRN1 are excluded, never replayed. New bank_verification_v5 namespace for six untouched inputs only. Same eight nice19 one-thread workers, owner-stopped/no-clean2 wave guard, full1200s/exact offsets/native all8/solo/safety, sole clean1 ledger and600000phase cap. Local REGISTRATION.md registered before dispatch. Exact source '+json.dumps(manifest,sort_keys=True)+'\n')
 out.flush();os.fsync(out.fileno())
'''
code=code[:start]+authority+code[end:]
proof=r'''started=list((w/'bank_verification_v4').glob('*/STARTED.json'));assert len(started)==1
item=next(row for row in json.loads((w/'BANK_INVENTORY.json').read_text())['rows'] if row['id']==started[0].parent.name)
from hashlib import sha256
input_sha=sha256(json.dumps(item,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert input_sha==json.loads(started[0].read_text())['input_sha256']
failure=dict(input_sha256=input_sha,charged_sim_s=2400,original_start=str(started[0]),reason='missing staged adapter config, retained failed without replay')
p=w/'ENVIRONMENT_FAILED_INPUT_V4.json'
if p.exists():assert json.loads(p.read_text())==failure
else:
 with p.open('x') as out:json.dump(failure,out,sort_keys=True)
probe=ops/'planfix_spawn_probe.py';probe_text=__PROBE__
with probe.open('x') as out:out.write(probe_text)
manifest[probe.name]=hashlib.sha256(probe.read_bytes()).hexdigest()
check=subprocess.run([python,'-B',str(probe)],env=env,cwd=ops,capture_output=True,text=True,timeout=25)
print('EXACT_SPAWN_NO_SUMO_PROBE',check.stdout,check.stderr,flush=True)
assert check.returncode==0 and 'ZERO_SUMO_SPAWN_PROBE' in check.stdout and 'FileNotFoundError' not in check.stdout+check.stderr
assert '/home/ubuntu/e3final/runtime/traffic_shepherd/c4/env.py' in check.stderr
'''.replace('__PROBE__',repr((here/'planfix_spawn_probe.py').read_text()))
code=code.replace("with (r/'AUTHORITY.md').open('a') as out:",proof+"with (r/'AUTHORITY.md').open('a') as out:")
code=code.replace('__WRAPPER__',repr((here/'planfix_bank_v5.py').read_text()))
ast.parse(code)
remote('cpuV',code,timeout=40)

"""Keep the old own WM stopped after detecting a pre-search export-ABI bug."""
import json
from pathlib import Path
from planfix_ops import remote,note

CODE=r'''
import hashlib,json,os,time
from pathlib import Path
w=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/cpu_wm_v6')
owner=json.loads((w/'CPU_OWNER.json').read_text());assert owner['pid']==364064 and owner['start_ticks']=='8210497'
f=Path('/proc/364064/stat').read_text().split();assert f[21]=='8210497' and f[2]=='T'
assert not (w/'SPEED_SEARCH_STARTED.json').exists() and not list(w.glob('STARTUP_*.json'))
receipt=dict(reason='bare PolicyExecutor call returns acceleration array, not pair; use original act_with_lane API',
 owner=owner,unix=time.time(),new_sim_s=0,completed_gates=0,model_paused_preserved=True,
 source_contract='frozen22cc32f6 src/wms/control/mappo/eval_bridge.py: exe(state)->accel; exe.act_with_lane(state)->(accel,lane)',
 next='stage corrected query ABI and cheap preflight in freshv7; wait unchanged pace; retire only exactownv6 before any replacement model dispatch')
p=w/'QUERY_ABI_CORRECTION.json'
if p.exists():receipt=json.loads(p.read_text())
else:
 with p.open('x') as out:json.dump(receipt,out,sort_keys=True);out.flush();os.fsync(out.fileno())
stop=dict(reason='operator_query_abi_correction_preserve_stopped_v6',evidence=str(p),evidence_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),new_sim_s=0)
p=w/'STOP.json'
if p.exists():assert json.loads(p.read_text())==stop
else:
 with p.open('x') as out:json.dump(stop,out,sort_keys=True);out.flush();os.fsync(out.fileno())
print('QUERY_ABI_HELD',json.dumps(receipt),flush=True)
'''
if __name__=='__main__':
    text=remote('cpuV',CODE,timeout=25)
    value=json.loads(next(x[len('QUERY_ABI_HELD '):] for x in text.splitlines() if x.startswith('QUERY_ABI_HELD ')))
    with Path(__file__).with_name('PLANFIX_QUERY_ABI_HELD.json').open('x') as out:json.dump(value,out,indent=2,sort_keys=True)
    note('cpuV','old own WM remains stopped/RAM preserved; pre-search export ABI correction identified before any gate completion/search, corrected fresh source will be pace-gated; no new model or SUMO')

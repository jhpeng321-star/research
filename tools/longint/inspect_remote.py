"""Read only named longint and loop receipts; never dispatch."""
from tools.longint.operate import remote, RUNTIME, RELEASE

remote(f'''import pathlib,subprocess,json
r=pathlib.Path({RUNTIME!r})
print('RUNTIME_RECEIPTS')
if r.exists():
 for p in r.iterdir():
  if p.name.endswith('.json') and p.name!='SOURCE.json':
   print(p.name,p.read_text()[-5000:])
print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True))
r=pathlib.Path({RELEASE!r})
print('RELEASE_RECEIPTS',[p.name for p in r.iterdir() if 'LONGINT' in p.name])
for name in ('arm/CONFIG.json','arm/loop/LOOP14_STATE.json','AUTONOMY_HOLD_C3_RESULT.json'):
 print(name,(r/name).read_text())
''')

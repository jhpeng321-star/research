import json
from pathlib import Path

root = Path.home()/'longhzn'
for name in ('LAUNCH.json','DONE.json','coordinator.log','worker00.log','worker16.log'):
    p = root/name
    if p.is_file():
        print(name,p.read_text())
rows = list((root/'rows').glob('*.json')) if (root/'rows').is_dir() else []
print('ROWS',len(rows), {n:sum(p.name.startswith(n+'_') for p in rows) for n in ('wm0','wm1','r16')})
for file in ('LAUNCH.json','WORKERS.json'):
    p = root/file
    if not p.is_file():
        continue
    records = json.loads(p.read_text())
    if isinstance(records,dict):
        records=[records]
    print('OWN_PID_STATUS',file,[{**r,'live':Path(f"/proc/{r['pid']}/stat").exists() and
          Path(f"/proc/{r['pid']}/stat").read_text().rsplit(')',1)[1].split()[19]==r['start_ticks']} for r in records])
for p in root.glob('worker*.log'):
    text = p.read_text()
    if 'Traceback' in text:
        print('ERROR',p.name,text)

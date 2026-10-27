"""Read only exact recorded main WM/coordinator process metadata and progress files."""
from tools.loopresume.c1collect import PRELUDE
from tools.loopresume.operate import remote


def main():
    remote('gpuA', PRELUDE+'''
for pid in (80364,81201,85784):
 p=pathlib.Path('/proc')/str(pid)
 if p.exists():
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  print('PID',pid,json.dumps({'state':stat[0],'start_ticks':stat[19], 'utime':stat[11],
   'stime':stat[12],'children':(p/'task'/str(pid)/'children').read_text()}))
 else: print('PID',pid,'EXITED')
for name in ('arm/loop/policy/c02/STOP','arm/loop/policy/c02/RUN_SUMMARY.json'):
 p=r/name
 if p.exists():
  if name.endswith('RUN_SUMMARY.json'):
   doc=loop.read_json(p)
   print(name,json.dumps({k:v for k,v in doc.items() if k in ('status','error','critic_warmup','resumed_from','paircrit_monitor')}))
  else: print(name,p.read_text())
for h in (60,30,10):
 p=r/f'C2PREP_SMOKE_{h}.json'
 if p.exists():
  doc=loop.read_json(p)
  rows=doc.pop('updates',[])
  doc['updates']=[{'seconds':row['seconds'], 'horizon':row['metrics']['imagination_horizon_s']} for row in rows]
  print('SMOKE',json.dumps(doc))
''')


if __name__ == '__main__':
    main()

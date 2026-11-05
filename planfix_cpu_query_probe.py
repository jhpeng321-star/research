"""Read source context schema and pose-only API for the one CPU WM speed search."""
from planfix_ops import remote
CODE=r'''
import json,ast
from pathlib import Path
w=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix')
print('OWN_NAMES',[p.name for p in w.iterdir() if p.is_file()],flush=True)
for name in ('BANK_INVENTORY.json','INVENTORY.json','BANK_RESULTS.json'):
 p=w/name
 if p.exists():
  d=json.loads(p.read_text());print('SCHEMA',name,list(d) if isinstance(d,dict) else len(d),flush=True)
  if name!='BANK_RESULTS.json':print('SOURCE_CONTEXT',json.dumps(d)[:20000],flush=True)
r=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa/pytools/tools/wmteach')
for name in ('scale.py','runtime.py'):
 t=(r/name).read_text();print('API',name,flush=True)
 for n in ast.parse(t).body:
  if isinstance(n,ast.FunctionDef) and n.name in ('prepare_batch','source_for','option_controller','diagnostic_source'):
   print(ast.get_source_segment(t,n),flush=True)
'''
if __name__=='__main__':remote('cpuV',CODE,timeout=25)

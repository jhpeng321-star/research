"""Zero-physics source/dependency inventory for the CPU WM migration."""
from planfix_ops import remote

CODE = r'''
import ast,json
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa')
for rel in ('WM_PACKAGE','INPUTS.json','FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','pytools/tools/wmteach/runtime.py','pytools/tools/wmteach/scale.py'):
 p=r/rel;print('CANDIDATE',str(p),p.exists(),flush=True)
 if p.is_file() and p.suffix=='.json':
  data=json.loads(p.read_text());print('JSON_KEYS',rel,list(data)[:35],flush=True)
 if p.is_file() and p.suffix=='.py':
  text=p.read_text();tree=ast.parse(text)
  for n in tree.body:
   if isinstance(n,(ast.Import,ast.ImportFrom)) or isinstance(n,(ast.Assign,ast.AnnAssign)) or isinstance(n,ast.FunctionDef) and n.name=='bootstrap':
    print('SOURCE',rel,ast.get_source_segment(text,n),flush=True)
for base in ('/home/ubuntu/coopmwm20261104_launchcoop_src7','/home/ubuntu/coopmwm20261103_solo1755_v2'):
 for rel in ('INPUTS.json','FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py','pytools/tools/wmteach/runtime.py'):
  p=Path(base)/rel;print('LEGACY',str(p),p.exists(),flush=True)
print('CLEAN2_EXISTS',Path('/home/ubuntu/coopmwm20261105_clean2').exists(),flush=True)
for base in (r/'WM_PACKAGE',Path('/home/ubuntu/coopmwm20261104_launchcoop_src7')):
 print('TOP',str(base),[p.name for p in base.iterdir() if not any(x in p.name.lower() for x in ('holdout','heldout','cal_split','chunk5','j4b','inc1','truth','vehicle_params','vtype'))],flush=True)
p=Path('/home/ubuntu/coopmwm20261104_launchcoop_src7/pytools/tools/wmteach/runtime.py')
if p.is_file():
 text=p.read_text();tree=ast.parse(text)
 for n in tree.body:
  if isinstance(n,ast.FunctionDef) and n.name in ('bootstrap','diagnostic_source'):
   print('LEGACY_FUNCTION',ast.get_source_segment(text,n),flush=True)
'''
if __name__ == '__main__':
    remote('cpuV', CODE, timeout=25)

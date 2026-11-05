"""Read specific CPU bootstrap source and candidate asset metadata only."""
from planfix_ops import remote
CODE=r'''
import ast,json,hashlib
from pathlib import Path
s=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa')
for rel,names in [('pytools/tools/rbeval/core.py',('reference','load')),('pytools/tools/rbint/recipe.py',('load_candidate',)),('pytools/tools/wmteach/runtime.py',('option_controller',))]:
 p=s/rel;print('SOURCE_PATH',str(p),p.exists(),flush=True)
 if not p.exists():continue
 text=p.read_text();tree=ast.parse(text)
 for n in tree.body:
  if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)) or isinstance(n,ast.FunctionDef) and n.name in names:
   print(ast.get_source_segment(text,n),flush=True)
home=Path('/home/ubuntu');m=json.loads((home/'cpuv_setup_20261105/RESEARCH3_MANIFEST.json').read_text())
want=[x for x in m if x.get('sha256')=='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436']
for x in want:
 p=home/x['dst']
 if p.is_file():
  print('WM_FOUND',str(p),hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
  print('WM_SIBLING_NAMES',[q.name for q in p.parent.iterdir() if q.name in ('physics.json','config.resolved.json','wm_s.pt','lane_calib.json','manifest.json')],flush=True)
  for n in ('physics.json','config.resolved.json'):
   q=p.parent/n
   if q.exists():print('WM_SIDECAR',str(q),q.read_text()[:32000],flush=True)
  break
else:print('WM_ASSET_MISSING_ON_CPUV',len(want),flush=True)
print('CLEAN2_EXISTS',(home/'coopmwm20261105_clean2').exists(),flush=True)
'''
if __name__=='__main__':remote('cpuV',CODE,timeout=25)

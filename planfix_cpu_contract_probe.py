"""Explicit source-only reads; no model bootstrap or WM search."""
from planfix_ops import remote
CODE=r'''
import ast,json
from pathlib import Path
s=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa')
for rel in ('pytools/tools/wmteach/scale.py','pytools/tools/wmteach/runtime.py','pytools/tools/wmteach/marginal.py'):
 p=s/rel;text=p.read_text();tree=ast.parse(text);print('FILE',str(p),flush=True)
 for n in tree.body:
  if isinstance(n,ast.FunctionDef) and n.name in ('futures','prepare_batch','plans','source_for','roster'):
   print(ast.get_source_segment(text,n),flush=True)
legacy=Path('/home/ubuntu/coopmwm20261104_launchcoop_src7')
for rel in ('FROZEN_COSTS.json','FROZEN_MOTION_SOURCE.py'):
 p=legacy/rel;text=p.read_text();print('FROZEN',str(p),text[:48000],flush=True)
for rel in ('pytools/tools/wmteach','WM_PACKAGE/wms'):
 p=s/rel;print('MODULE_NAMES',str(p),[x.name for x in p.iterdir()],flush=True)
'''
if __name__=='__main__':remote('cpuV',CODE,timeout=25)

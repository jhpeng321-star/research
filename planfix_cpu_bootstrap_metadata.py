"""Inspect explicit frozen source metadata and candidate loader config ABI only."""
import json
from pathlib import Path
from planfix_ops import remote

CODE=r'''
import ast,json,hashlib
from pathlib import Path
home=Path('/home/ubuntu');deny=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','truth','vtype','vehicle_params')
for name in ('RESEARCH3_MANIFEST.json','RESEARCH_MANIFEST.json'):
 p=home/'cpuv_setup_20261105'/name
 if not p.exists():continue
 for row in json.loads(p.read_text()):
  n=row.get('dst','')
  if any(x in n.lower() for x in deny):continue
  if any(x in n.lower() for x in ('policy_000190','export_bound','dreamer','wmcalib','wmdata20261031','969420_ctrl0')):
   print('ASSET_MAPPING',json.dumps(row),flush=True)
for base in ('coopmwm20261104_launchcoop_src7','coopmwm20261104_launchcoop_src9'):
 for name in ('SOURCE_PROVENANCE.json','STAGED.json'):
  p=home/base/name
  if p.exists():print('SOURCE_METADATA',str(p),p.read_text()[:30000],flush=True)
r=home/'coopmwm20261105_clean1_src_7cc6edaa/pytools/tools/wmteach'
for name in ('runtime.py','scale.py','marginal.py'):
 p=r/name;t=p.read_text();tree=ast.parse(t)
 attrs=sorted({n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and ast.unparse(n.value) in ('rt.cfg','cfg')})
 print('CONSUMED_CFG',name,attrs,flush=True)
for base in ('/home/ubuntu/e3final','/home/ubuntu/mappospeed'):
 p=Path(base);print('CANDIDATE_TOP',base,[(q.name,q.is_file()) for q in p.iterdir() if not any(x in q.name.lower() for x in deny)],flush=True)
'''
if __name__=='__main__':remote('cpuV',CODE,timeout=25)

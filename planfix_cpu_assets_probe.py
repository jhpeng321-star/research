"""Inspect existing manifest entries, never walk research datasets."""
from planfix_ops import remote

CODE=r'''
import json,hashlib
from pathlib import Path
home=Path('/home/ubuntu');control=home/'cpuv_setup_20261105'
for name in ('RESEARCH3_MANIFEST.json','RESEARCH_MANIFEST.json','RESEARCH_ROOTS.json'):
 p=control/name
 if not p.exists():continue
 data=json.loads(p.read_text())
 if name.endswith('ROOTS.json'):print('ROOTS',json.dumps(data),flush=True);continue
 for row in data:
  name=row.get('dst','')
  if any(x in name.lower() for x in ('holdout','heldout','cal_split','chunk5','j4b','inc1','truth','vehicle_params','vtype')):continue
  if (row.get('sha256')=='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436' or
      any(x in name.lower() for x in ('wmcalib','wm_phys.toml','shrinkage_v1','observed_arrivals.json','references.json','speedhead20261030','policy_000190.pt'))):
   print('ASSET_MANIFEST',json.dumps(row),flush=True)
p=home/'coopmwm20261105_clean1_src_7cc6edaa/pytools/tools/rbeval/references.json'
if p.exists():
 data=json.loads(p.read_text());print('REF_KEYS',list(data),flush=True)
 for key in ('worker','costs','motion','starts'):
  if key in data:print('REFERENCE',key,json.dumps(data[key]),flush=True)
print('CLEAN2_EXISTS',(home/'coopmwm20261105_clean2').exists(),flush=True)
print('HOME_TOP_ALLOWED',[p.name for p in home.iterdir() if p.name.startswith(('cpuv','wm','coopmwm','e3final','mappo'))],flush=True)
'''
if __name__=='__main__':remote('cpuV',CODE,timeout=25)

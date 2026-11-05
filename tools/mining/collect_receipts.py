"""Fetch small named gpuA receipts without mutating remote research state."""
import base64
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from remote import run
names=['CLONE_QUALIFICATION.json','BUDGET_ADMISSION.json','CHARGES.jsonl',
       'P3_COMPARE.json','P4_COMPARE.json','PARITY_PASS.json']
code="""import base64,json
from pathlib import Path
r=Path('/root/mining20261106')
print(json.dumps({n:base64.b64encode((r/n).read_bytes()).decode() for n in NAMES}))
"""
result=run('gpuA','NAMES='+repr(names)+'\n'+code,'/root/miniconda3/bin/python -B -')
assert result['returncode']==0,result
root=Path(__file__).resolve().parents[2]/'mining_receipts'
assert root.is_dir()
out=[]
for name,blob in json.loads(result['stdout']).items():
 data=base64.b64decode(blob);p=root/name
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data)
 out.append(dict(name=name,size=len(data),sha256=hashlib.sha256(data).hexdigest()))
print(json.dumps(out,sort_keys=True))

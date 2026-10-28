"""Read the registered horizon threshold and named acceptance receipts only."""
import sys
from remote import ssh

ssh(sys.argv[1], '''
import json
from pathlib import Path
p=Path.home()/'entrant/runtime'
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth')
for f in (p/'configs').glob('*horizon*gate*.json'):
    if any(x in str(f).lower() for x in bad):continue
    d=json.loads(f.read_text());print(str(f),json.dumps({k:v for k,v in d.items() if k!='rows'}))
for name in ('loop14','wmcombined','lrdiag'):
    root=Path.home()/name
    print('ROOT',str(root),root.exists())
    if root.exists():
        print('ALLOWED TOP', [f.name for f in root.iterdir() if not any(x in str(f).lower() for x in bad)])
''')

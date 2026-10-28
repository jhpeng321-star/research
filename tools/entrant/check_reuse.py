"""Check prior LRDIAG OFF evidence before allowing any result reuse."""
import sys
from remote import ssh

ssh(sys.argv[1], '''
import json,copy
from pathlib import Path
p=Path.home()/'entrant'
a=Path.home()/'lrdiag/results/g01_baseline_k0_s600.json'
b=p/'results_v2/e3_g01_baseline_k0_s600.json'
print('FILES',a.exists(),b.exists())
if a.exists() and b.exists():
    x=json.loads(a.read_text());y=json.loads(b.read_text())
    print('KEYS',x.keys())
    print('ARM_KEYS',x['arms']['native'].keys())
    for arm in x['arms']:
        q=copy.deepcopy(y['arms'][arm]);q['delay']={h:q['delay'][h] for h in ('120','300')}
        print(arm,'all_science_identical',q==x['arms'][arm])
        for key in x['arms'][arm]:
            if q.get(key)!=x['arms'][arm][key]:print('DIFFERENT',arm,key)
        for h in (10,30,60,90,120,180,240,300):
            vals=[r for r in x['arms'][arm]['timeline'] if r[1]=='HDV' and r[2]<=h]
            computed=[sum(r[i] for r in vals) for i in (3,4)]
            print(arm,h,'timeline',computed,'direct',y['arms'][arm]['delay'][str(h)])
for panel,pattern in (('e3','g??_baseline_k?_s600.json'),('pairs300','p300_g??_baseline_k?_s600.json')):
    print(panel,'baseline available',len(list((Path.home()/'lrdiag/results').glob(pattern))))
''')

"""Stage the just-captured immutable cpuV allocation audit on gpuA."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
from remote import run
p=Path('E:/Temp/opencode/MINING_BUDGET_0150.json')
audit=json.loads(json.loads(p.read_text())['stdout'])
code="from pathlib import Path; p=Path('/root/mining20261106/BUDGET_ADMISSION.json'); assert p.parent.is_dir(); p.open('x').write("+repr(json.dumps(audit,sort_keys=True,indent=2)+'\n')+"); print('budget admission staged; no physics')"
print(json.dumps(run('gpuA',code,'/root/miniconda3/bin/python -B -')))

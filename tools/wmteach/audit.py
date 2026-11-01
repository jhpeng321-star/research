"""Read-only paired-decision parity with the published S-RB-DECIDE implementation."""
import json
from pathlib import Path

from tools.coopbandit.protocol import sha
from tools.wmteach.ops import remote,REMOTE,ROOT


def main():
    implementation=Path('E:/agent-work/c4-rb-int/tools/wmdecide/decision.py')
    source=implementation.read_text(encoding='utf-8')
    source_sha=sha(implementation)
    result=json.loads(remote(f'''import pathlib,json,sys,types,time
root=pathlib.Path({REMOTE!r})
module=types.ModuleType('wmteach_published_decision');sys.modules[module.__name__]=module
exec(compile({source!r},'<published-S-RB-DECIDE>','exec'),module.__dict__)
rows=[]
for path in sorted((root/'tasks').glob('*/LABEL_H*.json')):
 label=json.loads(path.read_text());h=str(label['horizon_s']);actions={{}}
 for k in range(8):
  task=json.loads((path.parent/f'k{{k}}.json').read_text())
  for a,f in task['results'][h].items():actions.setdefault(a,[]).append(module.Future(**f))
 check=module.decide(actions,'O0');expected=label['paired_gate']
 assert check['selected']==expected['selected']
 assert check['eligible']==expected['eligible']
 for pair in check['summary']['pairwise']:
  if pair['reference']=='O0':assert pair['probability_beats']==expected['probability_beats'][pair['action']]
 for a,summary in check['summary']['actions'].items():
  assert abs(summary['team_delay']['mean']-expected['expected_delay'][a])<1e-8
 rows.append(dict(state_id=label['state_id'],horizon_s=label['horizon_s'],selected=check['selected'],pass_=True))
print(json.dumps(dict(unix=time.time(),published_source_sha256={source_sha!r},contexts=len(rows),rows=rows,
 strict_paired_probability_parity=True,new_inference=False,new_sumo_sim_s=0)))
'''))
    if not result['contexts']:
        print('No complete K8 context yet; no audit result claimed.');return
    path=ROOT/'docs/paper_results/wmteach'/f'WMDECIDE_PARITY_{int(result["unix"])}.json'
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,sort_keys=True);f.write('\n')
    print(json.dumps(dict(path=str(path),contexts=result['contexts'],pass_=True)))


if __name__=='__main__':main()

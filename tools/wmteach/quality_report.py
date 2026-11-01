"""Report completed paired teacher predictions against exact labels, no inference."""
import json
from pathlib import Path

from tools.wmteach.ops import remote,ROOT,REMOTE


def main():
    result=json.loads(remote(f'''import pathlib,json,sys,time
root=pathlib.Path({REMOTE!r});sys.path.insert(0,str(root))
from tools.wmteach.labels import compare,quality
from tools.coopbandit.protocol import sha
inputs=json.loads((root/'QUALITY_INPUTS.json').read_text())
predictions=[]
for r in inputs:
 for h in (120,300):
  p=root/'tasks'/r['state_id']/f'LABEL_H{{h}}.json'
  if p.is_file():predictions.append(json.loads(p.read_text()))
# Forecast files exist before their exact targets are consulted here.
exact={{(r['state_id'],r['horizon_s']):r for r in json.loads((root/'EXACT_QUALITY_ONLY.json').read_text())}}
rows=[compare(r,exact[r['state_id'],r['horizon_s']]) for r in predictions]
print(json.dumps(dict(unix=time.time(),source_root=str(root),completed_before_exact_read=len(predictions),
 full_quality_complete=(root/'QUALITY.json').is_file(),expected_selected_states=len(inputs),
 exact_sha256=sha(root/'EXACT_QUALITY_ONLY.json'),rows=rows,
 quality={{str(h):quality([r for r in rows if r['horizon_s']==h],300) for h in (120,300)
          if any(r['horizon_s']==h for r in rows)}},new_inference=False,new_sumo_sim_s=0)))
'''))
    from tools.wmteach.labels import helpful_quality
    result['helpful_H300']=helpful_quality([r for r in result['rows'] if r['horizon_s']==300])
    result['status']='COMPLETE' if result['full_quality_complete'] else 'PARTIAL_NOT_A_LIVE_GATE'
    path=ROOT/'docs/paper_results/wmteach'/f'EXACT_QUALITY_{int(result["unix"])}.json'
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(path),status=result['status'],quality=result['quality'],
                          helpful_H300=result['helpful_H300'])))


if __name__=='__main__':main()

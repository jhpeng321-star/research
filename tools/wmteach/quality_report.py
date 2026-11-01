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
    if result['full_quality_complete']:
        q=result['quality']['300'];helpful=result['helpful_H300']
        line=(f'- S-WMTEACH TEACHER QUALITY FIRST COMPLETE (FULL-v4 source22cc32f6, before scale fitting): '
              f'{q["states"]}/12prospectivelyselectedEXAMstates/300fullEXAM, K8/H120,H300; '
              f'H300argmaxagreement{q["choice_agreement"]:.6f}/strictPgateagreement{q["gated_choice_agreement"]:.6f}/'
              f'Dsign{q["sign_agreement"]}/rank{q["mean_within_state_spearman"]}; '
              f'exactchosenD{q["exact_chosen_saving"]:.6f}/oracleD{q["exact_oracle_saving"]:.6f}/'
              f'regret{q["exact_regret"]:.6f}veh-s/state. HelpfulH300 '
              f'{helpful["exact_argmax_matches"]}/{helpful["helpful_states"]}exactargmaxhits, '
              f'{helpful["helpful_positive_choices"]}positivechoices, Wilson95lower{helpful["wilson95_lower"]:.6f}, '
              f'meanexactchosenD{helpful["helpful_exact_chosen_saving"]}; '
              f'prospectivelivequalitygatePASS={helpful["pass_"]}. '
              'Roster-only diagnostics marginalize FITdriver/sourcepriors/coldmemory; weaker-input '
              'coverage, not full descendant qualification/calibration/livepolicy efficacy. No exact targets '
              'fit student; no repeated forecasts/newSUMO0. Fixed5120distinctFIT/imaginedstate continuation '
              'remains required before student/live; originalsmallcontinuation retirement fixed at quality '
              'milestone regardless of outcomes, preserving all work. See docs/paper_results/WMTEACH.md '
              'and wmteach/EXACT_QUALITY receipts. No policy-beats-MAPPO claim yet.\n')
        registry=Path('E:/research-dense/runs/active/C4-V7FT-20261017/REGISTRATION.md')
        if '- S-WMTEACH TEACHER QUALITY FIRST COMPLETE ' not in registry.read_text(encoding='utf-8'):
            with registry.open('a',encoding='utf-8') as f:f.write(line);f.flush()
        result['registration_line']=line
    path=ROOT/'docs/paper_results/wmteach'/f'EXACT_QUALITY_{int(result["unix"])}.json'
    with path.open('x',encoding='utf-8') as f:json.dump(result,f,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(path=str(path),status=result['status'],quality=result['quality'],
                          helpful_H300=result['helpful_H300'])))


if __name__=='__main__':main()

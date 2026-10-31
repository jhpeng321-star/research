"""Receipt collection and complete-panel-only report; no simulation/inference."""
import argparse
import csv
import json
from pathlib import Path

from .ops import EVIDENCE, ssh
from .shared import CPU, GPU
from .storage import write

MIRROR=EVIDENCE/'shared_v10_panel'


def collect():
    data=ssh('cpuQ',f'''import json,time
from pathlib import Path
r=Path({CPU!r});files={{}}
for p in r.glob('jobs/*/*.json'):
 if p.name in ('RESULT.json','PHYSICS.json','CLOSE.json','START.json'):
  files[str(p.relative_to(r))]=json.loads(p.read_text())
for name in ('BINDING.json','STOP','recovery/COMPLETE.json','recovery/STATUS.json'):
 p=r/name
 if p.exists():files[name]=json.loads(p.read_text())
owners=[json.loads(p.read_text()) for p in (r/'recovery').glob('OWN_*.json')]+[json.loads((r/'recovery_DISPATCH.json').read_text())]
print(json.dumps(dict(files=files,accounting=[json.loads(x) for x in (r/'accounting.jsonl').read_text().splitlines()],
 own_pids={{str(x['pid']):Path('/proc',str(x['pid'])).exists() for x in owners}},unix=time.time())))
''',timeout=90)
    gpu=ssh('gpuR',f'''import json,time
from pathlib import Path
r=Path({GPU!r});out={{}}
for n in ['SHARED_BUDGET.json','GPU_STARTED.json','GPU_FINISHED.json','gpu_DISPATCH.json']:
 p=r/n
 if p.exists():out[n]=json.loads(p.read_text())
pid=out.get('gpu_DISPATCH.json',{{}}).get('pid')
out['owned_pid_alive']=bool(pid and Path('/proc',str(pid)).exists())
print(json.dumps(out))
''')
    for name,row in data['files'].items():write(MIRROR/name,row)
    with (MIRROR/'accounting.jsonl').open('w',encoding='utf-8') as f:
        for row in data['accounting']:f.write(json.dumps(row)+'\n')
    rows=data['accounting'];files=data['files'];binding=files['BINDING.json']
    if not any(data['own_pids'].values()):
        from . import resume
        from .design import CONTROLLERS, measurement_indices
        original=resume.ROOT;resume.ROOT=MIRROR
        try:
            for n,r in list(files.items()):
                if not n.endswith('_episode/PHYSICS.json'):continue
                arm,world=r['arm'],r['world']
                prefix=n.rsplit('/',1)[0];target=prefix+'/RESULT.json'
                if target in files:continue
                ready=arm not in ('wm','bb') or all(
                    f'jobs/{arm}_s{world["seed"]}_j{j}_{c}/RESULT.json' in files
                    for j in measurement_indices(world['seed']) for c in CONTROLLERS)
                if ready:
                    resume.finalize(arm,world)
                    files[target]=json.loads((MIRROR/target).read_text())
        finally:
            resume.ROOT=original
    closes={n:r for n,r in files.items() if n.endswith('/CLOSE.json')}
    charged=sum(x['sim_s'] for x in rows)
    actual=47010+binding['retained_superseded_sim_s']+binding['retained_abandoned_sim_s']+sum(x['actual_sim_s'] for x in closes.values())
    episode_rows={n:r for n,r in files.items() if n.endswith('_episode/RESULT.json')}
    physics_rows={n:r for n,r in files.items() if n.endswith('_episode/PHYSICS.json')}
    physics_ids={n.rsplit('/',1)[0] for n in episode_rows}|{n.rsplit('/',1)[0] for n in physics_rows}
    summary=dict(charged_sim_s=charged,actual_closed_sim_s=actual,all_cpu_owned_absent=not any(data['own_pids'].values()),
        complete_episodes=len(episode_rows),complete_physics=len(physics_ids),
        complete_branches=sum(n.endswith('/RESULT.json') and '_episode/' not in n for n in files),
        registered_total_sim_s=binding['total_sim_s'],gpu=gpu,unix=data['unix'])
    if summary['all_cpu_owned_absent'] and charged!=actual:raise ValueError('terminal actual/charged mismatch')
    if 'GPU_FINISHED.json' in gpu and gpu['GPU_FINISHED.json']['cumulative_gpu_s']>14400:raise ValueError('owner GPU cap exceeded')
    write(EVIDENCE/'V10_COLLECTION.json',summary)
    print(json.dumps(summary,indent=2))
    return summary


def report():
    summary=json.loads((EVIDENCE/'V10_COLLECTION.json').read_text())
    complete=(MIRROR/'recovery/COMPLETE.json').exists()
    if not summary['all_cpu_owned_absent'] or summary['gpu']['owned_pid_alive']:
        raise ValueError('terminal report requires closed owned jobs')
    target=Path('docs/paper_results/C1SWITCH.md')
    gpu=summary['gpu']['GPU_FINISHED.json']['cumulative_gpu_s']
    common=[
        '# S-C1SWITCH — controller-switching experiment',
        '',
        '## Status',
        '**Complete registered panel.**' if complete else '**Incomplete panel: operational/budget stop; no full-panel efficacy verdict.**',
        ('Terminal stop receipt: '+json.loads((MIRROR/'STOP').read_text()).get('reason','unspecified')) if (MIRROR/'STOP').exists() else 'All registered jobs closed.',
        '',
        '## Frozen scientific contract',
        '- WM2 b8797242+mix (entries OFF), full 60 s anonymous observable history; frozen BB-traj afaaa32c. Original v7 anonymous-seed adapter/controller plant retained.',
        '- Ten decisions every 60 s after 300 s native warmup; library native/CTRL0/MAPPO169/PI/CIRCLES; K8 median predicted on-road fleet delay, incumbent-retaining 1e-6 tie rule.',
        '- Evaluation is true fleet + pending-source team delay. Predicted entries/source delay are unsupported; this objective mismatch is disclosed. BB-id cannot execute feedback controllers.',
        '- Main WM/BB/five fixed arms retain all 16 paired TRAIN worlds969112–969127. Random retains first6 only969112–969117; last10 omitted prospectively for interruption accounting.',
        '- Exact counterfactuals rerun from t0 with the episode controller schedule; three pre-registered shared decision indices/world; all five candidates at60/90s, matching live prefix before intervention. Evaluation branch scheduling cannot feed picks.',
        '- Exposure and limitations: see C1SWITCH_DESIGN.md/C1SWITCH_WINNERS.md and frozen C1-EXPOSURE labels. No model selection/tuning used switching outcomes.',
        '',
        '## Retained evidence and compute',
        f'- Final charged = closed actual: {summary["charged_sim_s"]:,.0f} sim-s. Registered complete-recovery projection499680; owner cap500000.',
        f'- Retained {summary["complete_episodes"]} fully evaluated episodes, {summary["complete_physics"]} complete episode-physics runs, {summary["complete_branches"]} completed full-rerun branches. Complete physics without branches is retained and never rerun.',
        '- All47010 Step0 sim-s,12450 superseded original WM canary sim-s and17820 abandoned prior-prefix sim-s remain charged. Any terminal incomplete shared-v10 physics also remains charged.',
        f'- Cumulative conservative GPU allocation wall: {gpu:,.6f}s ({gpu/3600:.6f}h), including loading, parity validation, idle and interrupted allocations; owner cap14400s.',
        '- Every original v4–v7 source, STOP, schedule and ledger remains preserved. Initial 10 completed episodes +231 branches were imported once; subsequent complete physics/forecasts reused exactly.',
        '',
        '## Engineering gates and interruption history',
        '- Initial paired canary, both original-draw GPU parity gates and anonymous BB-reuse validation passed. Corrected WM-only canary replaced the original scientifically superseded schedule.',
        '- First main interruption hit the old BB3600s internal cap; owner MAIN23:25 authorised shared GPU budget and random-tail trim.',
        '- gpuA recovery CPU gates passed but full60s rawK8 bitwise parity failed both models before physics;57.231276GPU-s retained. No numerical cause or efficacy conclusion was inferred.',
        '- Original gpuR rawK8 bitwise parity passed both models. A subsequent BB200s internal partition expired before owner-cap exhaustion; this operational error caused another stop and6720 abandoned sim-s.',
        '- Final shared_v10 uses one process/one4117s combined alarm for both unchanged models, no per-model subcaps. CPU gates49cpuQ/50gpuR passed, including shared-budget alarm/debit and completed-physics retention tests.',
        '- Own PIDs only; all terminal owned jobs closed. Durable atomic requests, hashes, observable-history cache equality, committed-pick reproduction and live-prefix gates remain enforced.',
        '',
    ]
    if complete:
        from .analyze import summarize
        result=summarize(MIRROR)
        write(Path('docs/paper_results/c1switch/main_results.json'),result)
        for key,title in [('wm_minus_bb_regret60','WM minus BB'),('wm_minus_best_fixed_regret60','WM minus best fixed')]:
            r=result['primary'][key]
            common.append(f'- {title} 60s regret: {r["mean"]:.3f} veh-s;95% world-cluster CI [{r["ci95"][0]:.3f},{r["ci95"][1]:.3f}].')
        common += ['', '## Statistics','Equal-world three-decision means;2000 paired world-cluster bootstrap draws,numpy seed0,percentile95% intervals. Best fixed reselected inside each draw. Two exploratory primary contrasts;no multiplicity adjustment. WM/BB states differ on-policy. Random comparisons use only its6paired worlds. Full secondary/safety/rank/cell results are in c1switch/main_results.json.']
        with Path('docs/paper_results/c1switch/main_decisions.csv').open('w',newline='',encoding='utf-8') as f:
            rows=result['measured_decisions'];writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else:
        common += ['## Efficacy analysis gate',
            'The registered complete-panel bootstrap, WM-versus-BB/best-fixed primary contrasts, and paired episode comparisons are **not evaluated**. Availability depends on the interrupted scheduler/model requests; treating the retained subset as a paired16-world experiment would be invalid.',
            '', '## Existing paper figure',
            'Recorded WMDATA/WMDATA2 fixed-controller winner-by-condition figure remains c1switch/recorded_winner_by_condition.png (and PDF), with90 recorded worlds and zero new SUMO. All five fixed controllers win somewhere; this is not evidence that WM switching works.',
            '', '## Receipts',
            'tools/c1switch/evidence/V10_COLLECTION.json, V10_PROGRESS.json, V10_PLAN.json, V10_STAGE.json, V10_CPU.json and shared_v10_panel/ contain the terminal accounting and retained scientific receipts. Historical V7_INTERRUPTION_AUDIT.json/V9_FINAL_AUDIT.json preserve preceding interruptions.']
    target.write_text('\n'.join(common)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(target),complete=complete,charged_sim_s=summary['charged_sim_s'],gpu_s=gpu),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('collect','report'))
    globals()[p.parse_args().mode]()

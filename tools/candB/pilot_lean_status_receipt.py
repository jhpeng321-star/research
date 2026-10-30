"""Append actual pre-update profiling evidence; never claim pending startup/updates."""
from datetime import datetime,timezone
import json

from tools.candB.pilot_lean_observe import DIRECTORY,LEDGER,append


if __name__=='__main__':
    path=DIRECTORY/'pilot_lean_FIRST_SCORE_ACK.json'
    if path.exists(): raise RuntimeError('already receipted')
    profile=json.loads((DIRECTORY/'pilot_lean_profile_1793377558.json').read_text())
    observation=json.loads((DIRECTORY/'pilot_lean_latest.json').read_text())
    trainer=observation['owners']['TRAINER_OWNERSHIP.json']
    assert trainer['same_incarnation'] and trainer['state']!='Z'
    first=profile['profile'][0]
    assert first['phase']=='independent_score' and first['update']==191
    doc=dict(label='EXPLORATORY-LEAN',observed_utc=datetime.now(timezone.utc).isoformat(),
        trainer=trainer,first_independent_score=first,
        observed_updates=[r['phase_update'] for r in observation['rows']],
        startup=observation['files'].get('LEAN_STARTUP.json'),
        frozen_baseline_durable='frozen_scoring_baseline.json' in observation['files'],
        GPU_samples=observation['files'].get('GPU.jsonl'),
        phase3='PENDING unless actual pilot_lean_update3_ACK.json exists; sole observer42388 continues')
    with path.open('x') as stream: json.dump(doc,stream,indent=1)
    append(DIRECTORY/'PILOT.md','\n### First actual scoring component timing\n'
        f'- Initial independent K8 score: **{first["seconds"]:.3f}s wall / {first["cpu_seconds"]:.3f}s CPU** '
        '(update191, fresh phase). This is NOT startup or seconds/update.\n'
        f'- Exact trainer incarnation remains live at epoch{observation["epoch"]}; completed phase rows='
        f'{doc["observed_updates"]}; baseline durable={doc["frozen_baseline_durable"]}. '
        'Startup/update1-3 and phase3 verification remain pending actual receipts; '
        'sole durable observer42388 remains responsible. `pilot_lean_FIRST_SCORE_ACK.json`.\n')
    append(LEDGER,'\n- S-PILOTB EXPLORATORY-LEAN FIRST-SCORE/STATUS '+json.dumps(doc)+'\n')
    print(json.dumps(doc))
